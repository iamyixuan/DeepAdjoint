import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.func import jacrev
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ..utils import FNO_losses
from ..utils.losses import Losses


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12532"
    init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method="env://"
    )


class BaseTrainer(ABC):
    def __init__(
        self,
        net,
        optimizer_name,
        loss_name,
        gpu_id=None,
        **kwargs,
    ) -> None:
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam

        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            self.net = net.to(gpu_id)
        else:
            self.net = net

        self.ls_fn = Losses(loss_name)()

        # set up logging
        if kwargs.get("model") is not None:
            model_type = kwargs.get("model")
        if kwargs.get("data_name") is not None:
            data_name = kwargs.get("data_name")

        self.exp_path = f"./experiments/{model_type}-{data_name}-{loss_name}-{optimizer_name}/"

        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            filename=f"{self.exp_path}/log.txt",
            filemode="a",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        if int(os.environ["TRAIN"]) == 1:
            self.net = DDP(
                net, device_ids=[self.gpu_id], find_unused_parameters=True
            )

    @abstractmethod
    def train(self):
        pass

    def predict(self):
        pass


class TrainerSOMA(BaseTrainer):
    """
    Basic trainer class
    """

    def __init__(
        self,
        net,
        optimizer_name,
        loss_name,
        gpu_id,
        dual_train=False,
        **kwargs,
    ):
        super().__init__(net, optimizer_name, loss_name, gpu_id, **kwargs)
        self.net = net
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam

        self.gpu_id = gpu_id

        if os.environ["LOSS"] == "FNO":
            print("Using Lp loss...")
            self.ls_fn = FNO_losses.LpLoss(d=4, p=2)
            # self.ls_fn = FNO_losses.MSE_ACCLoss()
        else:
            self.ls_fn = Losses(loss_name)()()
        self.net = net.to(gpu_id)

        if int(os.environ["TRAIN"]) == 1:
            self.net = DDP(
                net, device_ids=[self.gpu_id], find_unused_parameters=True
            )
        self.dual_train = dual_train

    def train(
        self,
        train,
        val,
        epochs,
        batch_size,
        learning_rate,
        save_freq=10,
        model_name="test",
        mask=None,
        **kwargs,
    ):
        """
        args:
            train: training dataset
            val: validation dataset
        """

        if "load_model" in kwargs:
            if kwargs.get("load_model") is True:
                print("Loading pretrained model...")
                assert "model_path" in kwargs, "Provide a saved model path!"
                model_path = kwargs.get("model_path")
                self.net.module.load_state_dict(torch.load(model_path))

        self.train_vae = kwargs.get("train_vae", False)

        # self.net.to(self.device)
        self.net.train()
        optimizer = self.optimizer(self.net.parameters(), lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR (optimizer , 100 , gamma=0.1, verbose=True)
        if int(os.environ["TRAIN"]) == 1:
            train_loader = DataLoader(
                train, batch_size=batch_size, sampler=DistributedSampler(train)
            )
            val_loader = DataLoader(
                val, batch_size=10, sampler=DistributedSampler(val)
            )
        else:
            train_loader = DataLoader(train, batch_size=batch_size)
            val_loader = DataLoader(val, batch_size=10)

        for val_data in val_loader:
            if self.dual_train:
                x_val, y_val = val_data
                y_val, adj_val = y_val
            else:
                x_val, y_val = val_data

            x_val = x_val.to(self.gpu_id)
            y_val = y_val.to(self.gpu_id)
            break

        print("Starts training...")
        print(
            f"Training set size: {train.__len__()}, validation set size: {val.__len__()}"
        )
        if self.train_vae:
            print("Training the autoencoder...")
        best_val = np.inf
        pbar = tqdm(range(epochs))
        for ep in pbar:
            running_loss = []
            for x_train, y_train in train_loader:
                x_train = x_train.to(self.gpu_id)
                y_train = y_train.to(self.gpu_id)

                optimizer.zero_grad()

                if self.train_vae:
                    out, out_mean, out_log_var = self.net(x_train)
                    batch_loss = self.ls_fn(
                        x_train, out, out_mean, out_log_var
                    )
                else:
                    out = self.net(x_train)
                    batch_loss = self.ls_fn(
                        y_train, out, mask=mask
                    )  # enable mask for Unet and ResNet

                batch_loss.backward()
                running_loss.append(batch_loss.detach().cpu().numpy())
                # running_metrics.append(batch_metric)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
                optimizer.step()
            # scheduler.step()
            with torch.no_grad():
                if self.train_vae:
                    val_out, val_mean, val_log_var = self.net(x_val)
                    val_loss = self.ls_fn(
                        x_val, val_out, val_mean, val_log_var
                    )
                else:
                    val_out = self.net(x_val)
                    val_loss = self.ls_fn(y_val, val_out, mask=mask)
                    # val_metrics = get_metrics(y_val.detach().cpu().numpy(), val_out.detach().cpu().numpy())

            self.logger.info(
                f"Epoch: {ep + 1}, Train Loss: {np.mean(running_loss)}, Val Loss: {val_loss.item()}"
            )
            pbar.set_postfix(
                {"train": np.mean(running_loss), "val": val_loss.item()}
            )
            if self.gpu_id == 0 and val_loss.item() < best_val:
                best_val = val_loss.item()
                torch.save(
                    self.net.module.state_dict(),
                    f"{self.exp_path}/best_model_state.pt",
                )

    def predict(self, test_data, checkpoint=None):
        test_loader = DataLoader(test_data, batch_size=12)
        y_true = []
        y_pred = []
        x_ = []
        self.net.eval()
        self.net.to(self.gpu_id)
        if checkpoint is not None:
            self.net.load_state_dict(torch.load(checkpoint))
        for x, y in tqdm(test_loader):
            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)
            with torch.no_grad():
                pred, mu, var = self.net(x)
                y_true.append(y.detach().cpu().numpy())
                y_pred.append(pred.detach().cpu().numpy())
                x_.append(x.detach().cpu().numpy())
        return y_true, y_pred, x


def pred_rollout(net, gpu_id, test_data, checkpoint=None):
    MAX_STEPS = 29
    ROLLOUT_STEPS = 29  # needs to be less than or equal to the max steps
    test_loader = DataLoader(test_data, batch_size=MAX_STEPS)
    y_true = []
    y_pred = []
    net.eval()
    net.to(gpu_id)
    if checkpoint is not None:
        net.load_state_dict(torch.load(checkpoint))
    for x, y in test_loader:
        # when rollout need to make sure to feed in the true external parameters
        PARAM_IDX = [-1]  # needs to be a list
        x = x.to(gpu_id)
        y = y.to(gpu_id)
        temp_yp = []
        temp_yt = []
        for t in tqdm(range(ROLLOUT_STEPS)):
            with torch.no_grad():
                if t == 0:
                    pred = net(x[0:1])
                    pred = torch.cat([pred, x[1:2, PARAM_IDX, ...]], axis=1)
                    temp_yp.append(pred)
                    temp_yt.append(y[0:1])
                else:
                    pred = net(temp_yp[-1])
                    pred = torch.cat(
                        [pred, x[t : t + 1, PARAM_IDX, ...]], axis=1
                    )
                    temp_yp.append(pred)
                    temp_yt.append(y[t : t + 1])
        y_pred.append([y_p.detach().cpu().numpy() for y_p in temp_yp])
        y_true.append([y_t.detach().cpu().numpy() for y_t in temp_yt])

    return np.array(y_true), np.array(y_pred)


# Obtain the adjoint of NN wrt model input


def rvs_adjoint(model, inputs, axis):
    inputs.requires_grad = True
    model.eval()

    def f(inpus):
        out = model(inputs)
        out = torch.mean(out, dim=axis)
        return out

    jacobian = jacrev(f)(inputs)
    # out_jac = torch.autograd.functional.jacobian(model, inputs)
    return jacobian


# class AdjointTrainer(Trainer):
#     def __init__(
#         self, net, optimizer_name, loss_name, dual_train=False
#     ) -> None:
#         super().__init__(net, optimizer_name, loss_name, dual_train)
#
#     def train(
#         self,
#         train,
#         val,
#         epochs,
#         batch_size,
#         learning_rate,
#         save_freq=10,
#         portion="u",
#     ):
#         self.net.to(self.device)
#         self.net.train()
#         optimizer = self.optimizer(self.net.parameters(), lr=learning_rate)
#         train_loader = DataLoader(train, batch_size=batch_size)
#         val_loader = DataLoader(val, batch_size=100)
#
#         for val in val_loader:
#             x_val, y_val = val
#             y_val_out, y_val_adj = y_val
#
#         print("Starts training...")
#         for ep in range(epochs):
#             running_loss = []
#             for x_train, y_train in tqdm(train_loader):
#                 y_out, y_adj = y_train
#                 optimizer.zero_grad()
#                 out = self.net(x_train)
#                 pred_adj = self.get_grad(x_train, portion=portion)
#                 batch_loss = self.ls_fn(y_out, out) + self.ls_fn(
#                     y_adj, pred_adj
#                 )
#                 batch_loss.backward()
#                 running_loss.append(batch_loss.detach().cpu().numpy())
#                 optimizer.step()
#
#             pred_val_out = self.net(x_val)
#             val_pred_adj = self.get_grad(x_val, portion=portion)
#             adj_ls_val = self.ls_fn(y_val_adj, val_pred_adj)
#             val_loss = self.ls_fn(y_val_out, pred_val_out) + adj_ls_val
#
#             self.logger.record("epoch", ep + 1)
#             self.logger.record("train_loss", np.mean(running_loss))
#             self.logger.record("val_loss", val_loss.item())
#             self.logger.record("val_adj_loss", adj_ls_val.item())
#             self.logger.print()
#             torch.save(
#                 self.net.state_dict(),
#                 f"./checkpoints/{self.now}/model_saved_ep_{ep}",
#             )
#
#         self.logger.finish()
#
#     def get_grad(
#         self, x, portion="u"
#     ):  # x shape [batch, in_dim]; output shape [batch, out_dim]
#         x = torch.tensor(x, requires_grad=True)
#
#         def compute_grad(x, net):
#             x = x.unsqueeze(0)
#             sum_square = 0.5 * torch.sum(net(x))
#             grad = torch.autograd.grad(sum_square, x, retain_graph=True)[0]
#             return grad
#
#         grad = [compute_grad(x[i], self.net) for i in range(len(x))]
#         # grad = zip(*grad)
#         grad = torch.concat(grad)
#         if portion == "u":
#             return grad[:, :80]
#         elif portion == "all":
#             return grad
#         elif portion == "p":
#             return grad[:, -79:]
#         else:
#             raise Exception(f'"{portion}" is not in the list...')
