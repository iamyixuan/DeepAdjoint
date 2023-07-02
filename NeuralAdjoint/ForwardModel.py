import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import get_activation, BurgersDataset, combine_burgers_data, split_data
from torch.utils.data import DataLoader


class FFN(nn.Module):
    def __init__(self, LayerSizes, activation):
        super(FFN, self).__init__()
        self.layers = nn.ModuleList()
        act = get_activation(activation)
        for i in range(len(LayerSizes) - 2):
            self.layers.append(nn.Linear(LayerSizes[i], LayerSizes[i + 1]))
            self.layers.append(act)
        self.out_layer = nn.Linear(LayerSizes[-2], LayerSizes[-1])

    def forward(self, x):
        # residual = x[:, :-1]
        for l in self.layers:
            x = l(x)
        out = self.out_layer(x)  # + residual
        return out


class Trainer:
    def __init__(self, net, numEpoch, lr, batch_size):
        self.net = net
        self.numEpcohs = numEpoch
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def train(self, trainSet, valSet):
        trainLoader = DataLoader(trainSet, batch_size=self.batch_size, shuffle=True)
        valLoader = DataLoader(valSet, batch_size=valSet.__len__(), shuffle=False)
        logger = {"trainLoss": [], "valLoss": []}

        self.net.train()
        for epoch in range(self.numEpcohs):
            trainRunningLoss = []
            for train_batch in trainLoader:
                self.optimizer.zero_grad()
                x_batch, y_batch = train_batch
                out_batch = self.net(x_batch)
                loss_batch = self.loss(y_batch, out_batch)
                loss_batch.backward()
                trainRunningLoss.append(loss_batch.item())
                self.optimizer.step()
            for val_data in valLoader:
                x_val, y_val = val_data
                val_out = self.net(x_val)
                val_loss = self.loss(y_val, val_out)

            logger["trainLoss"].append(np.mean(trainRunningLoss))
            logger["valLoss"].append(val_loss.item())

            print(
                f"Epoch {epoch+1}: Training Loss: {logger['trainLoss'][-1]:.4e}; Validation Loss: {logger['valLoss'][-1]:.4e}"
            )

    def eval(self, net, testSet):
        net.eval()
        testLoader = DataLoader(testSet, batch_size=testSet.__len__())
        for test in testLoader:
            x_test, y_test = test
        pred = self.net(x_test)

    def pred_time_roll_out(self, net, initCond, time):
        net.eval()
        nu = initCond[:, -1].reshape(-1, 1)
        pred = [initCond]
        for t in range(time):
            out = net(pred[-1])
            pred.append(
                torch.cat([out, nu], dim=1)
            )  # adding the viscosity as part of the input
        return pred


if __name__ == "__main__":
    import pickle
    from utils import load_burgers_data

    torch.manual_seed(42)
    device = torch.device("cpu")
    x, y, adj = combine_burgers_data("../AdjointMatchingNN/Data/mixed_nu/")
    train, val, test = split_data(x, y, adj, shuffle_all=True)
    trainSet = BurgersDataset(train["x"], train["y"], device)
    valSet = BurgersDataset(val["x"], val["y"], device)
    testSet = BurgersDataset(test["x"], test["y"], device)

    net = FFN([129] + [200] * 5 + [128], "relu")
    trainer = Trainer(net.to(device), 1000, 5e-5, 128)
    # trainer.train(trainSet, valSet)
    # torch.save(net.state_dict(), "./saved/forwardModel_2.pt")
    # -----------------------------------------------------------
    net.load_state_dict(torch.load("./saved/forwardModel_2.pt"))

    # load data
    with open("../AdjointMatchingNN/Data/mixed_nu/0.000607-nu.pkl", "rb") as f:
        data = pickle.load(f)
    x, y, adj = load_burgers_data(data)
    x = torch.from_numpy(x).float()

    # load the recovered initial condition
    # with open("./logs/0_NuOnlylogger.pkl", "rb") as f:
    #     logger = pickle.load(f)

    # initCond = logger["solution"][0].reshape(1, -1)
    # initCond = torch.from_numpy(initCond).float()

    pred = trainer.pred_time_roll_out(net, x[0].reshape(1, -1), 200)
    pred = np.array([i.detach().numpy() for i in pred])
    pred = np.squeeze(pred)

    error = np.mean((pred[1:, :-1] - y) ** 2)

    print(error)
    # plot out the true and predictions

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1)
    vmax = np.max(y)
    vmin = np.min(y)
    ax[0].set_box_aspect(0.5)
    ax[0].set_title("True")
    ax[0].imshow(
        y.T,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=[0, 1, 0, 1],
        transform=ax[0].transAxes,
    )
    ax[0].set_axis_off()
    ax[1].set_box_aspect(0.5)
    ax[1].set_axis_off()
    ax[1].set_title("Predicted")
    im = ax[1].imshow(
        pred.T,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=[0, 1, 0, 1],
        transform=ax[1].transAxes,
    )
    fig.tight_layout()
    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.show()
