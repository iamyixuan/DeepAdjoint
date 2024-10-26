import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"
mpi4py.rc.recv_mprobe = False


import logging
import os

import numpy as np
from data import SOMAdata
from deephyper.evaluator import Evaluator, RunningJob, profile
from deephyper.hpo import CBO, HpProblem
from metrics import NLLLoss, QuantileLoss
from model import *
from modulus.models.fno import FNO
from mpi4py import MPI
from sklearn.preprocessing import QuantileTransformer
from train_model_nll import FNO_MU_STD as FNO_nll
from train_model_quantile import FNO_QR as FNO_quantile

# Avoid some errors on some MPI implementations


problem = HpProblem()
activations = [
    "relu",
    "leaky_relu",
    "prelu",
    "relu6",
    "elu",
    "selu",
    "silu",
    "gelu",
    "sigmoid",
    "logsigmoid",
    "softplus",
    "softshrink",
    "softsign",
    "tanh",
    "tanhshrink",
    "threshold",
    "hardtanh",
    "identity",
    "squareplus",
]
losses = ["nll", "quantile"]
optimizers = ["Adadelta", "Adagrad", "Adam", "AdamW", "RMSprop", "SGD"]
schedulers = ["cosine", "step"]

problem.add_hyperparameter((1, 16), "padding", default_value=8)
problem.add_hyperparameter(
    ["constant", "reflect", "replicate", "circular"],
    "padding_type",
    default_value="constant",
)
problem.add_hyperparameter([True, False], "coord_feat", default_value=True)
problem.add_hyperparameter(activations, "lift_act", default_value="gelu")
problem.add_hyperparameter((2, 16), "num_FNO", default_value=4)
problem.add_hyperparameter((2, 32), "num_modes", default_value=16)
problem.add_hyperparameter((2, 16), "latent_ch", default_value=8)
problem.add_hyperparameter((1, 16), "num_projs", default_value=1)
problem.add_hyperparameter((2, 16), "proj_size", default_value=8)
problem.add_hyperparameter(activations, "proj_act", default_value="silu")

problem.add_hyperparameter(optimizers, "optimizer", default_value="Adam")
problem.add_hyperparameter(
    (1e-6, 1e-2, "log-uniform"), "lr", default_value=1e-3
)
problem.add_hyperparameter((0.0, 0.1), "weight_decay", default_value=0)
problem.add_hyperparameter((8, 64), "batch_size", default_value=16)
problem.add_hyperparameter(losses, "loss", default_value="nll")
# the activation parameter for the standard deiviation
# the activation as HP only applies to the hidden layers


@profile
def run(job: RunningJob):
    config = job.parameters.copy()
    job_id = job.id

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if config["loss"] == "nll":
        net = FNO_nll(
            in_channels=5,
            out_channels=4 * 2,
            decoder_layers=config["num_projs"],
            decoder_layer_size=config["proj_size"],
            decoder_activation_fn=config["proj_act"],
            dimension=2,
            latent_channels=config["latent_ch"],
            num_fno_layers=config["num_FNO"],
            num_fno_modes=int(config["num_modes"]),
            padding=config["padding"],
            padding_type=config["padding_type"],
            activation_fn=config["lift_act"],
            coord_features=config["coord_feat"],
        )
    elif config["loss"] == "quantile":
        net = FNO_quantile(
            in_channels=5,
            out_channels=4 * 3,
            decoder_layers=config["num_projs"],
            decoder_layer_size=config["proj_size"],
            decoder_activation_fn=config["proj_act"],
            dimension=2,
            latent_channels=config["latent_ch"],
            num_fno_layers=config["num_FNO"],
            num_fno_modes=int(config["num_modes"]),
            padding=config["padding"],
            padding_type=config["padding_type"],
            activation_fn=config["lift_act"],
            coord_features=config["coord_feat"],
        )

    trainset = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
        "train",
        y_noise=True,
        transform=True,
    )
    valset = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
        "val",
        y_noise=True,
        transform=True,
    )
    # testSet = SOMAdata(
    #     "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5",
    #     "test",
    #     transform=True,
    # )

    # declare loss functions
    if config["loss"] == "nll":
        TrainLossFn = NLLLoss()
        ValLossFn = NLLLoss()
    elif config["loss"] == "quantile":
        TrainLossFn = QuantileLoss()
        ValLossFn = QuantileLoss()

    trainer = Trainer(
        model=net,
        TrainLossFn=TrainLossFn,
        ValLossFn=ValLossFn,
        loss=config["loss"],
    )

    trainLoader = DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True
    )
    valLoader = DataLoader(valset, batch_size=valset.__len__())

    try:
        log, best_model = trainer.train(
            trainLoader=trainLoader,
            valLoader=valLoader,
            epochs=500,  # with early stopping
            optimizer=config["optimizer"],
            learningRate=config["lr"],
            # scheduler=config['scheduler'],
            weight_decay=config["weight_decay"],
        )

        # save the model
        model_dir = "/pscratch/sd/y/yixuans/saved_models_hpo/"
        os.makedirs(model_dir, exist_ok=True)

        torch.save(
            best_model.state_dict(),
            f"{model_dir}/{job_id}.pth",
        )

        obj_0 = float(log.logger["Val_ll"][-1].numpy())
        obj_1 = -float(log.logger["Val_MSE"][-1])

        ll_objective = "F" if np.isnan(obj_0) or np.isinf(obj_0) else obj_0
        mse_objective = "F" if np.isnan(obj_1) or np.isinf(obj_1) else obj_1

        mse_objective = float(log.logger["Val_MSE"][-1])  # minimize mse score
        objective = [ll_objective, -mse_objective]
        trainLoss = log.logger["TrainLoss"]
        valLoss = log.logger["ValLoss"]
    except Exception as e:
        objective = ["F", "F"]
        trainLoss = 0
        valLoss = 0
        print(f"Exception: {e}")

    # print("objective:", objective)
    return {
        "objective": objective,
        "metadata": {
            "TrainLoss": trainLoss,
            "valLoss": valLoss,
        },
    }


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()

    log_dir = "./hpo_logs/hpo_nll_quantile_checkpointing/"

    if not MPI.Is_initialized():
        MPI.Init_thread()

        if MPI.COMM_WORLD.Get_rank() == 0:
            # Only the root rank will create the directory
            print("creating logging................")
            os.makedirs(log_dir, exist_ok=True)

        MPI.COMM_WORLD.barrier()  # Synchronize all processes
        logging.basicConfig(
            filename=os.path.join(
                log_dir, f"deephyper.{MPI.COMM_WORLD.Get_rank()}.log"
            ),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )

    with Evaluator.create(
        run,
        method="mpicomm",
    ) as evaluator:
        if evaluator is not None:
            search = CBO(
                problem,
                evaluator,
                moo_scalarization_strategy="Chebyshev",
                moo_scalarization_weight="random",
                objective_scaler="quantile-uniform",
                acq_func="UCB",
                surrogate_mode="DUMMY",
                multi_point_strategy="qUCB",
                n_jobs=8,
                verbose=0,
                initial_points=[problem.default_configuration],
                log_dir=log_dir,
            )
            results = search.search(max_evals=1000)
            results.to_csv("results-100nodes-stopper.csv")
