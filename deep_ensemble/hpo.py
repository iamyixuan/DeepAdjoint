import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"
mpi4py.rc.recv_mprobe = False


import logging
import os

from deephyper.evaluator import Evaluator, RunningJob, profile
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from modulus.models.fno import FNO
from mpi4py import MPI
from sklearn.preprocessing import QuantileTransformer

from metrics import MSE_ACCLoss, anomalyCorrelationCoef
from model import *

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
problem.add_hyperparameter((2, 32), "num_FNO", default_value=4)
problem.add_hyperparameter((2, 32), "num_modes", default_value=16)
problem.add_hyperparameter((2, 64), "latent_ch", default_value=32)
problem.add_hyperparameter((1, 16), "num_projs", default_value=1)
problem.add_hyperparameter((2, 32), "proj_size", default_value=32)
problem.add_hyperparameter(activations, "proj_act", default_value="silu")

problem.add_hyperparameter(optimizers, "optimizer", default_value="Adam")
problem.add_hyperparameter(
    (1e-6, 1e-2, "log-uniform"), "lr", default_value=1e-3
)
problem.add_hyperparameter((0.0, 0.1), "weight_decay", default_value=0)
# problem.add_hyperparameter(schedulers, "scheduler", default_value='cosine')
problem.add_hyperparameter((8, 256), "batch_size", default_value=128)
problem.add_hyperparameter((0.0, 1.0), "alpha", default_value=0.5)
# problem.add_hyperparameter((20, 100), 'epochs', default_value=20)


def rollout_metric_avg(true, pred, metric):
    """
    True and pred are from a single rollout - single sequence
    """
    avg_score = []
    for i in range(len(true)):
        scores = []
        for t in range(29):
            cur_score = np.mean(
                metric(
                    true[i][t].detach().numpy(), pred[i][t].detach().numpy()
                )
            )
            scores.append(cur_score)
        avg_score.append(scores)

    avg_score = np.array(avg_score).mean(axis=0)

    return avg_score


@profile
def run(job: RunningJob):
    config = job.parameters.copy()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    net = FNO(
        in_channels=6,
        out_channels=5,
        decoder_layers=config["num_projs"],
        decoder_layer_size=config["proj_size"],
        decoder_activation_fn=config["proj_act"],
        dimension=2,
        latent_channels=config["latent_ch"],
        num_fno_layers=config["num_FNO"],
        num_fno_modes=config["num_modes"],
        padding=config["padding"],
        padding_type=config["padding_type"],
        activation_fn=config["lift_act"],
        coord_features=config["coord_feat"],
    )

    trainset = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5",
        "train",
        transform=True,
    )
    valset = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5",
        "val",
        transform=True,
    )
    testSet = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5",
        "test",
        transform=True,
    )

    # declare loss functions
    TrainLossFn = MSE_ACCLoss(alpha=config["alpha"])
    # TrainLossFn = torch.nn.MSELoss()

    ValLossFn = torch.nn.MSELoss()

    trainer = Trainer(model=net, TrainLossFn=TrainLossFn, ValLossFn=ValLossFn)

    trainLoader = DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True
    )
    valLoader = DataLoader(valset, batch_size=valset.__len__())
    testLoader = DataLoader(testSet, batch_size=testSet.__len__())

    rollout_val_data = DataLoader(valset, batch_size=1)
    rollout_test_data = DataLoader(testSet, batch_size=1)

    log = trainer.train(
        trainLoader=trainLoader,
        valLoader=valLoader,
        epochs=30,
        optimizer=config["optimizer"],
        learningRate=config["lr"],
        # scheduler=config['scheduler'],
        weight_decay=config["weight_decay"],
    )

    testLoss, inferenceTime = trainer.test(testLoader)

    # get rollout objectives
    rollout_val = trainer.rollout(rollout_val_data)
    rollout_test = trainer.rollout(rollout_test_data)

    val_rollout_metric = np.mean(
        rollout_metric_avg(
            rollout_val["true"], rollout_val["pred"], anomalyCorrelationCoef
        )
    )
    test_rollout_metric = np.mean(
        rollout_metric_avg(
            rollout_test["true"], rollout_test["pred"], anomalyCorrelationCoef
        )
    )

    trainLoss = log.logger["TrainLoss"]
    valLoss = log.logger["ValLoss"]
    valACC = log.logger["ValACC"]  # 5 dim vector, take the avg
    valR2 = log.logger["ValR2"]
    NumTrainableParams = log.logger["NumTrainableParams"]

    # Val MSE loss maximize neg value
    objective_0 = (
        "F"
        if np.isnan(valLoss).any() or np.isinf(valLoss).any()
        else -valLoss[-1]
    )

    # Val ACC score maximize
    objective_1 = (
        "F"
        if np.isnan(valACC).any() or np.isinf(valACC).any()
        else np.mean(valACC[-1])
    )  # lower the better so taking the negative to maximize

    # objective_1 = (
    #      "F" if np.isnan(inferenceTime).any() or np.isinf(inferenceTime).any() else -inferenceTime
    # )# lower the better so taking the negative to maximize

    # objective_2 = (
    #    "F" if np.isnan(val_rollout_metric).any() or np.isinf(val_rollout_metric).any() else val_rollout_metric
    # ) # improve rollout performance when training for one step forward

    objective = [objective_0, objective_1]
    # objective = [objective_0]

    return {
        "objective": objective,
        "metadata": {
            "TrainLoss": trainLoss,
            "valLoss": valLoss,
            "ValR2": valR2[0].tolist(),
            "valACC": valACC[0].tolist(),
            "testLoss": testLoss,
            "testRollout": test_rollout_metric,
            "inferenceTime": inferenceTime,
        },
    }


def trainWithBest(config):
    net = FNO(
        in_channels=6,
        out_channels=5,
        decoder_layers=config["p:num_projs"],
        decoder_layer_size=config["p:proj_size"],
        decoder_activation_fn=config["p:proj_act"],
        dimension=2,
        latent_channels=config["p:latent_ch"],
        num_fno_layers=config["p:num_FNO"],
        num_fno_modes=config["p:num_modes"],
        padding=config["p:padding"],
        padding_type=config["p:padding_type"],
        activation_fn=config["p:lift_act"],
        coord_features=config["p:coord_feat"],
    )

    trainset = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5",
        "train",
        transform=True,
    )
    valset = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5",
        "val",
        transform=True,
    )
    testSet = SOMAdata(
        "/home/iamyixuan/work/ImPACTs/HPO/datasets/GM-prog-var-surface.hdf5",
        "test",
        transform=True,
    )

    trainer = Trainer(model=net, lossFn=torch.nn.MSELoss())

    trainLoader = DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True
    )
    valLoader = DataLoader(valset, batch_size=valset.__len__())
    testLoader = DataLoader(testSet, batch_size=testSet.__len__())

    log = trainer.train(
        trainLoader=trainLoader,
        valLoader=valLoader,
        epochs=3,
        optimizer=config["optimizer"],
        learningRate=config["lr"],
        # scheduler=config['scheduler'],
        weight_decay=config["weight_decay"],
    )

    pred, true = trainer.predict(testLoader)

    return {"true": true, "pred": pred}


def getBestConfig(df):
    # load the result.csv datafraem
    # remove 'F'
    df = df[df["objective_0"] != "F"]
    df = df[df["objective_2"] != "F"]
    # take out the pareto front
    # df = df[df['pareto_efficient']==True]

    # convert to float and take the negative
    qt = QuantileTransformer()

    df["objective_0"] = qt.fit_transform(
        df["objective_0"].astype(float).values.reshape(-1, 1)
    )
    # df['objective_1'] = qt.fit_transform(-df['objective_1'].astype(float).values.reshape(-1, 1))
    # df['objective_2'] = qt.fit_transform(-df['objective_2'].astype(float).values.reshape(-1, 1))

    # Pick the best objectives
    # quantile transformation

    # min_row_index = (df['objective_0'] + df['objective_1'] + df['objective_2']).idxmin()

    min_row_index = (df["objective_0"]).idxmax()
    df = df.rename(columns=lambda x: x.replace("p:", ""))
    return df.loc[min_row_index]

    # get hyperparmeter configs


if __name__ == "__main__":
    from datetime import datetime

    current_time = datetime.now().strftime("%Y%m%d%H")

    log_dir = f"/home/iamyixuan/work/ImPACTs/HPO/{current_time}_DoubleStop100Nodes-MSE_ACC"

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
                verbose=1,
                initial_points=[problem.default_configuration],
                log_dir=log_dir,
            )
            results = search.search(max_evals=2000)
            results.to_csv("results-100nodes-stopper.csv")
