from deep_adjoint.model.ForwardSurrogate import ResnetSurrogate
from deep_adjoint.utils.data import MultiStepData
from deep_adjoint.train.trainer import Trainer


def main(args):
    net = ResnetSurrogate(
        time_steps=args.time_steps,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        h_dim=args.h_dim,
    )

    dual_trainer = Trainer(
        net=net, optimizer_name="Adam", loss_name="Lag", dual_train=True
    )

    train_data = MultiStepData()
    val_data = MultiStepData(mode="val")

    dual_trainer.train(
        train=train_data, val=val_data, epochs=100, batch_size=8, learning_rate=0.001
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-time_steps", type=int, default=200)
    parser.add_argument("-in_dim", type=int, default=129)
    parser.add_argument("-out_dim", type=int, default=128)
    parser.add_argument("-h_dim", type=int, default=150)

    args = parser.parse_args()
    main(args)
