from deep_adjoint.model.ForwardSurrogate import FNN
from deep_adjoint.utils.data import GlacierData
from deep_adjoint.train.trainer import AdjointTrainer


def main(args):
    train_data = GlacierData(
        path="./data/sv_cf_squared_velocity.npz", mode="train", portion="p"
    )
    val_data = GlacierData(
        path="./data/sv_cf_squared_velocity.npz", mode="val", portion="p"
    )
    net = FNN(in_dim=args.in_dim, out_dim=args.out_dim, layer_sizes=args.layer_sizes)

    trainer = AdjointTrainer(
        net=net,
        optimizer_name="Adam",
        loss_name="MSE",
    )
    trainer.train(
        train=train_data,
        val=val_data,
        epochs=2000,
        batch_size=512,
        learning_rate=0.01,
        save_freq=50,
        portion="p",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dim", type=int, default=159)
    parser.add_argument("-out_dim", type=int, default=80)
    parser.add_argument("-layer_sizes", type=int, nargs="+", default=[20, 20, 20])

    args = parser.parse_args()

    print(args.layer_sizes)

    main(args)
