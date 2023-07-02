import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ..utils.data import split_data
from torch.utils.data import Dataset, DataLoader
from torch.autograd.functional import jacobian
from torch.func import jacrev, vmap


class Data(Dataset):
    def __init__(self, data):
        super(Data, self).__init__()
        self.data = data

    def __len__(self):
        return self.data["x"].shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data["x"][idx]).float(), (
            torch.from_numpy(self.data["y"][idx]).float(),
            torch.from_numpy(self.data["adj"][idx]).float(),
        )


class ResidualBlock(nn.Module):
    """Residual block for fully connected layer"""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()
        self.residual = nn.Sequential()
        if out_dim != in_dim:
            self.residual = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.bn2(out)

        out += self.residual(x)
        out = self.act(out)
        return out


class OneStepNet(nn.Module):
    def __init__(self, layerSizes, hidden_dim, activation):
        """Time stepping predictor with residual connections
        args:
            layersSizes: the sizes of all layers in a list, including the input and output layers.
            hidden_dim: the hidden dimension for the residual block.
            activation: the activation function.
        """
        super(OneStepNet, self).__init__()
        self.layers = nn.ModuleList()
        self.act = nn.ReLU()
        for i in range(len(layerSizes) - 1):
            resLayer = ResidualBlock(layerSizes[i], hidden_dim, layerSizes[i + 1])
            self.layers.append(resLayer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, x):
        pass


class Trainer:
    def __init__(self, net, num_epochs, batch_size, learning_rate, optimizer):
        self.num_epochs = num_epochs
        self.net = net  # torch.compile(net)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.ls = nn.MSELoss()
        self.logger = {"train_loss": [], "val_loss": []}

    def loss(self, x, true, alpha):
        x.requires_grad = True
        pred = self.net(x)
        mse_loss = self.ls(pred, true[0])
        # print('x.shape', x.shape)
        # J_pred = jacobian(self.net, x, vectorize=True, create_graph=True)
        # J_pred = torch.sum(J_pred, dim=2)
        # print(J_pred.shape)
        adj_loss = 0  # self.ls(true[1], J_pred)
        totalLoss = mse_loss + alpha * adj_loss
        return totalLoss, (mse_loss, adj_loss)

    def _train(self, train, val, alpha):
        train_loader = DataLoader(train, batch_size=self.batch_size)
        val_loader = DataLoader(val, batch_size=val.__len__())
        for val in val_loader:
            x_val, y_val = val

        optimizer = self.optimizer(self.net.parameters(), lr=self.learning_rate)
        self.net.train()
        for epoch in range(self.num_epochs):
            running_losses = np.array([0, 0, 0])
            for (x_batch, y_batch) in tqdm(train_loader):
                optimizer.zero_grad()
                train_batch_loss, aux_loss = self.loss(x_batch, y_batch, alpha)
                running_losses[0] += train_batch_loss.item()
                running_losses[1] += aux_loss[0]
                running_losses[2] += aux_loss[1]
                train_batch_loss.backward()
                optimizer.step()
            val_loss, aux_val = self.loss(x_val, y_val, 1)
            self.logger["train_loss"].append(
                np.mean(running_losses.reshape(-1, 1), axis=1)
            )
            self.logger["val_loss"].append(
                np.array(
                    [val_loss.detach().numpy(), aux_val[0].detach().numpy(), aux_val[1]]
                )
            )

            trainTotLoss, trainMSELoss, trainAdjLoss = self.logger["train_loss"][-1]
            valTotLoss, valMSELoss, valAdjLoss = self.logger["val_loss"][-1]
            print(
                f"Epoch {epoch+1}; Training loss: [{trainTotLoss:.3f}, {trainMSELoss:.3f}, {trainAdjLoss:.3f}]; Validation loss: [{valTotLoss:.3f}, {valMSELoss:.3f}, {valAdjLoss:.3f}]"
            )

    def predict(self, x):
        self.net.eval()
        # pred = self.net
        pass


if __name__ == "__main__":
    layers = [159] + [256] * 10 + [80]
    data = np.load("../data/vary_beta.dat.npz")
    inputs = data["inputs"]
    uout = data["uout"]
    jrav = data["jrav"]

    train, val, test = split_data(inputs, uout, jrav, shuffle_all=True)
    train_set = Data(train)
    val_set = Data(val)

    net = OneStepNet(layerSizes=layers, hidden_dim=256, activation=nn.ReLU())
    trainer = Trainer(
        net=net,
        num_epochs=1000,
        batch_size=64,
        learning_rate=0.001,
        optimizer=torch.optim.Adam,
    )
    trainer._train(train_set, val_set, 1)
