import torch
import pickle
import numpy as np
from utils import load_burgers_data, combine_burgers_data, split_data
from ForwardModel import FFN, Trainer

class NeuralAdjoint:
    def __init__(self, forwardNet, initGuess, timesteps) -> None:
        self.forwardNet = forwardNet
        self.initGuess = initGuess 
        self.loss_fn = torch.nn.MSELoss()
        self.timesteps = timesteps
    
    def rollout(self, x):
        self.forwardNet.eval()
        nu = x[:, -1].reshape(-1, 1)
        pred = [x]
        for t in range(self.timesteps):
            out = self.forwardNet(pred[-1]) 
            pred.append(torch.cat([out, nu], dim=1)) # adding the viscosity as part of the input
        return out

    def resimulation_loss(self, x, y):
        yHat = self.rollout(x)
        lossValue = self.loss_fn(y, yHat)
        return lossValue

    def boundary_loss(self, x, mu_x, R_x):
        '''Boundary loss for getting the inverse solution
        x: the current solution.
        mu_x: the mean of the input variables from the training set
        R_x: the range of the input variable
        ''' 
        relu = torch.nn.ReLU()
        L_bnd = relu(torch.abs(x - mu_x) + 0.5 * R_x)
        return L_bnd.sum()

    def loss(self, x, y, mu_x, R_x):
        resim_loss = self.resimulation_loss(x, y) 
        bn_loss = self.boundary_loss(x, mu_x, R_x)
        return resim_loss + bn_loss, (resim_loss, bn_loss)
    
    def inverse_solve(self, y, mu_x, R_x, numIter, log_name=None):
        # Trace the graph
        logger = {'solution': [], 'loss': []}
        x = torch.tensor(self.initGuess, requires_grad=True)
        # Wrap as a trainable parameter
        x = torch.nn.Parameter(x)
        # Set the optimizer: optimize over the input.
        optimizer = torch.optim.Adam([x], lr=0.001)
        for i in range(numIter):
            optimizer.zero_grad()
            loss, (resim_loss, bn_loss) = self.loss(x, y, mu_x, R_x)
            loss.backward()
            sol = x.detach().numpy().copy()
            print(f'Iteration {i + 1}, the resimulation loss {resim_loss.item():.4e} and boundary loss {bn_loss.item():.4e}')
            logger['loss'].append(loss.item())
            optimizer.step()
            logger['solution'].append(sol)
            with open('./logs/' + log_name +'logger.pkl', 'wb') as f:
                pickle.dump(logger, f)
        return x

            
def main():
    net = FFN([129] + [200]*5 + [128], 'tanh')
    trainer = Trainer(net, 1000, 0.0001, 64)
    net.load_state_dict(torch.load('./saved/forwardModel.pt'))

    with open('../AdjointMatchingNN/Data/mixed_nu/0.001674-nu.pkl', 'rb') as f:
        data = pickle.load(f)
    x, y, adj = load_burgers_data(data)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()


    x_all, y_all, adj_all = combine_burgers_data('../AdjointMatchingNN/Data/mixed_nu/')
    train, val, test = split_data(x_all, y_all, adj_all, shuffle_all=True)
    inputs = torch.from_numpy(train['x']).float()
    mu_x = torch.mean(inputs, dim=0)
    std_x = torch.std(inputs, dim=0)
    R_x = torch.max(inputs, dim=0)[0] - torch.min(inputs, dim=0)[0]

    
    initCond = x[0, :].reshape(1, -1)


    for n in range(1):
        initGuess = np.random.rand(1, 129)  * std_x.numpy() + mu_x.numpy()
        initGuess = torch.from_numpy(initGuess).float()

        inverse_model = NeuralAdjoint(net, initGuess, 100)
        pred = inverse_model.inverse_solve(y[99:100, :], mu_x, R_x, 2000, str(n) + '_')



if __name__ == '__main__':
    main()
