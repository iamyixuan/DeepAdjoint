from typing import Any
import torch 
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from ..utils.losses import Losses
from ..utils.logger import Logger

class Trainer:
    '''
    Basic trainer class
    '''
    def __init__(self, net, optimizer_name, loss_name, dual_train=False) -> None:
        self.net = net
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam
        
        self.ls_fn = Losses(loss_name)()
        self.logger = Logger()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use CUDA device
        else:
            self.device = torch.device("cpu")

        self.dual_train = dual_train

        self.now = datetime.now().strftime('%Y-%m-%d')
        if not os.path.exists(f'./checkpoints/{self.now}/'):
            print("Creating model saving folder...")
            os.makedirs(f'./checkpoints/{self.now}/')

    def train(self, train,
              val,
              epochs,
              batch_size,
              learning_rate,
              save_freq=10):
        '''
        args:
            train: training dataset
            val: validation dataset
        '''
        self.net.to(self.device)
        self.net.train()
        optimizer = self.optimizer(self.net.parameters(), lr=learning_rate)
        train_loader = DataLoader(train, batch_size=batch_size)
        val_loader = DataLoader(val, batch_size=10)

        for val in val_loader:
            if self.dual_train:
                x_val, y_val = val
                y_val, adj_val = y_val
            else:
                x_val, y_val = val
            break

        print("Starts training...")
        for ep in range(epochs):
            running_loss = []
            for x_train, y_train in tqdm(train_loader):
                if self.dual_train:
                    y_train, adj_train = y_train
                optimizer.zero_grad()
                out = self.net(x_train)
                if self.dual_train:
                    batch_loss = self.ls_fn(y_train, out, adj_train)
                else:
                    batch_loss = self.ls_fn(y_train, out)
                batch_loss.backward()
                running_loss.append(batch_loss.detach().cpu().numpy())
                optimizer.step()
            with torch.no_grad():
                val_out = self.net(x_val) 
                if self.dual_train:
                    val_loss = self.ls_fn(y_val, val_out, adj_val)
                else:
                    val_loss = self.ls_fn(y_val, val_out)
            self.logger.record('epoch', ep+1)
            self.logger.record('train_loss', np.mean(running_loss))
            self.logger.record('val_loss', val_loss.item())
            self.logger.print()
            torch.save(self.net.state_dict(), f'./checkpoints/{self.now}/model_saved_ep_{ep}')

        self.logger.finish()
            
    def eval(self, test_set):
        self.net.eval()
        x_test, y_test = test_set
        pred_test = self.net(x_test)


class AdjointTrainer(Trainer):
    def __init__(self, net, optimizer_name, loss_name, dual_train=False) -> None:
        super().__init__(net, optimizer_name, loss_name, dual_train)
        
    def train(self, train, val, epochs, batch_size, learning_rate, save_freq=10, portion='u'):
        self.net.to(self.device)
        self.net.train()
        optimizer = self.optimizer(self.net.parameters(), lr=learning_rate)
        train_loader = DataLoader(train, batch_size=batch_size)
        val_loader = DataLoader(val, batch_size=100)

        for val in val_loader:
            x_val, y_val = val
            y_val_out, y_val_adj = y_val

        print("Starts training...")
        for ep in range(epochs):
            running_loss = []
            for x_train, y_train in tqdm(train_loader):
                y_out, y_adj = y_train
                optimizer.zero_grad()
                out = self.net(x_train)
                pred_adj = self.get_grad(x_train, portion=portion)
                batch_loss = self.ls_fn(y_out, out) + self.ls_fn(y_adj, pred_adj)
                batch_loss.backward()
                running_loss.append(batch_loss.detach().cpu().numpy())
                optimizer.step()
          
            pred_val_out = self.net(x_val) 
            val_pred_adj = self.get_grad(x_val, portion=portion)
            adj_ls_val = self.ls_fn(y_val_adj, val_pred_adj)
            val_loss = self.ls_fn(y_val_out, pred_val_out) + adj_ls_val

            self.logger.record('epoch', ep+1)
            self.logger.record('train_loss', np.mean(running_loss))
            self.logger.record('val_loss', val_loss.item())
            self.logger.record('val_adj_loss', adj_ls_val.item())
            self.logger.print()
            torch.save(self.net.state_dict(), f'./checkpoints/{self.now}/model_saved_ep_{ep}')

        self.logger.finish()


    def get_grad(self, x, portion='u'): # x shape [batch, in_dim]; output shape [batch, out_dim]
        x = torch.tensor(x, requires_grad=True)
        def compute_grad(x, net):
            x = x.unsqueeze(0) 
            sum_square = 0.5 * torch.sum(net(x))
            grad = torch.autograd.grad(sum_square, x, retain_graph=True)[0]
            return grad
        grad = [compute_grad(x[i], self.net) for i in range(len(x))]
        # grad = zip(*grad)
        grad = torch.concat(grad)
        if portion == 'u':
            return grad[:, :80]
        elif portion == 'all':
            return grad
        elif portion == 'p':
            return grad[:, -79:]
        else:
            raise Exception(f'"{portion}" is not in the list...')
        
    
        


class MultiStepTrainer(Trainer):
    def __init__(self, net, epochs) -> None:
        super(MultiStepTrainer, self).__init__(net, epochs)
        pass