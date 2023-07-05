import torch 
import numpy as np
from torch.utils.data import DataLoader
from ..utils.losses import Losses
from ..utils.logger import Logger

class Trainer:
    '''
    Basic trainer class
    '''
    def __init__(self, net, optimizer_name, loss_name ) -> None:
        self.net = net
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam
        
        self.ls_fn = Losses(loss_name)()
        self.logger = Logger()

    def train(self, train,
              val,
              epochs,
              batch_size,
              learning_rate):
        '''
        args:
            train: training dataset
            val: validation dataset
        '''
        self.net.train()
        optimizer = self.optimizer(self.net.parameters(), lr=learning_rate)
        train_loader = DataLoader(train, batch_size=batch_size)
        val_loader = DataLoader(val, batch_size=val.__len__())

        for val in val_loader:
            x_val, y_val = val
            y_val, _ = y_val

        for ep in range(epochs):
            running_loss = []
            for x_train, y_train in train_loader:
                y_train, _ = y_train
                optimizer.zero_grad()
                out = self.net(x_train)
                batch_loss = self.ls_fn(y_train, out)
                batch_loss.backward()
                running_loss.append(batch_loss.item())
                optimizer.step()

            val_out = self.net(x_val) 
            val_loss = self.ls_fn(y_val, val_out)
            self.logger.record('epoch', ep+1)
            self.logger.record('train_loss', np.mean(running_loss))
            self.logger.record('val_loss', val_loss.item())
            self.logger.print()
            
    def eval(self, test_set):
        self.net.eval()
        x_test, y_test = test_set
        pred_test = self.net(x_test)
    
        


class MultiStepTrainer(Trainer):
    def __init__(self, net, epochs) -> None:
        super(MultiStepTrainer, self).__init__(net, epochs)
        pass