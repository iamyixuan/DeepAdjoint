import numpy as np
import pickle
import os
from datetime import datetime


class Logger:
    def __init__(self) -> None:
        self.logger = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        } 
    def record(self, key, value):
        self.logger[key].append(value)
    
    def print(self):
        print('Epoch: %d , Training loss: %.4f, Validation loss: %.4f' % (self.logger['epoch'][-1], self.logger['train_loss'][-1], self.logger['val_loss'][-1]))

    def finish(self):
        time = datetime.now().strftime('%Y-%M-%D-%H')
        if not os.path.exists('./logs/'):
            os.makedirs('./logs/')
        with open('./logs/' + time + '_logs.pkl') as f:
            pickle.dump(self.logger, f)
        print('Saving the training logs...')
