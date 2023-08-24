import numpy as np
import pickle
import os
from datetime import datetime


class Logger:
    def __init__(self, path) -> None:
        self.logger = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        } 
        self.path = path

        try:
            if not os.path.exists(os.path.join(self.path, 'logs/')):
                os.makedirs(os.path.join(self.path, 'logs/'))
        except:
            pass

    def record(self, key, value):
        if key not in self.logger.keys():
            self.logger[key] = []
        self.logger[key].append(value)
    
    def print(self):
        print('Epoch: %d , Training loss: %.4f, Validation loss: %.4f' % (self.logger['epoch'][-1], self.logger['train_loss'][-1], self.logger['val_loss'][-1]))

    def save(self):
        with open(self.path + '/logs/logs.pkl', 'wb') as f:
            pickle.dump(self.logger, f)
        print('Saving the training logs...')
