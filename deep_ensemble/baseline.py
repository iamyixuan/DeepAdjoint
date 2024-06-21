import numpy as np
import pickle
import matplotlib.pyplot as plt


with open('/home/iamyixuan/work/ImPACTs/HPO/log/log_bsize_128_ep_100.pkl', 'rb') as f:
    log = pickle.load(f)

print(log.keys())

print((log['ValLoss'][:20]))