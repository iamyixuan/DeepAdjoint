import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hpo import getBestConfig

df = pd.read_csv('./results_20epochFixed.csv')

bestConfig = getBestConfig(df)
print(bestConfig)
