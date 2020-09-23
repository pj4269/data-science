import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/micah/Desktop/data/housing price/train.csv', sep=",", header=None)
#df.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'target']


pd.set_option('display.max_columns', 999)

print df.head(n=3)
