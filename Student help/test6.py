
#import gensim
# Set up & handling the data
'''
import csv
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action="ignore")

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#import umap
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import datasets, metrics

import time

import seaborn as sns
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
'''

import pandas as pd
import numpy as np

import pandas as pd



array = np.array([[1, '', 3], [4, 5, np.nan]])
#array
#array([[ 1., nan,  3.],
#       [ 4.,  5., nan]])
df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', '', 'fly']])

#print (np.all(np.isfinite(df.values)))
print (np.any(np.isnan(df)) )

print (pd.isnull(array) )
print (df.isnull().sum() )
#array([[False,  True, False],
#       [False, False,  True]])


'''
df = pd.DataFrame({"A":[-5, 8, 12, -9, 5, 3],
"B":[-1, -4, 6, 4, 11, 3],
"C":[11, 4, -8, 7, 3, -2]})

print (df )

#df_clipped = df[['A', 'B']].clip(-1, 1)

#print (df_clipped)


df.loc[(df.A > 0),'A']=1
print df
'''
'''
model_pretrained = gensim.models.KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz', binary=True)

word2vec_arr = np.zeros((df.shape[0],300))

print model_pretrained.head()
'''

'''
for i, row in enumerate(df['comment_text']):
try:
word2vec_arr[i,:] = np.mean([model_pretrained[lemma] for lemma in row], axis=0)
except KeyError:
word2vec_arr[i,:] = np.full((1,300), np.nan)
continue

word2vec_arr = pd.DataFrame(word2vec_arr)
df = pd.concat([df, word2vec_arr], axis=0)

'''
