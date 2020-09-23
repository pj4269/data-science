import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


l1 = ['how', 'to', 'do', 'it', 'in', 'java']

l1 = ['a', 'b', 'c', 'd', 'e', 'f']
print max(l1), min(l1)
'''
data_path = ("https://tf-assets-prod.s3.amazonaws.com/tf-curric/data-science/cleveland-data.csv")
heartdisease_df = pd.read_csv(data_path, header=None)

print heartdisease_df.head()

# Define the features and the outcome
X = heartdisease_df.iloc[:, :13]
y = heartdisease_df.iloc[:, 13]

# Replace missing values (marked by ?) with a 0
X = X.replace(to_replace='?', value=0)

# Binarize y so that 1 means heart disease diagnosis and 0 means no diagnosis
y = np.where(y > 0, 0, 1)

# Normalize
X_std = StandardScaler().fit_transform(X)

# Data frame to store features and predicted cluster memberships.
ypred = pd.DataFrame()

# Create the two-feature PCA for graphing purposes.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Split the data into four equally-sized samples. First we break it in half:
X_half1, X_half2, X_pcahalf1, X_pcahalf2 = train_test_split( X_std, X_pca, test_size=0.5, random_state=123)

# Pass a list of tuples and a counter that increments each time we go
# through the loop. The tuples are the data to be used by k-means,
# and the PCA-derived features for graphing. We use k-means to fit a
# model to the data, then store the predicted values and the two-feature
# PCA solution in the data frame.
data2 = []
columns = []
cols = []

for counter, data in enumerate([(X_half1, X_pcahalf1), (X_half2, X_pcahalf2)]):
  x_counter = pd.Series(list(data[1][:, counter])) 
  cols.append(x_counter)
  num = counter+1
  columns.append( 'pca_f'+ str(num) + '_sample' + str(counter) )
new_df =   pd.concat(cols, axis=1)
new_df.columns = columns


print new_df.head()
'''

# Generate cluster predictions and store them for clusters 2 to 4.
#for nclust in range(2, 5):
#  pred = KMeans(n_clusters=nclust, random_state=123).fit_predict(data[0])
#  ypred['clust' + str(nclust) + '_sample' + str(counter)] = pred



