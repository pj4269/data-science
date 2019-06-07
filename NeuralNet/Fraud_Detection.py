import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#from tensorflow.keras.utils import np_utils
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


df = pd.read_csv(filepath_or_buffer = "/home/micah/Desktop/data/creditcard.csv")

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print (df.head() )
print (df.info() )
print (df.isna().sum() )

print (df.head() )
print (df.info() )
print (df.isna().sum() )

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')



# approach 1: Keras
dataset = df.values
X = dataset[:,1:30].astype(float)
Y = dataset[:,:-1]


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(1, input_dim=29, activation='relu'))
	model.add(Dense(30, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=2, batch_size=5, verbose=0)
kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, Y, cv=kfold)
print("Keras baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# approach 2: Tensorflow

model = tf.keras.Sequential([

    tf.keras.layers.Dense(1,input_dim=29, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax) # predicting 2 numbers
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit( train.iloc[:,1:30].values, train.iloc[:,-1:].values, epochs=1)

val_loss, val_acc = model.evaluate(val.iloc[:,1:30].values, val.iloc[:,-1:].values)

print('Validation accuracy:', val_acc)


predictions = model.predict(test.iloc[:,1:30].values)
idx = predictions.argmax(axis=1)

test = test.iloc[:,-1:].values.flatten()

print (len(test) )
print (len(predictions) )
acc = sklearn.metrics.accuracy_score(list(test), list(idx))
print (acc)

