# conda activate tf-2

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

#!pip install -q tensorflow==2.0.0-alpha0
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
tf.enable_eager_execution()



df = pd.read_csv(filepath_or_buffer = "/home/micah/Desktop/data/creditcard.csv")

print (df.head() )
print (df.info() )
print (df.isna().sum() )

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Class')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return dataframe.values


#batch_size = 5 # A small batch sized is used for demonstration purposes
#train_ds = df_to_dataset(train, batch_size=batch_size)
#val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
#test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#print (type(val_ds) )
#print (val_ds[:3])



feature_columns = []

features = df.columns.tolist()[:-1]

print (len(features))
# numeric cols
for header in features:
  feature_columns.append(feature_column.numeric_column(header))
#print (feature_columns)

#feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32 
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
'''
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of Amount:', feature_batch['Amount'])
  print('A batch of targets:', label_batch )
'''

features, labels = next(iter(train_ds))


model = tf.keras.Sequential([
  tf.keras.layers.Dense(1, activation=tf.nn.relu, input_shape=(30,)  ),  # input shape required
  tf.keras.layers.Dense(1, activation=tf.nn.relu),
  tf.keras.layers.Dense(1)
])

predictions = model(features)

#print (predictions[:5])
#print (tf.nn.softmax(predictions[:5]) )

#print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
#print("    Labels: {}".format(labels))

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

#l = loss(model, features, labels)
#print("Loss test: {}".format(l))

'''
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

'''
'''
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

'''
