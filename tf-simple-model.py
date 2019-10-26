from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os.path
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import utils.pymark as pymark

# https://www.tensorflow.org/tutorials/keras/regression
print(tf.__version__)

DATA_TYPE = 'simple'
data_type = sys.argv[1] if len(sys.argv) == 2 else DATA_TYPE

TRAIN_TEST_DATA_CSV = f'data/{data_type}_train_test.csv'
PREDICT_DATA_CSV    = f'data/{data_type}_predict.csv'

EPOCHS = 500
VALIDATION_SPLIT_PERCENT = 0.2

if not os.path.exists(TRAIN_TEST_DATA_CSV):
  print(f'ERROR: File does not exist: {TRAIN_TEST_DATA_CSV}')
  print(f'Please run the following to genrate data:\n\tpython generate-data.py {data_type}')
  sys.exit()


train_dataset, train_labels, test_dataset, test_labels, output_column_name = pymark.get_data(TRAIN_TEST_DATA_CSV)
input_size = len(train_dataset.keys())

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[input_size]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
    optimizer=optimizer,
    metrics=['mae', 'mse'])

  return model
#end def ------------------------------------------------------------------------------------------

model = build_model()

model.summary()

# Now try out the model. Take a batch of 10 examples from the training data and call model.predict on it.
# example_batch = train_dataset[:10]
# example_result = model.predict(example_batch)
# print(example_result)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print(f'{epoch},', end='')

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# let's test the model BEFORE we train
#pymark.test_model(model, 'test_dataset', test_dataset, test_labels, output_column_name)

history = model.fit(
  train_dataset, 
  train_labels, 
  epochs=EPOCHS, 
  validation_split=VALIDATION_SPLIT_PERCENT, 
  verbose=0, 
  callbacks=[early_stop, PrintDot()])

# pymark.plot_history(history, output_column_name)

# let's test the model with our test data (from the training set)
pymark.test_model(model, 'test_dataset', test_dataset, test_labels, output_column_name)

# let's try it on some brand new data it has never seen
predict_dataset, predict_labels, output_column_name = pymark.get_data(TRAIN_TEST_DATA_CSV, split_percent=0)

pymark.test_model(model, 'predict_dataset', predict_dataset, predict_labels, output_column_name)