from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os.path
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, accuracy_score
import utils.pymark as pymark

# https://colab.research.google.com/drive/1md0xrsTWjrYH3R8l5z96FpWiy8IuWZWT#scrollTo=uFt2-9epOpN4
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

DATA_TYPE = 'complex'
data_type = sys.argv[1] if len(sys.argv) == 2 else DATA_TYPE

TRAIN_TEST_DATA_CSV = f'data/{data_type}_train_test.csv'
PREDICT_DATA_CSV    = f'data/{data_type}_predict.csv'

EPOCHS = 5000
VALIDATION_SPLIT_PERCENT = 0.2

if not os.path.exists(TRAIN_TEST_DATA_CSV):
  print(f'ERROR: File does not exist: {TRAIN_TEST_DATA_CSV}')
  print(f'Please run the following to genrate data:\n\tpython generate-data.py --type {data_type}')
  sys.exit()


train_dataset, train_labels, test_dataset, test_labels, output_column_name = pymark.get_data(TRAIN_TEST_DATA_CSV, normalize=True)

print(train_dataset.tail())
input_size = len(train_dataset.keys())

# LR Model
model = LinearRegression()
model.fit(train_dataset, train_labels)

def test_model(model, label, X, y, output_column_name):
  score = model.score(X, y)

  count = y.shape[0]

  y_predictions = model.predict(X).flatten()
  pymark.compare_actual_and_predicted(y, y_predictions)

  label = f'{label} ({count} records)'

  print(f'{label} Score: {score:5.2f} {output_column_name}')

  pymark.plot_actual_and_predicted(label, y, y_predictions, output_column_name)

#end def ------------------------------------------------------------------------------------------


# # let's test the model with our test data (from the training set)
test_model(model, 'test_dataset', test_dataset, test_labels, output_column_name)


# let's try it on some brand new data it has never seen
predict_dataset, predict_labels, output_column_name = pymark.get_data(PREDICT_DATA_CSV, normalize=True, split_percent=0)

test_model(model, 'predict_dataset', predict_dataset, predict_labels, output_column_name)