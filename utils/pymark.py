import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
###################################################################################################

# With my standard, I prefix all columns that are garbage with the word "Ignore"
# I also prefix output columns with the word "Output"
def get_x_and_y(df, y_column_name):
  df_y = df[y_column_name]
  
  ignore_cols = [col for col in df if col.startswith('Ignore') or col.startswith('Output')]

  print('Dropping these columns:', ignore_cols)
  
  df_x = df[df.columns.drop(ignore_cols)]

  return df_x, df_y
#end def ------------------------------------------------------------------------------------------

def get_data(csv_filepath, split_percent=0.8, output_column_name=None, normalize=True):
  raw_dataset = pd.read_csv(csv_filepath)

  if output_column_name == None:
    # figure it out
    output_column_name = 'Output0'

  dataset = raw_dataset.copy()
  # print(dataset.tail())

  # don't split if it is zero or 100%
  split_data = (split_percent > 0.0 and split_percent < 1.0)

  if split_data:
    train_dataset = dataset.sample(frac=split_percent, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
  else:
    train_dataset = dataset
        
  train_stats = train_dataset.describe()
  train_stats.pop(output_column_name)
  train_stats = train_stats.transpose()
  train_stats

  train_labels = train_dataset.pop(output_column_name)
  
  if split_data:
    test_labels = test_dataset.pop(output_column_name)

  def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

  if normalize:
    train_dataset = norm(train_dataset)
    if split_data:
      test_dataset = norm(test_dataset)

  if split_data:
    return train_dataset, train_labels, test_dataset, test_labels, output_column_name
  else:
    return train_dataset, train_labels, output_column_name
#end def ------------------------------------------------------------------------------------------

def plot_history(history, output_column_name):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel(f'Mean Abs Error [{output_column_name}]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel(f'Mean Square Error [${output_column_name}^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()
#end def ------------------------------------------------------------------------------------------

def test_model(model, label, testing_dataset, testing_labels, output_column_name):
  loss, mae, mse = model.evaluate(testing_dataset, testing_labels, verbose=2)
  
  # LIMIT_RECORDS = 5
  LIMIT_RECORDS = None

  if LIMIT_RECORDS:
    testing_dataset = testing_dataset[:LIMIT_RECORDS]
    testing_labels  = testing_labels[:LIMIT_RECORDS]

  count = testing_labels.shape[0]

  label = f'{label} ({count} records)'

  print(f'{label} Mean Abs Error: {mae:5.2f} {output_column_name}')

  test_predictions = model.predict(testing_dataset).flatten()

  print('actual', 'pred')
  for (actual, pred) in zip(testing_labels, test_predictions):
    print(actual, pred)

  plt.scatter(testing_labels, test_predictions, edgecolors='g')
  plt.legend([ f'Predicted Y {output_column_name}'])
  plt.title(label)
  plt.xlabel(f'Actual {output_column_name}')
  plt.ylabel(f'Predictions {output_column_name}')
  plt.axis('equal')
  plt.axis('square')
  _ = plt.plot([-100, 100], [-100, 100])
  plt.show()

  error = test_predictions - testing_labels
  plt.title(label)
  plt.hist(error, bins = 25)
  plt.xlabel(f'Prediction Error {output_column_name}')
  _ = plt.ylabel("Count")
  plt.show()
#end def ------------------------------------------------------------------------------------------




# With my standard, I prefix all categories with the word "Category"
def encode_category_features(df):
    features = [col for col in df if col.startswith('Category')]
    #features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df
#end def ------------------------------------------------------------------------------------------

# Build the model def for ludwig based on column names
def ludwig_build_model_definition(df, output_col=None, output_type=None):
  # returns "{input_features: [{name: text, type: text, encoder: parallel_cnn, level: word}], output_features: [{name: class, type: category}]}"
  
  if (output_col == None):
    output_col = [col for col in df if col.startswith('Output')][0] # get the FIRST "Output" column 
  
  # todo: handle category
  if (output_type == None):
    if (min(df[output_col]) == 0 and max(df[output_col]) == 1):
      output_type = 'binary'
    else:
      output_type = 'numerical'

  # print(min(df[output_col]), max(df[output_col]), output_col, output_type)
  
  inputs = []
  
  for col in df:
    if not col.startswith('Output'): # skip if it does
      if col.startswith('Category'):
        input_type = 'category'
      else:
        input_type = 'numerical'
      
      inputs.append({ 'name': col, 'type': input_type })
  
  return {
    "input_features": inputs,
    "output_features": [{
      "name": output_col,
      "type": output_type,
    }]
  }
#end def ------------------------------------------------------------------------------------------