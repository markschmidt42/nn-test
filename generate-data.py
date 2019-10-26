import sys
import csv
import random

RECORDS = 1000
MIN_VALUE = -10
MAX_VALUE = 10
OUTPUT_TYPE = 'simple'

def calculate_output(inputs, output_function_type = 'simple'):
  output = 0
  if output_function_type == 'simple':
    output = inputs[0] + inputs[1] 
  elif output_function_type == 'simple2':
    output = inputs[0] + inputs[1] + inputs[2] - inputs[8] + inputs[6]
  elif output_function_type == 'simple3':
    output = inputs[0] + inputs[1] + inputs[2] - inputs[8] + inputs[6] - inputs[4] + inputs[3] - inputs[5] + inputs[9]
  elif output_function_type == 'complex':
    output = ((inputs[0] + inputs[1]) * inputs[2] + inputs[3]) / (inputs[4] * inputs[8])
  else:
    raise Exception(f'invalid output_type: {output_function_type}')

  return output
# end function --------------------------------------------------------------

def create_file(filename, records = 1000, columns = 10, min_value=-1, max_value=1, output_function_type = 'simple'):
  # add headers
  header = [f'Input{i}' for i in range(columns)]
  header.append('Output0')

  data = [header]
  for r in range(0, records):
    # generate random inputs
    lst = [random.uniform(min_value, max_value) for i in range(columns)]
    
    # add the output
    lst.append(calculate_output(lst, output_function_type))

    data.append(lst)

  # write the file  
  with open(f'data/{output_function_type}_{filename}', 'w') as f:
    w = csv.writer(f)
    w.writerows(data)

# end function --------------------------------------------------------------

function_type = sys.argv[1] if len(sys.argv) == 2 else OUTPUT_TYPE
print(function_type)
create_file('train_test.csv', records=RECORDS, min_value=MIN_VALUE, max_value=MAX_VALUE, output_function_type=function_type)
create_file('predict.csv', records=int(RECORDS/10), min_value=MIN_VALUE, max_value=MAX_VALUE, output_function_type=function_type)
