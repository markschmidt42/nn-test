import sys
import csv
import random

RECORDS = 1000
MIN_VALUE = -10
MAX_VALUE = 10
OUTPUT_TYPE = 'simple'

def create_file(filename, records = 1000, columns = 10, min_value=-1, max_value=1, output_function_type = 'simple'):

  # add headers
  header = [f'Input{i}' for i in range(columns)]
  header.append('Output0')

  data = [header]
  for r in range(0, records):
    # generate random inputs
    lst = [random.uniform(min_value, max_value) for i in range(columns)]
    
    # add the output
    output = 0
    if output_function_type == 'simple':
      output = lst[0] + lst[1] 
    elif output_function_type == 'simple2':
      output = lst[0] + lst[1] + lst[2] - lst[8] + lst[6]
    elif output_function_type == 'simple3':
      output = lst[0] + lst[1] + lst[2] - lst[8] + lst[6] - lst[4] + lst[3] - lst[5] + lst[9]
    elif output_function_type == 'complex':
      output = ((lst[0] + lst[1]) * lst[2] + lst[3]) / (lst[4] * lst[8])
    
    lst.append(output)
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
