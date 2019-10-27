import sys
import csv
import random
import argparse

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
    output = (inputs[0] + inputs[1]) * (inputs[2] + inputs[3])
  elif output_function_type == 'complex2':
    output = ((inputs[0] + inputs[1]) * (inputs[2] + inputs[3])) / (inputs[4] * inputs[8])
    # print(f'{output} = (({inputs[0]} + {inputs[1]}) * ({inputs[2]} + {inputs[3]})) / ({inputs[4]} * {inputs[8]})')
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

def main():
  create_file('train_test.csv', records=RECORDS, min_value=MIN_VALUE, max_value=MAX_VALUE, output_function_type=OUTPUT_TYPE)
  create_file('predict.csv', records=int(RECORDS/10), min_value=MIN_VALUE, max_value=MAX_VALUE, output_function_type=OUTPUT_TYPE)
# end function --------------------------------------------------------------

def parsargs():
  global RECORDS, OUTPUT_TYPE
  parser = argparse.ArgumentParser(description='Generate some fake data for Neural Networks.')

  parser.add_argument('--records', dest='records', default=RECORDS, type=int,
                      help=f'Number of records to generate (default: {RECORDS}).')

  parser.add_argument('--type', dest='type', default=OUTPUT_TYPE,
                      help=f'Which output calculation should we use? simple, simple2, simple3, complex (default: {OUTPUT_TYPE})')


  args = parser.parse_args()
  RECORDS = args.records
  OUTPUT_TYPE = args.type
# end function --------------------------------------------------------------
      

if __name__ == '__main__':
  parsargs()
  main()