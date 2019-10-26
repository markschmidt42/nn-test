import tensorflow as tf

x = [[2.]]; 

print('tensorflow version', tf.__version__); 
print('hello, {}'.format(tf.matmul(x, x)))