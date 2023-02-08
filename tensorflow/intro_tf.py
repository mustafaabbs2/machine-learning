import tensorflow as tf

# Check if GPU is available
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

# # Place the computations on GPU if available
with tf.device('GPU:0' if tf.config.experimental.list_physical_devices('GPU') else 'CPU:0'):
  # Define two matrices
  a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

  # Perform matrix multiplication
  c = tf.matmul(a, b)

#   1 2
#   3 4

#   x

#   5 6
#   7 8

#   =
#  19 22
#  43 50


print(c)
