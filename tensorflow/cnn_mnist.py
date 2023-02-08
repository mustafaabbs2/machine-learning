import tensorflow as tf
from tensorflow.keras import layers

# Load MNIST dataset
# MNIST (Modified National Institute of Standards and Technology) is a large dataset of handwritten digits that is commonly used as a benchmark for image classification tasks in the field of machine learning and deep learning. It consists of 60,000 training images and 10,000 test images, each of which is 28x28 pixels in size and labeled with a single digit between 0 and 9. The MNIST dataset is widely used as a simple and well-understood starting point for developing and testing machine learning algorithms for image classification.


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape data to fit into a ConvNet
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the data
x_train /= 255
x_test /= 255

# Define the model
model = tf.keras.Sequential()

# Add the first convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Add the second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the output from the ConvNet
model.add(layers.Flatten())

# Add the dense layer
model.add(layers.Dense(64, activation='relu'))

# Add the output layer
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
