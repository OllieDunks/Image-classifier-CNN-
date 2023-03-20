# Importing libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tabulate import tabulate
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from sklearn import metrics
from tensorflow.python.ops.numpy_ops import np_config

# Downloading the MNIST dataset into it's training and testing variables.
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Validation set created from 0 to 9999 from the training data.
Validation_X = train_X[0:10000]
Validation_y = train_y[0:10000]

# Setting training data to start from 10,000 instead of 0.
train_X = train_X[10000:60000]
train_y = train_y[10000:60000]


# Changing the values of the image from 0 to 255 to 0 to 1.
train_X: float = train_X / 255
Validation_X: float = Validation_X / 255
test_X: float = test_X / 255

# Reshaping the images so they can pass through the CNN.
train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
Validation_X = Validation_X.reshape((Validation_X.shape[0], 28, 28, 1))
test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))


# Creating a table to see a visual of how many images in each variable.
def plotting_a_table_of_number_of_images_per_section():
    table_data = [['Train X', train_X.shape], ['Train y', train_y.shape],
                  ['Validation X', Validation_X.shape], ['Validation y', Validation_y.shape],
                  ['Test X', test_X.shape], ['Test y', test_y.shape]]
    column_names = ['Data Type', 'Quantity and Image Size']
    print(tabulate(table_data, column_names, tablefmt='fancy_grid', showindex='always'))


# Plotting the second image in the training set after flipping it
# to see if it came out the right way.
def flipping_the_image_when_plotting():
    plt.imshow(train_X[n])
    plt.show()
    image1 = np.flipud(train_X[n])
    plt.imshow(image1, origin='lower', cmap=plt.get_cmap('Greys'))
    plt.show()
    print('The number shown is ' + str(train_y[n]))


# function outlining the general shape of the images link [1].
def printing_dataset_layout():
    print('train_X: ' + str(train_X.shape))
    print('train_y: ' + str(train_y.shape))
    print('Validation_X: ' + str(Validation_X.shape))
    print('Validation_y: ' + str(Validation_y.shape))
    print('test_X:  ' + str(test_X.shape))
    print('test_y:  ' + str(test_y.shape))


# function that prints the images of the first 9 images link [1].
def printing_numbers():
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(train_X[i], cmap=plt.get_cmap('Greys'))
        plt.show()


def create_model(my_learning_rate):
    '''Creating the Neural Network link[4]'''

    model = tf.keras.models.Sequential()

    # The neural network which consists of 3 convolution layers,
    # 2 max pooling layers and 1 dropout layer.

    # Current model with
    # model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(124, (3, 3), activation='relu'))
    # model.add(Dropout(rate=0.2))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))

    # Optimising the model
    model.add(Conv2D(25, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(12, (3, 3), activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(25, (3, 3), activation='relu'))

    # Used to flatten the data from 2D to 1D
    # Shape was from the model summary.
    model.add(Flatten(input_shape=(3, 3, 25)))

    # Define the output layer. There are 10 units due to
    # the 10 possible outcomes 0 to 9 inclusive.
    model.add(Dense(units=10, activation='softmax'))
    model.summary()

    # Formatting the layers into a model so that they can be executed.
    # optimizer.Adam is the gradient decent method
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    return model


def train_model(model, train_X, train_y, epochs,
                batch_size=None, Validation_data=(Validation_X, Validation_y)):
    # See link 3 for link.
    history = model.fit(train_X, train_y, batch_size=batch_size,
                        epochs=epochs, validation_data=(Validation_X, Validation_y))

    # To track the progression of training, gather a snapshot
    # of the model's metrics at each epoch.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


# The following variables are the parameters for the model.
learning_rate = 0.003  # good learning rate
epochs = 50  # should be 50
batch_size = 1000
Validation = (Validation_X, Validation_y)

# Establish the model's topography.
my_model = create_model(learning_rate)

# Train the model on the training set.
epochs, hist = train_model(my_model, train_X, train_y,
                           epochs, batch_size, Validation)


#pred = hist.predict(Validation)
pred = my_model.predict(test_X)
# See link [8] for the line below to decode the prediction
pred_class = tf.argmax(pred, axis=-1)  # either tf.math.argmax() or tf.argmax will work
print(str(pred_class))

confusion_matrix = metrics.confusion_matrix(test_y, pred_class)
display_labels = [0,1,2,3,4,5,6,7,8,9]
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
cm_display.plot()
plt.title('Confusion matrix for the CNN')
plt.show()


# Plot a graph of the training and validation accuracy.

plt.subplot(1, 2, 1)
plt.plot(hist['accuracy'], label='training accuracy')
plt.plot(hist['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.95, 1])
plt.legend(loc='lower right')
plt.title('Training accuracy compared to validation accuracy')
#plt.show()

# Plotting a graph of the training and validation loss
plt.subplot(1, 2, 2)
plt.plot(hist['loss'], label='training loss')
plt.plot(hist['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 0.2])
plt.legend(loc='upper right')
plt.title('Training loss compared to validation loss')
#plt.show()

# Evaluate against the Testing set.
#print("\n Evaluate the new model against the test set:")
#my_model.evaluate(x=test_X, y=test_y, batch_size=batch_size)


