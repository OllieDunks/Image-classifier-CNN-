# Importing libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tabulate import tabulate
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from tensorflow.keras.losses import mse
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.utils import shuffle
from sklearn import metrics
from tensorflow.python.ops.numpy_ops import np_config

# Downloading the MNIST dataset into it's training and testing variables.
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X,train_y = shuffle(train_X, train_y, random_state=0)
test_X,test_y = shuffle(test_X, test_y, random_state=0)

################################ Preprocessing the data ################################################################

# Validation set created from 0 to 9999 from the training data.
Validation_X = train_X[0:10000]
Validation_y = train_y[0:10000]

# Setting training data to start from 10,000 instead of 0.
train_X = train_X[20000:60000]
train_y = train_y[20000:60000]

# Attack data.
Attack_X = train_X[10000:15000]
Attack_y = train_y[10000:15000]

# Defence training
Defence_X = train_X[15000:20000]
Defence_y = train_y[15000:20000]

# Changing the values of the image from 0 to 255 to 0 to 1.
train_X: float = train_X / 255
Validation_X: float = Validation_X / 255
test_X: float = test_X / 255
Attack_X: float = Attack_X / 255
Defence_X: float = Defence_X / 255

# Reshaping the images so they can pass through the CNN.
train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
Validation_X = Validation_X.reshape((Validation_X.shape[0], 28, 28, 1))
test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))
Attack_X = Attack_X.reshape((Attack_X.shape[0], 28, 28, 1))
Defence_X = Defence_X.reshape((Defence_X.shape[0], 28, 28, 1))


# Creating a table to see a visual of how many images in each variable.
def plotting_a_table_of_number_of_images_per_section():
    table_data = [['Train X', train_X.shape], ['Train y', train_y.shape],
                  ['Validation X', Validation_X.shape], ['Validation y', Validation_y.shape],
                  ['Attack X', Attack_X.shape], ['Attack y', Attack_y.shape], ['Defence X', Defence_y.shape],
                  ['Defence y', Defence_y.shape], ['Test X', test_X.shape],
                  ['Test y', test_y.shape]]
    column_names = ['Data Type', 'Quantity and Image Size']
    print(tabulate(table_data, column_names, tablefmt='fancy_grid', showindex='always'))


plotting_a_table_of_number_of_images_per_section()


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


##################################### Creating the CNN model ###########################################################

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
    model.add(Dense(units=11, activation='softmax'))
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
epochs = 40
batch_size = 1000
Validation = (Validation_X, Validation_y)

# Establish the model's topography.
my_model = create_model(learning_rate)

################################ Defence Technique #####################################################################


Defence_image = Defence_X
Defence_label = Defence_y


# Onenote Reference [7]
def generate_adversary(my_model, Defence_image, Defence_label, eps=2 / 255):
    Defence_image = tf.cast(Defence_image, tf.float64)
    np_config.enable_numpy_behavior()
    with tf.GradientTape() as tape:
        tape.watch(Defence_image)
        prediction = my_model(Defence_image)
        prediction = np.max(prediction, axis=1)
        loss = mse(Defence_label, prediction)
    gradient = tape.gradient(loss, Defence_image)

    # print(str(gradient))
    sign_grad = tf.sign(gradient)
    adversary = (Defence_image + (sign_grad * eps)).numpy()
    return adversary


adversary_defence = generate_adversary(my_model, Defence_image, Defence_label, eps=0.15)
#print(str(adversary.shape))
adversary_label_defence = [None] * 5000
for i in range(5000):
    adversary_label_defence[i] = 10

train_X = np.concatenate((train_X, adversary_defence))
train_y = np.concatenate((train_y, adversary_label_defence))

train_X,train_y = shuffle(train_X, train_y, random_state=0)


print(str(train_X.shape))
print(str(train_y.shape))

################################ Training the model ####################################################################

# Train the model on the training set.
epochs, hist = train_model(my_model, train_X, train_y,
                           epochs, batch_size, Validation)


plt.plot(hist['accuracy'], label='training accuracy')
plt.plot(hist['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.95, 1])
plt.legend(loc='lower right')
plt.title('Training accuracy compared to validation accuracy')
# plt.show()

# Plotting a graph of the training and validation loss
plt.plot(hist['loss'], label='training loss')
plt.plot(hist['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.ylim([0, 0.2])
plt.legend(loc='upper right')
plt.title('Training loss compared to validation loss')
# plt.show()



################################# Generating Attack on the CNN model ###################################################

input_image = Attack_X
input_label = Attack_y

# Onenote Reference [7]
def generate_adversary(my_model, input_image, input_label, eps=2 / 255):
    input_image = tf.cast(input_image, tf.float64)
    np_config.enable_numpy_behavior()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = my_model(input_image)
        prediction = np.max(prediction, axis=1)
        loss = mse(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    sign_grad = tf.sign(gradient)
    adversary = (input_image + (sign_grad * eps)).numpy()
    return adversary


adversary_attack = generate_adversary(my_model, input_image, input_label, eps=0.2)

adversary_label_attack = [None] * 5000
for i in range(5000):
    adversary_label_attack[i] = 10


################################# Evaluating the model with attacked images and clean ones #############################


test_X = np.concatenate((test_X, adversary_attack))
test_y = np.concatenate((test_y, adversary_label_attack))

test_X,test_y = shuffle(test_X, test_y, random_state=0)

pred = my_model.predict(test_X)
# See link [8] for the line below to decode the prediction
pred_class = tf.argmax(pred, axis=-1)  # either tf.math.argmax() or tf.argmax will work
print(str(pred_class))

confusion_matrix = metrics.confusion_matrix(test_y, pred_class)
display_labels = [0,1,2,3,4,5,6,7,8,9,10]
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
cm_display.plot()
plt.title('Confusion matrix of the model with the defence')
plt.show()

# Evaluate against the Testing set.
#print("\n Evaluate the new model against the test set:")
#my_model.evaluate(x=test_X, y=test_y, batch_size=batch_size)