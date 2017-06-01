# The modules we're going to use
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from data_util import BasicHeroData

import numpy as np

#set numpy seed
np.random.seed(1337)

def cnn(data, targets, modelfile=None):
    # split the data up into multiple sets: training, testing validation
    train_data, data_set_2, train_target, target_set_2 = train_test_split(data, targets, test_size=0.25,random_state=42)
    test_data, val_data, test_target, val_target = train_test_split(data_set_2, target_set_2, test_size=0.15,random_state=24)
    # display dataset sizes
    print('Normal, X_train size: ', train_data.shape)
    print('Normal, y_train size: ', train_target.shape)
    print('Normal, X_test size: ', test_data.shape)
    print('Normal, y_test size: ', test_target.shape)
    print()
    # pre-processing
    X_train = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    X_test = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
    y_train = np_utils.to_categorical(train_target, 2)
    y_test = np_utils.to_categorical(test_target, 2)
    val_data = val_data.reshape(val_data.shape[0], -1, 1)
    val_target = np_utils.to_categorical(val_target, 2)
    # display dataset sizes after processing
    print('After pre-processing, X_train size: ', X_train.shape)
    print('After pre-processing, y_train size: ', y_train.shape)
    print('After pre-processing, X_test size: ', X_test.shape)
    print('After pre-processing, y_test size: ', y_test.shape)
    # create a linear model
    model = Sequential()
    # add a convolutional layer
    model.add(convolutional.Conv1D(
        filters=16,
        kernel_size=1,
        padding='same',
        strides=1,
        activation='relu',
        input_shape=X_train.shape[1:]
    ))
    # add max pooling layer
    model.add(pooling.MaxPooling1D(
        pool_size=1,
        padding='same',
    ))
    # add a convolutional layer
    model.add(convolutional.Conv1D(
        filters=32,
        kernel_size=2,
        padding='same',
        strides=1,
        activation='relu',
    ))
    # add max pooling layer
    model.add(pooling.MaxPooling1D(
        pool_size=1,
        padding='same',
    ))
    # flatten the activation maps into a 1d vector
    model.add(Flatten())
    # add a dense layer with 128 neurons
    model.add(Dense(128))
    # set activation layer
    model.add(Activation('relu'))
    # set drop out rate
    model.add(Dropout(.5))
    # add a dense layer with 2 neurons
    model.add(Dense(2))
    # set softmax function to make the categories
    model.add(Activation('softmax'))
    # define adam optimizer
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # compile mode to use cross entropy
    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # fit the model and use cross validation
    model.fit(X_train, y_train, batch_size=64, epochs=25, verbose=2, validation_data=(val_data, val_target))
    # get the loss and accuracy of our model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    # display results on testing data
    print('The loss on testing data', loss)
    print('The accuracy on testing data', accuracy)
    # get the loss and accuracy of our model
    loss, accuracy = model.evaluate(val_data, val_target, verbose=2)
    # display results on validation data
    print('The loss on validation data', loss)
    print('The accuracy on validaiton data', accuracy)
    # save model if flag is set
    if modelfile:
        if modelfile.endswith('.h5'):
            model.save('./models/{}'.format(modelfile))
        else:
            print ("Can't save your model; bad extension")

if __name__ == '__main__':
    # create hero data object
    h = BasicHeroData()
    # load in dataset
    d = h.load_saved_hero_data('./Data/hero_data/full_40000_plus_data.json')
    # extract dataset info
    data, targets, features, target_labels = d
    # call machine learning function
    cnn(data, targets)
