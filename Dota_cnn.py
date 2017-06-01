# The modules we're going to use
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from data_util import BasicHeroData, double_inverse_samples

import numpy as np

np.random.seed(1337)
import matplotlib.pyplot as plt
import matplotlib

from model_output import ModelOutput


class CnnModel(ModelOutput):

    def run_model(self, data, targets, batch_size, epochs):
        # double the sample
        data = double_inverse_samples(data)
        targets = double_inverse_samples(targets)

        test_size_1 = 0.25
        test_size_2 = 0.15
        drop_out = 0.5

        # split the data up into multiple sets: training, testing validation
        train_data, data_set_2, train_target, target_set_2 = train_test_split(data, targets, test_size=test_size_1,
                                                                              random_state=42)
        test_data, val_data, test_target, val_target = train_test_split(data_set_2, target_set_2, test_size=test_size_2,
                                                                        random_state=24)
        # pre-processing
        X_train = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
        X_test = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
        y_train = np_utils.to_categorical(train_target, 2)
        y_test = np_utils.to_categorical(test_target, 2)
        val_data = val_data.reshape(val_data.shape[0], -1, 1)
        val_target = np_utils.to_categorical(val_target, 2)

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

        # add a max pooling layer
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
        # add a max pooling layer
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
        model.add(Dropout(drop_out))
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
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(val_data, val_target))
        # get the test loss and accuracy of our model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
        # get the validation loss and accuracy of our model
        val_loss, val_accuracy = model.evaluate(val_data, val_target, verbose=2)
        # collect metrics for output
        metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "batch_size": batch_size,
            "epochs": epochs,
            "test_size_1": test_size_1,
            "test_size_2": test_size_2,
            "drop_out": drop_out
        }

        return metrics, model


if __name__ == '__main__':
    # run model with various thresholds and epoch/batch sizes

    #CnnModel('./Data/hero_data/threshold_001.json', 'cnn', 'cnn', 64, 25)
    #CnnModel('./Data/hero_data/threshold_002.json', 'cnn', 'cnn', 64, 25)
    CnnModel('./Data/hero_data/threshold_003.json', 'cnn', 'cnn', 64, 25)
    #CnnModel('./Data/hero_data/threshold_004.json', 'cnn', 'cnn', 64, 25)
    #CnnModel('./Data/hero_data/threshold_005.json', 'cnn', 'cnn', 64, 25)
    #CnnModel('./Data/hero_data/full_40000_plus_data.json', 'cnn', 'cnn', 32, 20)
    #CnnModel('./Data/hero_data/threshold_001.json', 'cnn', 'cnn', 32, 20)
    #CnnModel('./Data/hero_data/threshold_002.json', 'cnn', 'cnn', 32, 20)
    #CnnModel('./Data/hero_data/threshold_003.json', 'cnn', 'cnn', 32, 20)
    #CnnModel('./Data/hero_data/threshold_004.json', 'cnn', 'cnn', 32, 20)
    #CnnModel('./Data/hero_data/threshold_005.json', 'cnn', 'cnn', 32, 20)
    #CnnModel('./Data/hero_data/full_40000_plus_data.json', 'cnn', 'cnn', 32, 20)

