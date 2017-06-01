# The modules we're going to use
#from __future__ import print_function

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, GaussianNoise, GaussianDropout
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from data_util import BasicHeroData, double_inverse_samples
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
import numpy as np

from model_output import ModelOutput

from keras.regularizers import l2

class DnnModel(ModelOutput):

    def run_model(self, data, targets, batch_size, epochs):

        test_size_1 = 0.25
        test_size_2 = 0.2
        noise = 0.5
        drop_out = 0.5


        # split the data up into multiple sets: training, testing validation
        train_data, data_set_2, train_target, target_set_2 = train_test_split(data, targets, test_size=test_size_1, random_state=42)
        test_data, val_data, test_target, val_target = train_test_split(data_set_2, target_set_2, test_size=test_size_2, random_state=24)

        # Pre-processing
        X_train = train_data.reshape(train_data.shape[0],-1)
        X_test = test_data.reshape(test_data.shape[0],-1)
        y_train = np_utils.to_categorical(train_target, 2)
        y_test = np_utils.to_categorical(test_target, 2)
        val_data = val_data.reshape(val_data.shape[0],-1)
        val_target = np_utils.to_categorical(val_target, 2)

        # create a linear model

        model = Sequential()
        # add a dense layer with 2048 neurons, relu activation
        model.add(Dense(2048, activation='relu', input_dim=train_data.shape[1]))
        # add noise to the dataset
        model.add(GaussianNoise(noise))
        # set the dropout rate to avoid overfitting
        model.add(GaussianDropout(drop_out))
        #add dense layer with 2 neurons and softmax activation to get categories, also l2 norm
        model.add(Dense(2, activation='softmax', W_regularizer=l2(0.01)))
        # define adam optimizer
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # compile mode to use cross entropy
        model.compile(
            optimizer = adam,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # fit the model and use cross validation
        model.fit(X_train, y_train, batch_size = batch_size, epochs=epochs, verbose=2, validation_data=(val_data, val_target) )
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
            "noise": noise,
            "drop_out": drop_out,
            "data_doubled": "no",
            "l2_reg": "yes"
        }

        return metrics, model


if __name__ == '__main__':
    # run model with various thresholds and epoch/batch sizes

    #DnnModel('./Data/hero_data/threshold_001.json', 'dnn', 'dnn', 64, 25)
    #DnnModel('./Data/hero_data/threshold_002.json', 'dnn', 'dnn', 64, 25)
    DnnModel('./Data/hero_data/threshold_003.json', 'dnn', 'dnn', 64, 25)
    #DnnModel('./Data/hero_data/threshold_004.json', 'dnn', 'dnn', 64, 25)
    #DnnModel('./Data/hero_data/threshold_005.json', 'dnn', 'dnn', 64, 25)
    #DnnModel('./Data/hero_data/full_40000_plus_data.json', 'dnn', 'dnn', 64, 10)
    #DnnModel('./Data/hero_data/threshold_001.json', 'dnn', 'dnn', 64, 10)
    #DnnModel('./Data/hero_data/threshold_002.json', 'dnn', 'dnn', 64, 10)
    #DnnModel('./Data/hero_data/threshold_003.json', 'dnn', 'dnn', 64, 10)
    #DnnModel('./Data/hero_data/threshold_004.json', 'dnn', 'dnn', 64, 10)
    #DnnModel('./Data/hero_data/threshold_005.json', 'dnn', 'dnn', 64, 10)
    #DnnModel('./Data/hero_data/full_40000_plus_data.json', 'dnn', 'dnn', 64, 10)
    #DnnModel('./Data/hero_data/threshold_001.json', 'dnn', 'dnn', 32, 20)
    #DnnModel('./Data/hero_data/threshold_002.json', 'dnn', 'dnn', 32, 20)
    #DnnModel('./Data/hero_data/threshold_003.json', 'dnn', 'dnn', 32, 20)
    #DnnModel('./Data/hero_data/threshold_004.json', 'dnn', 'dnn', 32, 20)
    #DnnModel('./Data/hero_data/threshold_005.json', 'dnn', 'dnn', 32, 20)
    #DnnModel('./Data/hero_data/full_40000_plus_data.json', 'dnn', 'dnn', 32, 20)


