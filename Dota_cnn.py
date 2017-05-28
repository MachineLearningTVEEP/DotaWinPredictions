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
        print data.shape
        data = double_inverse_samples(data)
        targets = double_inverse_samples(targets)
        print data.shape

        test_size_1 = 0.25
        test_size_2 = 0.15
        drop_out = 0.5

        # train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=42)
        train_data, data_set_2, train_target, target_set_2 = train_test_split(data, targets, test_size=test_size_1,
                                                                              random_state=42)
        test_data, val_data, test_target, val_target = train_test_split(data_set_2, target_set_2, test_size=test_size_2,
                                                                        random_state=24)

        print('Normal, X_train size: ', train_data.shape)
        print('Normal, y_train size: ', train_target.shape)
        print('Normal, X_test size: ', test_data.shape)
        print('Normal, y_test size: ', test_target.shape)
        print()
        # train_data = np.concatenate((train_data, data_double), 0)
        # train_target = np.concatenate((train_target, target_double), 0)
        #
        # test_data = np.concatenate((test_data, test_data_double), 0)
        # test_target = np.concatenate((test_target, test_target_double), 0)




        # Display size
        # print('Doubling X_train size: ', train_data.shape)
        # print('Doubling y_train size: ', train_target.shape)
        # print('Doubling X_test size: ', test_data.shape)
        # print('Doubling y_test size: ', test_target.shape)
        # print()


        X_train = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
        X_test = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
        y_train = np_utils.to_categorical(train_target, 2)
        y_test = np_utils.to_categorical(test_target, 2)
        val_data = val_data.reshape(val_data.shape[0], -1, 1)
        val_target = np_utils.to_categorical(val_target, 2)

        # Pre-processing
        # X_train = train_data.reshape(train_data.shape[0],-1)
        # X_test = test_data.reshape(test_data.shape[0],-1)
        # y_train = np_utils.to_categorical(train_target, 2)
        # y_test = np_utils.to_categorical(test_target, 2)
        # val_data = val_data.reshape(val_data.shape[0],-1)
        # val_target = np_utils.to_categorical(val_target, 2)



        print('After pre-processing, X_train size: ', X_train.shape)
        print('After pre-processing, y_train size: ', y_train.shape)
        print('After pre-processing, X_test size: ', X_test.shape)
        print('After pre-processing, y_test size: ', y_test.shape)

        model = Sequential()

        model.add(convolutional.Conv1D(
            filters=16,
            kernel_size=1,
            padding='same',
            strides=1,
            activation='relu',
            input_shape=X_train.shape[1:]
        ))

        model.add(pooling.MaxPooling1D(
            pool_size=1,
            padding='same',
        ))

        model.add(convolutional.Conv1D(
            filters=32,
            kernel_size=2,
            padding='same',
            strides=1,
            activation='relu',
        ))

        model.add(pooling.MaxPooling1D(
            pool_size=1,
            padding='same',
        ))

        #
        # model.add(convolutional.Conv2D(
        #     filters=64,
        #     kernel_size=(5, 5),
        #     padding='same',
        #     strides=(1, 1),
        #     activation='relu',
        # ))
        #
        # model.add(pooling.MaxPooling2D(
        #     pool_size=(1, 1),
        #     padding='same',
        # ))


        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(drop_out))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(
            optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # model.fit(train_data, train_targets,epochs=10,batch_size=32,verbose=2)
        # model.fit(test_data, test_target, validation_split=0.25, validation_data=(test_data, test_target), epochs=50, batch_size=64, verbose=2)
        # model.fit(X_train, y_train, validation_split=0.25, epochs=15, batch_size=64, verbose=2)
        # model.fit(X_train, y_train, validation_split=0.25, epochs=15, batch_size=64, verbose=2)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(val_data, val_target))

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)

        # train_predict = model.predict(X_train, batch_size = 64, verbose=2)
        # test_predict = model.predict(X_test, batch_size = 64, verbose=2)

        print('The loss on testing data', test_loss)
        print('The accuracy on testing data', test_accuracy)

        val_loss, val_accuracy = model.evaluate(val_data, val_target, verbose=2)

        print('The loss on validation data', val_loss)
        print('The accuracy on validaiton data', val_accuracy)

        print()
        # print("Accuracy (Training Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " + str(accuracy_score(train_target, train_predict)))
        print()  # print()
        # print("Accuracy (Testing Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " + str(accuracy_score(test_target, test_predict)))

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
    CnnModel('./Data/hero_data/threshold_001.json', 'cnn', 'cnn', 64, 25)
    CnnModel('./Data/hero_data/threshold_002.json', 'cnn', 'cnn', 64, 25)
    CnnModel('./Data/hero_data/threshold_003.json', 'cnn', 'cnn', 64, 25)
    CnnModel('./Data/hero_data/threshold_004.json', 'cnn', 'cnn', 64, 25)
    CnnModel('./Data/hero_data/threshold_005.json', 'cnn', 'cnn', 64, 25)
    CnnModel('./Data/hero_data/full_40000_plus_data.json', 'cnn', 'cnn', 32, 20)
    CnnModel('./Data/hero_data/threshold_001.json', 'cnn', 'cnn', 32, 20)
    CnnModel('./Data/hero_data/threshold_002.json', 'cnn', 'cnn', 32, 20)
    CnnModel('./Data/hero_data/threshold_003.json', 'cnn', 'cnn', 32, 20)
    CnnModel('./Data/hero_data/threshold_004.json', 'cnn', 'cnn', 32, 20)
    CnnModel('./Data/hero_data/threshold_005.json', 'cnn', 'cnn', 32, 20)
    CnnModel('./Data/hero_data/full_40000_plus_data.json', 'cnn', 'cnn', 32, 20)

