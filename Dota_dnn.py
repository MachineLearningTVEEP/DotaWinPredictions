# The modules we're going to use
from __future__ import print_function

from keras import Input
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, GaussianNoise, GaussianDropout
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from tensorflow import Tensor

from data_util import BasicHeroData
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
import numpy as np



def dnn(data, targets, modelfile=None):

        # (X_train,y_train),(X_test,y_test) = mnist.load_data()
    # h = BasicHeroData()
    # matches = h.read_json_file('./Data/Matches/40k_matches_short.json')
    # h.load_data(matches)
    #
    # targets = h.targets
    # data = h.data

    # targets_double = np.copy(targets)
    # data_double = np.copy(data)
    #
    # data = np.concatenate((data, data_double))
    # targets = np.concatenate((targets, targets_double))


    # data[data == 0] = -1

    # train_data, train_data, train_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=42)


    # split up two groups, one beting the data, the other whil split up furture to a valdiation set and test set, no overlapping data
    train_data, data_set_2, train_target, target_set_2 = train_test_split(data, targets, test_size=0.25, random_state=42)
    test_data, val_data, test_target, val_target = train_test_split(data_set_2, target_set_2, test_size=0.15, random_state=24)

    # make two copies of the data
    target_double = np.copy(train_target)
    data_double = np.copy(train_data)

    # reverse the second set of data
    # target_double = target_double[::-1]
    # data_double = data_double[::-1]
    #
    #
    #
    # # make two copies of the data
    # test_target_double = np.copy(test_target)
    # test_data_double = np.copy(test_data)
    #
    # # reverse the second set of data
    # test_target_double = test_target_double[::-1]
    # test_data_double = test_data_double[::-1]




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


    # Pre-processing
    #     train_data = train_data.reshape(-1, 1, 48, 48)
    X_train = train_data.reshape(train_data.shape[0],-1)
    X_test = test_data.reshape(test_data.shape[0],-1)
    y_train = np_utils.to_categorical(train_target, 2)
    y_test = np_utils.to_categorical(test_target, 2)
    val_data = val_data.reshape(val_data.shape[0],-1)
    val_target = np_utils.to_categorical(val_target, 2)



    print('After pre-processing, X_train size: ', X_train.shape)
    print('After pre-processing, y_train size: ', y_train.shape)
    print('After pre-processing, X_test size: ', X_test.shape)
    print('After pre-processing, y_test size: ', y_test.shape)

    print()

    # https://keras.io/getting-started/sequential-model-guide/#examples
    # model = Sequential([
    #     #***********************************************************************what is units
    #     Dense(input_dim=224, units=32),
    #     Activation('relu'),
    #
    #
    #
    #
    #
    #     Dense(units=2),
    #     Activation('softmax')
    # ])

    model = Sequential()

    # from keras.regularizers import activity_l1
    from keras import regularizers
    from keras.models import Model





    inputs = Input(shape=(226, ))

    # h = Dense(64, activation='sigmoid', activity_regularizer=activity_l1(1e-5))(inputs)
    h = Dense(64, activation='sigmoid', activity_regularizer=regularizers.l1(1e-5))(inputs)
    outputs = Dense(226)(h)
    model = Model(input=inputs, output=outputs)
    # model = Model(input=Tensor(inputs), output=Tensor(outputs))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # model.compile(optimizer='adam', loss='mse')

    model.compile(
        optimizer=adam,
        # loss='categorical_crossentropy',
        loss='mse',
        metrics=['accuracy']
    )

    # model.fit(X, X, batch_size=64, nb_epoch=5)

    model.fit(X_train, X_train, batch_size=128, epochs=15, verbose=2, validation_data=(val_data, val_data))
    # model.fit(X_train, y_train, batch_size=128, epochs=50, verbose=2, validation_data=(val_data, val_target))
    # model.fit(X_train, y_train, batch_size=128, epochs=50, verbose=2)
    # model.fit(X_train, X_train, batch_size=128, epochs=50, verbose=2)


    # model.add(Dense(4096, activation='relu', input_dim=train_data.shape[1]))
    # model.add(GaussianNoise(0.5))
    # model.add(GaussianDropout(.1))
    # model.add(Dense(32, activation='relu'))
    # model.add(GaussianNoise(0.5))
    # model.add(Dense(4096, activation='relu'))
    # model.add(GaussianDropout(.1))
    #






    # model.add(Dense(2, activation='softmax'))
    #
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #
    # model.compile(
    #     optimizer = adam,
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy']
    # )

    # model.compile(
    #     optimizer='rmsprop',
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy'])

    # For a binary classification problem
    # model.compile(
    #     optimizer='rmsprop',
    #     loss='binary_crossentropy',
    #     metrics=['accuracy'])


    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['accuracy'])
    #


    # Training
    # model.fit(X_train, y_train, batch_size = 128, epochs=50, verbose=2, validation_data=(val_data, val_target) )


    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)


    # train_predict = model.predict(X_train, batch_size = 64, verbose=2)
    # test_predict = model.predict(X_test, batch_size = 64, verbose=2)

    # print('The loss on testing data', loss)
    # print('The accuracy on testing data', accuracy)
    #
    # loss, accuracy = model.evaluate(val_data, val_target, verbose=2)
    #
    # print('The loss on validation data', loss)
    # print('The accuracy on validaiton data', accuracy)

    # print()
    # # print("Accuracy (Training Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " + str(accuracy_score(train_target, train_predict)))
    # print()# print()
    # # print("Accuracy (Testing Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " + str(accuracy_score(test_target, test_predict)))
    # print('test loss:', loss)
    # print('test accuracy', accuracy)

    if modelfile:
        if modelfile.endswith('.h5'):
            model.save('./models/{}'.format(modelfile))
        else:
            print ("Can't save your model; bad extension")

if __name__ == '__main__':


    h = BasicHeroData()
    d = h.load_saved_hero_data('./Data/hero_data/threshold_003.json')
    # d = h.load_saved_hero_data('./Data/hero_data/full_40000_plus_data.json')
    data, targets, features, target_labels = d
    dnn(data, targets)


