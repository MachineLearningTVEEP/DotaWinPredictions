# The modules we're going to use
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from data_util import BasicHeroData


import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
import matplotlib




# targets_double = np.copy(targets)
# data_double = np.copy(data)
#
# data = np.concatenate((data, data_double))
# targets = np.concatenate((targets, targets_double))

def cnn(data, targets, modelfile=None):


    # train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=42)
    train_data, data_set_2, train_target, target_set_2 = train_test_split(data, targets, test_size=0.25, random_state=42)
    test_data, val_data, test_target, val_target = train_test_split(data_set_2, target_set_2, test_size=0.15, random_state=24)


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
    val_data = val_data.reshape(val_data.shape[0],-1, 1)
    val_target = np_utils.to_categorical(val_target, 2)
	
    # Pre-processing
    #X_train = train_data.reshape(train_data.shape[0],-1)
    #X_test = test_data.reshape(test_data.shape[0],-1)
    #y_train = np_utils.to_categorical(train_target, 2)
    #y_test = np_utils.to_categorical(test_target, 2)
    #val_data = val_data.reshape(val_data.shape[0],-1)
    #val_target = np_utils.to_categorical(val_target, 2)



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
    model.add(Dropout(.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))



    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(
        optimizer = adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # model.fit(train_data, train_targets,epochs=10,batch_size=32,verbose=2)
    # model.fit(test_data, test_target, validation_split=0.25, validation_data=(test_data, test_target), epochs=50, batch_size=64, verbose=2)
    # model.fit(X_train, y_train, validation_split=0.25, epochs=15, batch_size=64, verbose=2)
    # model.fit(X_train, y_train, validation_split=0.25, epochs=15, batch_size=64, verbose=2)
    model.fit(X_train, y_train, batch_size = 64, epochs=25, verbose=2, validation_data=(val_data, val_target))

    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)

    # train_predict = model.predict(X_train, batch_size = 64, verbose=2)
    # test_predict = model.predict(X_test, batch_size = 64, verbose=2)

    print('The loss on testing data', loss)
    print('The accuracy on testing data', accuracy)

    loss, accuracy = model.evaluate(val_data, val_target, verbose=2)

    print('The loss on validation data', loss)
    print('The accuracy on validaiton data', accuracy)

    print()
    # print("Accuracy (Training Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " + str(accuracy_score(train_target, train_predict)))
    print()  # print()
    # print("Accuracy (Testing Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " + str(accuracy_score(test_target, test_predict)))
    print('test loss:', loss)
    print('test accuracy', accuracy)

    if modelfile:
        if modelfile.endswith('.h5'):
            model.save('./models/{}'.format(modelfile))
        else:
            print ("Can't save your model; bad extension")

if __name__ == '__main__':


    h = BasicHeroData()
    d = h.load_saved_hero_data('./Data/hero_data/full_40000_plus_data.json')
    data, targets, features, target_labels = d
    cnn(data, targets)














#
