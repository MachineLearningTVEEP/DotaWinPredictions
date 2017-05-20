# The modules we're going to use
from keras.datasets import mnist
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


h = BasicHeroData()
matches = h.read_json_file('./Data/Matches/5000_matches_short.json')
print('size 10: ' + str(len(matches)))
h.load_data(matches)

targets = h.targets
data = h.data
print('size 11: ' + str(len(data)))


# targets_double = np.copy(targets)
# data_double = np.copy(data)
#
# data = np.concatenate((data, data_double))
# targets = np.concatenate((targets, targets_double))


train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=42)

# Display size
print('Before pre-processing, X_train size: ', train_data.shape)
print('Before pre-processing, y_train size: ', train_target.shape)
print('Before pre-processing, X_test size: ', test_target.shape)
print('Before pre-processing, y_test size: ',  test_target.shape)

# (nb_of_examples, nb_of_features, 1).
# http://stackoverflow.com/questions/43235531/convolutional-neural-network-conv1d-input-shape
X_train = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
X_test = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
y_train = np_utils.to_categorical(train_target, 2)
y_test = np_utils.to_categorical(test_target, 2)

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
model.fit(X_train, y_train, batch_size = 64, epochs=25, verbose=2, validation_data=(X_test, y_test) )



loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print('test loss:', loss)
print('test accuracy', accuracy)















#
