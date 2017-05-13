# The modules we're going to use
from __future__ import print_function

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from data_util import BasicHeroData
from sklearn.metrics import accuracy_score

# (X_train,y_train),(X_test,y_test) = mnist.load_data()
h = BasicHeroData()
matches = h.read_json_file('./Data/Matches/500_matches_short.json')
h.load_data(matches)

targets = h.targets
data = h.data

# train_data, train_data, train_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=42)
train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=42)



# Display size
print('X_train size: ', train_data.shape)
print('y_train size: ', train_target.shape)
print('X_test size: ', test_data.shape)
print('y_test size: ', test_target.shape)



# Pre-processing
# X_train = train_data.reshape(train_data.shape[0],-1)/255
X_train = train_data.reshape(train_data.shape[0],-1)
# X_test = test_data.reshape(test_data.shape[0],-1)/255
X_test = test_data.reshape(test_data.shape[0],-1)
y_train = np_utils.to_categorical(train_target,2)
y_test = np_utils.to_categorical(test_target,2)
print('After pre-processing, X_train size: ', X_train.shape)
print('After pre-processing, y_train size: ', y_train.shape)
print('After pre-processing, X_test size: ', X_test.shape)
print('After pre-processing, y_test size: ', y_test.shape)



# https://keras.io/getting-started/sequential-model-guide/#examples
model = Sequential([
    #***********************************************************************what is units
    Dense(input_dim=224, units=32),
    Activation('relu'),





    Dense(units=2),
    Activation('softmax')
])

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(
    optimizer = adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



# Training
model.fit(X_train, y_train, batch_size = 32, epochs=25, verbose=2)

# Testing
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)


train_predict = model.predict(X_train, batch_size = 32, verbose=2)
test_predict = model.predict(X_test, batch_size = 32, verbose=2)

print('The loss on testing data', loss)
print('The accuracy on testing data', accuracy)


# print()
# print("Mean Accuracy (Training Data (Data / True Target) /  sklearn.linear_model.LogisticRegression.Score): " + str(model.score(train_data, train_target)))
# print()
# print("Accuracy (Training Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " + str(accuracy_score(train_target, train_predict)))
print()
# print("Mean Accuracy (Testing Data (Data / True Target) /  sklearn.linear_model.LogisticRegression.Score): " + str(lgr.score(test_data, test_target)))
# print()
# print("Accuracy (Testing Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " + str(accuracy_score(test_target, test_predict)))
