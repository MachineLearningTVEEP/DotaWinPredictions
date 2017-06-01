# The modules we're going to use
from __future__ import print_function
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import random
from random import randint


data, target = datasets.load_boston(True)

train_data, test_data, train_target, test_target = train_test_split(data, (target[:, np.newaxis]), test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(train_data, train_target)

print(train_data.shape)
print(train_target.shape)

train_prediction = reg.predict(train_data)
test_prediction = reg.predict(test_data)

print(train_prediction.shape)
print(test_prediction.shape)

def makeTrainTarget():
    arr = np.zeros(shape=(100, 1))

    for i in range(0,100):
        arr[i] = randint(0, 1)

    return arr

def makeDummyInputArray():
    arr = np.zeros(shape=(226, 1))

    rand1 = random.sample(range(1, 113), 5)
    rand2 = random.sample(range(114, 226), 5)

    for i in range(0,5):
        arr[rand1[i]] = 1
        arr[rand2[i]] = 1

    return arr



if __name__ == '__main__':
    #X = makeDummyInputArray()
    #X = X.T

    temp = []
    X = np.empty((0, 226))
    for i in range(0,100):
        a = makeDummyInputArray()
        a=a.T
        #print(X.shape)
        temp.append(a)
        X = np.append(X, a, axis=0)
        #X = np.vstack((X, a))
        #print(X.shape)


    print(X.shape)
    y = makeTrainTarget()
    print(y.shape)

    lgr = linear_model.LogisticRegression()
    lgr.fit(X, y)

    print(lgr.predict(X))

