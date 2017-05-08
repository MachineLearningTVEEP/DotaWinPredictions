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


def make_train_target(num_samples):
    arr = np.zeros(shape=(num_samples, 1))

    for i in range(0,num_samples):
        arr[i] = randint(0, 1)

    return arr

def make_dummy_input_array(num_samples):

    X = np.empty((0, 226))
    for i in range(0,num_samples):

        arr = np.zeros(shape=(226, 1))

        rand1 = random.sample(range(1, 113), 5)
        rand2 = random.sample(range(114, 226), 5)

        for i in range(0,5):
            arr[rand1[i]] = 1
            arr[rand2[i]] = 1

        arr = arr.T
        X = np.append(X, arr, axis=0)

    return X



if __name__ == '__main__':

    # train data
    X = make_dummy_input_array(100)
    print(X.shape)

    # target data
    y = make_train_target(100)
    print(y.shape)

    # make & train log reg model
    lgr = linear_model.LogisticRegression()
    lgr.fit(X, y)

    # prediction
    hero_config_to_predict = make_dummy_input_array(1)
    print("Logistic Reg predicted " + str(lgr.predict(hero_config_to_predict)))


    # SVM
    """
    # make 100 hero selection samples
    X2 = []
    for i in range(0,100):
        a = make_dummy_input_array(1)
        X2.append(a)

    # 100 targer win/loss for the 100 hero selection samples
    y2 = make_train_target(100)
    """
    clf = svm.SVC(gamma=0.001, C=100, )

    # pedict with a sample hero selection
    clf.fit(X, y)

    print("SVM predicted " + str(clf.predict(hero_config_to_predict)))
