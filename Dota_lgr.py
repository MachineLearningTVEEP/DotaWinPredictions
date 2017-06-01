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
from sklearn.linear_model import LinearRegression
from data_util import BasicHeroData, double_inverse_samples
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.sparse import bsr_matrix

from model_output import ModelOutput

def p_intercept(intercept):
    print("The intercept is {}".format(intercept))

def p_coefficients(coefficients):
    print("The coefficients are {}".format(", ".join(str(x) for x in coefficients.flatten().tolist())))



class LgrModel(ModelOutput):
    def run_model(self, data, targets, batch_size, epochs):

        data = double_inverse_samples(data)
        targets = double_inverse_samples(targets)

        train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.3, random_state=42)


        lgr = linear_model.LogisticRegression()

        # http://stackoverflow.com/questions/34337093/why-am-i-getting-a-data-conversion-warning
        n = train_target.shape[0]
        y = train_target.reshape((n,))
        print(train_data.shape)
        print(y.shape)

        train_data = bsr_matrix(train_data)

        print(train_data.shape)
        print(y.shape)

        lgr.fit(train_data, y)



        test_predict_1 = lgr.predict(test_data)
        train_predict_1 = lgr.predict(train_data)


        train_score = str(lgr.score(train_data, train_target))
        train_accuracy = str(accuracy_score(train_target, train_predict_1))
        test_score = str(lgr.score(test_data, test_target))
        test_accuracy = str(accuracy_score(test_target, test_predict_1))


        metrics = {
            'train_score': train_score,
            'train_accuracy': train_accuracy,
            'test_score': test_score,
            'test_accuracy': test_accuracy

        }
        return metrics, lgr

if __name__ == '__main__':


    LgrModel('./Data/hero_data/threshold_001.json', 'lgr', 'lgr', None, None)
    LgrModel('./Data/hero_data/threshold_002.json', 'lgr', 'lgr', None, None)
    LgrModel('./Data/hero_data/threshold_003.json', 'lgr', 'lgr', None, None)
    LgrModel('./Data/hero_data/threshold_004.json', 'lgr', 'lgr', None, None)
    LgrModel('./Data/hero_data/threshold_005.json', 'lgr', 'lgr', None, None)
    LgrModel('./Data/hero_data/full_40000_plus_data.json', 'lgr', 'lgr', None, None)