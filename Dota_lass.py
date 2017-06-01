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
from data_util import BasicHeroData
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from model_output import ModelOutput


class LassModel(ModelOutput):
    def run_model(self, data, targets, batch_size, epochs):
        #get doubled, inverted data
        data = double_inverse_samples(data)
        targets = double_inverse_samples(targets)
        # split the data up into multiple sets: training, testing 
        train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.3, random_state=42)

        # create Lasso model
        lass = linear_model.Lasso(alpha=0.001, max_iter=10000)

        # http://stackoverflow.com/questions/34337093/why-am-i-getting-a-data-conversion-warning
        # fixed data to avoid compiler warning
        n = train_target.shape[0]
        y = train_target.reshape((n,))
        # fit data to model
        lass.fit(train_data, y)
        # make prediciton on test and training data
        test_predict_1 = lgr.predict(test_data)
        train_predict_1 = lgr.predict(train_data)
        print()
        print()
        # get acc on train and test data
        train_score = str(lgr.score(train_data, train_target))
        train_accuracy = str(accuracy_score(train_target, train_predict_1))
        test_score = str(lgr.score(test_data, test_target))
        test_accuracy = str(accuracy_score(test_target, test_predict_1))
        #print acc
        print("Mean Accuracy (Training Data (Data / True Target) /  sklearn.linear_model.LogisticRegression.Score): " + str(lgr.score(train_data, train_target)))
        print()
        print("Accuracy (Training Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " +  str(accuracy_score(train_target, train_predict_1)))
        print()
        print("Mean Accuracy (Testing Data (Data / True Target) /  sklearn.linear_model.LogisticRegression.Score): " + str(lgr.score(test_data, test_target)))
        print()
        print("Accuracy (Testing Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " +  str(accuracy_score(test_target, test_predict_1)))
        # collect metrics for output
        metrics = {
            'train_score': str(lass.score(train_data, train_target)),
            'test_score': str(lass.score(test_data, test_target))
        }
        return metrics, lass


if __name__ == '__main__':
    # run model with various thresholds and epoch/batch sizes

    #LassModel('./Data/hero_data/threshold_001.json', 'lass', 'lass', None, None)
    #LassModel('./Data/hero_data/threshold_002.json', 'lass', 'lass', None, None)
    LassModel('./Data/hero_data/threshold_003.json', 'lass', 'lass', None, None)
    #LassModel('./Data/hero_data/threshold_004.json', 'lass', 'lass', None, None)
    #LassModel('./Data/hero_data/threshold_005.json', 'lass', 'lass', None, None)
    #LassModel('./Data/hero_data/full_40000_plus_data.json', 'lass', 'lass', None, None)
