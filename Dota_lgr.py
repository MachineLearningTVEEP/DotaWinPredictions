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

def p_intercept(intercept):
	print("The intercept is {}".format(intercept))

def p_coefficients(coefficients):
	print("The coefficients are {}".format(", ".join(str(x) for x in coefficients.flatten().tolist())))

def mse(target, predictions):
	'''mean squared error according to slides (ie, div by 2N)'''
	return ((target - predictions)**2).mean() / 2

def p_errors(test_target, train_target, test_predictions, train_predictions):
	print("The mean squared error of the training dataset is {}".format(mse(train_target, train_predictions)))
	print("The mean squared error of the testing dataset is {}".format(mse(test_target, test_predictions)))

def plot_fig(target, predictions):
	plt.scatter(target, predictions)
	plt.xlabel("Real Wins")
	plt.ylabel("Predicted Wins")
	plt.show()



if __name__ == '__main__':
    h = BasicHeroData()
    matches = h.read_json_file('./Data/Matches/10_matches_short.json')
    h.load_data(matches)

    targets = h.targets
    data = h.data

    train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=42)

    lgr = linear_model.LogisticRegression()

    # http://stackoverflow.com/questions/34337093/why-am-i-getting-a-data-conversion-warning
    n = train_target.shape[0]
    y = train_target.reshape((n,))
    lgr.fit(train_data, y)

    p_intercept(lgr.intercept_[0])
    p_coefficients(lgr.coef_)

    test_predict_1 = lgr.predict(test_data)
    train_predict_1 = lgr.predict(train_data)

    p_errors(test_target, train_target, test_predict_1, train_predict_1)

    print(test_target)
    print()
    print(test_predict_1)

    # plot_fig(test_target, test_predict_1)

    # plot_fig(test_target, train_predict_1)
