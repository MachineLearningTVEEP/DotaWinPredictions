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

from data_util import BasicHeroData
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm


h = BasicHeroData()
matches = h.read_json_file('./Data/Matches/5000_matches_short.json')
h.load_data(matches)

targets = h.targets
data = h.data

# train_data, train_data, train_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=42)
train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=42)

s_machine = svm.SVC(decision_function_shape='ovo')
# http://stackoverflow.com/questions/34337093/why-am-i-getting-a-data-conversion-warning
n = train_target.shape[0]
y = train_target.reshape((n,))
s_machine.fit(train_data, y)
# s_machine.fit(train_data, train_target)

test_predict_1 = s_machine.predict(test_data)
train_predict_1 = s_machine.predict(train_data)


print()
print("Mean Accuracy (Training Data (Data / True Target) /  sklearn.svm.SVC.Score): " + str(s_machine.score(train_data, train_target)))
print()
print("Accuracy (Training Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " +  str(accuracy_score(train_target, train_predict_1)))
print()
print("Mean Accuracy (Testing Data (Data / True Target) /  sklearn.svm.SVC.Score): " + str(s_machine.score(test_data, test_target)))
print()
print("Accuracy (Testing Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " +  str(accuracy_score(test_target, test_predict_1)))
