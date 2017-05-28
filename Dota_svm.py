# The modules we're going to use
#from __future__ import print_function
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import random
from random import randint

from data_util import BasicHeroData, double_inverse_samples
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm
from scipy.sparse import bsr_matrix, csr_matrix

from model_output import ModelOutput


class SvmModel(ModelOutput):
    def run_model(self, data, targets, batch_size, epochs):
        # targets_double = np.copy(targets)
        # data_double = np.copy(data)
        #
        # data = np.concatenate((data, data_double))
        # targets = np.concatenate((targets, targets_double))

        print "1"

        data = double_inverse_samples(data)
        targets = double_inverse_samples(targets)
        print "2"

        # train_data, train_data, train_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=42)
        train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.4, random_state=42)
        print "3"

        s_machine = svm.SVC(decision_function_shape='ovo')
        print "4"
        # http://stackoverflow.com/questions/34337093/why-am-i-getting-a-data-conversion-warning
        n = train_target.shape[0]
        y = train_target.reshape((n,))


        # ********************************************************************************************************
        # print(train_data.shape)
        # print(y.shape)
        #
        # train_data = bsr_matrix(train_data)
        # # y = bsr_matrix(y)
        # # y = csr_matrix(y)
        # print(train_data.shape)
        # print(y.shape)


        s_machine.fit(train_data, y)
        print "5"
        # s_machine.fit(train_data, train_target)
        #test_predict_1 = s_machine.predict(test_data)
        #train_predict_1 = s_machine.predict(train_data)
        print "6"


        '''print()
                                print("Accuracy (Training Data (Data / True Target) /  sklearn.svm.SVC.Score): " + str(s_machine.score(train_data, train_target)))
                                print()
                                print("Accuracy (Training Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " +  str(accuracy_score(train_target, train_predict_1)))
                        
                        
                        
                        
                        
                                print()
                                print("Accuracy (Testing Data (Data / True Target) /  sklearn.svm.SVC.Score): " + str(s_machine.score(test_data, test_target)))
                                print()
                                print("Accuracy (Testing Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " +  str(accuracy_score(test_target, test_predict_1)))
                        
                        '''
        print()
        train_score = str(s_machine.score(train_data, train_target))
        print()
        #train_accuracy = str(accuracy_score(train_target, train_predict_1))





        print()
        test_score = str(s_machine.score(test_data, test_target))
        print()
        #test_accuracy = str(accuracy_score(test_target, test_predict_1))

        metrics = {
            'train_score': train_score,
            #'train_accuracy': train_accuracy,
            'test_score': test_score,
            #'test_accuracy': test_accuracy
        }
        print "7"
        return metrics, s_machine


if __name__ == '__main__':

    SvmModel('./Data/hero_data/threshold_001.json', 'svm', 'svm', None, None)
    SvmModel('./Data/hero_data/threshold_002.json', 'svm', 'svm', None, None)
    SvmModel('./Data/hero_data/threshold_003.json', 'svm', 'svm', None, None)
    SvmModel('./Data/hero_data/threshold_004.json', 'svm', 'svm', None, None)
    SvmModel('./Data/hero_data/threshold_005.json', 'svm', 'svm', None, None)
    SvmModel('./Data/hero_data/full_40000_plus_data.json', 'svm', 'svm', None, None)
