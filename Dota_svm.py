# The modules we're going to use
from __future__ import print_function

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_util import BasicHeroData

def _svm(data, targets, modelfile=None):
    # split the data up into multiple sets: training, testing validation
    train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.4, random_state=42)
    # create svm object using original one-vs-one (‘ovo’)
    s_machine = svm.SVC(decision_function_shape='ovo')
    # http://stackoverflow.com/questions/34337093/why-am-i-getting-a-data-conversion-warning
    # fixed data to avoid compiler warning
    n = train_target.shape[0]
    y = train_target.reshape((n,))
    # fit the data int othe svm object
    s_machine.fit(train_data, y)
    # make predicitons
    test_predict_1 = s_machine.predict(test_data)
    train_predict_1 = s_machine.predict(train_data)
    # print results on test and training data
    print()
    print("Accuracy (Training Data (Data / True Target) /  sklearn.svm.SVC.Score): " + str(s_machine.score(train_data, train_target)))
    print()
    print("Accuracy (Training Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " +  str(accuracy_score(train_target, train_predict_1)))
    print()
    print("Accuracy (Testing Data (Data / True Target) /  sklearn.svm.SVC.Score): " + str(s_machine.score(test_data, test_target)))
    print()
    print("Accuracy (Testing Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " +  str(accuracy_score(test_target, test_predict_1)))

if __name__ == '__main__':
    # create hero object
    h = BasicHeroData()
    # load datafrom dataset
    d = h.load_saved_hero_data('./Data/hero_data/full_40000_plus_data.json')
    # seperate the data by types
    data, targets, features, target_labels = d
    # run the machine learning svm
    _svm(data, targets)

