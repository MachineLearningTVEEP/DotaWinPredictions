# The modules we're going to use
from __future__ import print_function

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.sparse import bsr_matrix, csr_matrix

from model_output import ModelOutput
from data_util import BasicHeroData, double_inverse_samples


class SvmModel(ModelOutput):
    def run_model(self, data, targets, batch_size, epochs):

        #data = double_inverse_samples(data)
        #targets = double_inverse_samples(targets)

        # split the data up into multiple sets: training, testing
        train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.4, random_state=42)
        # create svm object using original one-vs-one (ovo)
        s_machine = svm.SVC(decision_function_shape='ovo')
        # http://stackoverflow.com/questions/34337093/why-am-i-getting-a-data-conversion-warning
        # fixed data to avoid compiler warning
        n = train_target.shape[0]
        y = train_target.reshape((n,))
        # fit the data int othe svm object
        s_machine.fit(train_data, y)
        # get score on training and test data
        train_score = str(s_machine.score(train_data, train_target))

        test_score = str(s_machine.score(test_data, test_target))

        # collect metrics for output
        metrics = {
            'train_score': train_score,
            'test_score': test_score,
        }
        return metrics, s_machine

if __name__ == '__main__':
    #SvmModel('./Data/hero_data/threshold_001.json', 'svm', 'svm', None, None)
    #SvmModel('./Data/hero_data/threshold_002.json', 'svm', 'svm', None, None)
    SvmModel('./Data/hero_data/threshold_003.json', 'svm', 'svm', None, None)
    #SvmModel('./Data/hero_data/threshold_004.json', 'svm', 'svm', None, None)
    #SvmModel('./Data/hero_data/threshold_005.json', 'svm', 'svm', None, None)
    #SvmModel('./Data/hero_data/full_40000_plus_data.json', 'svm', 'svm', None, None)

