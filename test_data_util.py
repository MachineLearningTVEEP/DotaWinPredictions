import unittest
from data_util import DotaData, drop_features
import numpy as np

class TestDotaData(unittest.TestCase):
    def setUp(self):
        super(TestDotaData, self).setUp()
        self.dota_data = DotaData()

    def test_feature_extract(self):
        a = dict((k,k) for k in [1,2,3])
        b = dict((k,k) for k in [2,1,4])
        c = dict((k,k) for k in [1,2,5,6,4])
        d = dict((k,k) for k in [1,7])

        data = [a,b,c]
        features, extras = self.dota_data.extract_base_features(data)
        self.assertEqual(features, set([1,2]))
        self.assertEqual(extras, set([3,4,5,6]))

        data = [a,b,c,d]
        features, extras = self.dota_data.extract_base_features(data)
        self.assertEqual(features, set([1]))
        self.assertEqual(extras, set([2,3,4,5,6,7]))

class TestDrop(unittest.TestCase):
    def test_drop_features(self):
        data = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1], [1,0,1,0], [0,1,0,1], [1,1,1,0], [0,1,1,1]])
        targets = np.array([1,1,0,0,1,0,0,0]).reshape(8,1)
        percentages = np.array([1, .4])
        threshold = .5
        features = ['zero', 'one', 'two', 'three']
        data, targets, features = drop_features(data, targets, features, percentages, threshold)
        self.assertEqual(len(data), len(targets))
        self.assertEqual(3, len(data))
        self.assertEqual(set(np.ndarray.tolist(targets.flatten())), set([1]))

if __name__ == '__main__':
    unittest.main()