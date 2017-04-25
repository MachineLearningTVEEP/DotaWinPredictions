import unittest
from data_util import DotaData

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

if __name__ == '__main__':
    unittest.main()