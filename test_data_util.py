import unittest
import json
from data_util import DotaData

def p(obj):
    print json.dumps(obj, indent=4)

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

    def test_shorten_dict(self):
        desired = {
            'b': None,
            'c': ['d', 'e', 'f'],
            'g': None,
            'h': None
        }
        actual_d = {
            'b': 0,
            'c': [{
                'd': 0,
                'e': 0,
                'f': 0,
                'a': []
            }],
            'g': 0,
            'h': 0,
            'j': {},
            'k': []
        }
        actual_l = [actual_d]
        p(actual_l)
        data = self.dota_data.shorten_data(actual_l, desired)
        p(data)

        for d in data:
            self.assertEqual(len(d), len(desired))
            self.assertEqual(d.keys(), desired.keys())
            for k,v in d.iteritems():
                if isinstance(v, list):
                    for _d in v:
                        self.assertEqual(len(_d), len(desired[k]))
                        self.assertEqual(sorted(_d.keys()), sorted(desired[k]))
                        for _k, _v in _d.iteritems():
                            self.assertEqual(_v, actual_d[k][0][_k])
                else:
                    self.assertEqual(v, actual_d[k])



if __name__ == '__main__':
    unittest.main()