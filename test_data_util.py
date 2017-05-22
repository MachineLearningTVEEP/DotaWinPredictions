import unittest
from data_util import DotaData, BasicHeroData
import numpy as np
import os

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

class TestHeroData(unittest.TestCase):
    def setUp(self):
        super(TestHeroData, self).setUp()
        self.h = BasicHeroData()

    def test_drop_features(self):
        data = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1], [1,0,1,0], [0,1,0,1], [1,1,1,0], [0,1,1,1]])
        targets = np.array([1,1,0,0,1,0,0,0]).reshape(8,1)
        self.h.percentages = np.array([1, .4])
        threshold = .5
        features = ['zero', 'one', 'two', 'three']
        data, targets, features = self.h._drop_features(data, targets, features, threshold)
        self.assertEqual(len(data), len(targets))
        self.assertEqual(3, len(data))
        self.assertEqual(set(np.ndarray.tolist(targets.flatten())), set([1]))

    def test_gather_chunked_data(self):
        filename = 'test.json'
        one = self.h.read_json_file('./Data/Matches/chunked/1.json')
        self.assertEqual(len(one), 1000)
        two = self.h.read_json_file('./Data/Matches/chunked/2.json')
        self.assertEqual(len(two), 1000)
        remainder = self.h.read_json_file('./Data/Matches/chunked/remainder.json')
        self.assertEqual(len(remainder), 806)
        
        self.h._gather_chunked_data(2, filename)
        data = self.h.read_json_file('./Data/Matches/{}'.format(filename))
        self.assertEqual(len(data), len(one) + len(remainder))

        self.h._gather_chunked_data(3, filename)
        data = self.h.read_json_file('./Data/Matches/{}'.format(filename))
        self.assertEqual(len(data), len(one) + len(two) + len(remainder))

        if os.path.isfile('./Data/Matches/{}'.format(filename)):
            os.remove('./Data/Matches/{}'.format(filename))

    def test_heroes(self):
        hero_features, id_index_map = self.h.heroes()
        self.assertEqual(len(id_index_map), 113)
        self.assertEqual(len(hero_features), len(id_index_map)*2)
        self.assertTrue(isinstance(hero_features, list))
        self.assertTrue(isinstance(hero_features[0], str))
        self.assertTrue(isinstance(id_index_map, dict))

    def test_load_heroes(self):
        self.h._load_heroes()
        heroes = self.h.read_json_file('./Data/heroes.json')
        self.assertEqual(len(heroes), 113)
        self.assertTrue(isinstance(heroes, list))
        self.assertTrue(isinstance(heroes[0], dict))
        self.assertEqual(len(heroes[0].keys()), 3)
        for i in ['localized_name', 'name', 'id']:
            self.assertIn(i, heroes[0].keys())

    def test_process_matches(self):
        #file has one hero that does not have an id
        self.h.shortened_data = self.h.read_json_file('./Data/test_data/10_matches_short.json')
        self.assertEqual(len(self.h.shortened_data), 10)

        data, targets = self.h.process_matches()
        self.assertTrue(isinstance(data, list))
        self.assertTrue(isinstance(targets, list))
        self.assertEqual(len(data), 9)
        self.assertEqual(len(targets), 9)
        self.assertEqual(len(data[0]), 226)

    def test_load_data(self):
        #file has one hero that does not have an id
        matches = self.h.read_json_file('./Data/test_data/10_matches_full.json')
        self.h.load_data(matches)
        data = self.h.data 
        targets = self.h.targets
        raw_data = self.h.raw_data
        raw_targets = self.h.raw_targets

        self.assertTrue(isinstance(raw_data, list))
        self.assertTrue(isinstance(raw_targets, list))
        self.assertTrue(isinstance(data, np.ndarray))
        self.assertTrue(isinstance(targets, np.ndarray))
        self.assertEqual(len(data), 9)
        self.assertEqual(len(targets), 9)
        self.assertEqual(len(data[0]), 226)

        #works with already shortened data as well
        matches = self.h.read_json_file('./Data/test_data/10_matches_short.json')
        self.h.load_data(matches)
        data = self.h.data 
        targets = self.h.targets
        raw_data = self.h.raw_data
        raw_targets = self.h.raw_targets

        self.assertTrue(isinstance(raw_data, list))
        self.assertTrue(isinstance(raw_targets, list))
        self.assertTrue(isinstance(data, np.ndarray))
        self.assertTrue(isinstance(targets, np.ndarray))
        self.assertEqual(len(data), 9)
        self.assertEqual(len(targets), 9)
        self.assertEqual(len(data[0]), 226)




if __name__ == '__main__':
    unittest.main()















