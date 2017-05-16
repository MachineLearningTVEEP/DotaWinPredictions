import requests
import numpy as np
from time import sleep
import json


class NumpyException(Exception):
    pass


class SubDictException(Exception):
    pass


class DotaData(object):
    def __init__(self):
        self.base_api = "https://api.opendota.com/api/"

    def get(self, api):
        '''
        Uses the requests module to get data from the api 
        Returns the python object that corresponds to the api's json 
        '''
        r = requests.get("{}{}".format(self.base_api, api))
        # assert r.status_code == 200

        if(r.status_code == 200):
            return r.status_code, r.json()
        else:
            return r.status_code, None

        # return r.json()

    def extract_base_features(self, data):
        '''
        Extracts keys (features) from a list of dicts 
        Returns a set of features that all dicts contain, 
            as well as a set of all the extra keys 
        (extra keys are supplied to show how messed up opendota's api is, 
            only features in the base_feature_set should be used)
        '''
        base_feature_set = set(data[0].keys())
        extras = set()
        for d in data:
            f = set(d.keys())
            extras = extras.union(base_feature_set.symmetric_difference(f))
            base_feature_set = base_feature_set.intersection(f)

        return base_feature_set, extras

    def np_ize(self, data, np_only=False):
        '''
        Turns a list of dicts into an np array 
        Returns only the subset of keys that belong to all dicts in the list, plus a numpy array
        '''
        if np_only is True:
            return np.array(data)
        if isinstance(data, list):
            features = data[0].keys()
            if all([d.keys() == features for d in data]):
                l = [[v for k, v in sorted(d.items())] for d in data]
                return features, np.array(l)
            else:
                base_feature_set, extra_features = self.extract_base_features(data)
                l = [[v for k, v in sorted(d.items()) if k in base_feature_set] for d in data]
                return base_feature_set, np.array(l)
        raise NumpyException("Unable to transform data into numpy array")

    def sub_dicts(self, data, desired_keys):
        '''
        Transforms dictionaries into sub dictionaries
        Returns a list of dictionaries comprised of the desired_keys
        '''
        base_feature_set, extras = self.extract_base_features(data)
        if all(k in base_feature_set for k in desired_keys):
            return [{k: d[k] for k in desired_keys} for d in data]
        raise SubDictException("Unable to extract sub dict. Not all keys exist in every member of the data set.")

    def read_json_file(self, filepath):
        '''
        Reads a json file 
        Returns a python object that corresponds to the file's json
        '''
        with open(filepath, 'r') as f:
            return json.load(f)

    def write_json_file(self, filepath, data):
        '''
        Writes a json file; writes json of python object
        '''
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def shorten_data(self, data, desired_keys):
        '''
        Takes in data as a list of dicts, and desired_keys, which mimics the structure of the dicts to be returned
        Returns a list of dicts with the structure of desired_keys (2 levels only)
        '''
        assert isinstance(desired_keys, dict)
        _data = self.sub_dicts(data, desired_keys.keys())
        for d in _data:
            # for key, value in desired_keys.iteritems():
            for key, value in desired_keys.items():
                if value is not None:
                    assert isinstance(value, list)
                    assert isinstance(d[key], list)
                    d[key] = self.sub_dicts(d[key], value)
        return _data


class BasicHeroData(DotaData):
    '''
    Basic Usage: Get dota data however you can, be it loaded from a file or directly from the api
        then call load_data on it
    '''

    def __init__(self):
        super(BasicHeroData, self).__init__()
        self.hero_features, self.hero_id_index_map = self.heroes()
        self.target_labels = ['radiant_win', 'dire_win']

    def heroes(self):
        heroes = self.read_json_file('./Data/heroes.json')['heroes']
        id_name_map = {h['id']: h['name'] for h in heroes}
        ids = id_name_map.keys()
        id_index_map = {x: i for i, x in enumerate(ids)}
        hero_features = [''] * (2 * (len(id_index_map)))
        for i, id in enumerate(id_index_map):
            hero_features[i] = '{}_{}'.format(id_name_map[id], 'radiant')
            hero_features[i + len(id_index_map)] = '{}_{}'.format(id_name_map[id], 'dire')
        return hero_features, id_index_map

    def process_matches(self):
        targets = []
        data = []
        for match in self.shortened_data:
            datum = [0] * (2 * len(self.hero_id_index_map))
            for player in match['players']:
                try:
                    if player['isRadiant'] is True:
                        index = self.hero_id_index_map[player['hero_id']]
                    else:
                        index = self.hero_id_index_map[player['hero_id']] + len(self.hero_id_index_map)
                    datum[index] = 1

                except KeyError:  # there is some data that has hero_ids that aren't in the heroes.json
                    datum = [0] * len(datum)
                    break
            if any([x != 0 for x in datum]):
                data.append(datum)
                # targets.append([int(match['radiant_win']), int(not match['radiant_win'])])
                # targets.append([int(match['radiant_win'])])
                targets.append([int(match['radiant_win'])])
        return data, targets

    def load_data(self, matches):
        self.shortened_data = self.shorten_data(matches, {'players': ['isRadiant', 'hero_id'], 'radiant_win': None})
        data, targets = self.process_matches()
        self.raw_data = data
        self.data = self.np_ize(data, True)
        self.targets = self.np_ize(targets, True)

def gatherdata(write_path, read_path):
    h = BasicHeroData()

    matches_by_id = h.read_json_file(read_path)
    # matches_by_id = h.read_json_file('./Data/Matches_By_Id/10_matches.json')
    # features, _data = h.np_ize(matches_by_id)
    matches = []
    match_ids = [datum['match_id'] for datum in matches_by_id]
    for mid in match_ids:
        # for mid in match_ids:
        print('Getting: ' + str(mid))

        status, code = h.get("matches/{}".format(mid))

        if(status == 200):
            matches.append(code)

        # matches.append(h.get("matches/{}".format(mid)))
        sleep(1.2)  # the opendota api requests that this endpoint only be hit 1/s
    # h.write_json_file('./Data/Matches/45852_matches_full.json', matches)

    h.load_data(matches)

    features = h.hero_features
    data = h.data
    target_labels = h.target_labels
    targets = h.targets

    # assert len(data[0]) == len(features)
    # assert len(targets[0]) == len(target_labels)

    h.write_json_file(write_path, h.shortened_data)

if __name__ == "__main__":
    print('Creating Data')
    gatherdata(write_path='./Data/Matches/10000_matches_short.json', read_path='./Data/Matches_By_Id/10000_matches.json')
