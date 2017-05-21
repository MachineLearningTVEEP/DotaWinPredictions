import requests
import numpy as np
from time import sleep
import json
import matplotlib.pyplot as plt


class NumpyException(Exception):
    pass


class SubDictException(Exception):
    pass

def pp(obj):
    print(json.dumps(obj, indent=4))

class DotaData(object):
    def __init__(self):
        self.base_api = "https://api.opendota.com/api/"

    def get(self, api):
        '''
        Uses the requests module to get data from the api 
        Returns the python object that corresponds to the api's json 
        '''

        # http://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
        # http://stackoverflow.com/questions/7160983/catching-all-exceptions-in-python


        try:
            r = requests.get("{}{}".format(self.base_api, api))

            if(r.status_code == 200):
                return r.status_code, r.json()
            else:
                return r.status_code, None

        # handle all exceptions
        except:
            print("EXCEPTION")
            # just return form function
            return 404, None

    def get_schema(self):
        status, schema = self.get('schema')
        print(schema)

        redone = {}

        for s in schema:
            if s['table_name'] not in redone:
                redone[s['table_name']] = []
            if s['column_name'] not in redone[s['table_name']]:
                redone[s['table_name']].append(s['column_name'])

        self.write_json_file('schema.json', redone)

    def extract_base_features(self, data):
        '''
        Extracts keys (features) from a list of dicts 
        Returns a set of features that all dicts contain, 
            as well as a set of all the extra keys 
        (extra keys are supplied to show how messed up opendota's api is, 
            only features in the base_feature_set should be used)
        '''
        if isinstance(data, list):
            base_feature_set = set(data[0].keys())
            extras = set()
            for d in data:
                f = set(d.keys())
                extras = extras.union(base_feature_set.symmetric_difference(f))
                base_feature_set = base_feature_set.intersection(f)

            return base_feature_set, extras
        elif isinstance(data, dict):
            return data.keys(), set()

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
            json.dump(data, f)

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

    def _chunk_match_ids(self):
        amount = 1000

        matches = dota_data.read_json_file("./Data/Matches_By_Id/40000_plus_matches.json")
        matches = [match['match_id'] for match in matches]
        iterations = len(matches) / amount
        remainder = len(matches) % amount

        base_filepath = "./Data/Matches_By_Id/chunked/"

        r = matches[:remainder]
        dota_data.write_json_file("{}{}".format(base_filepath, 'remainder.json'), r)
        for i in range(iterations):
            match_subset = matches[remainder + (i * amount):remainder + ((i + 1) * amount)]
            dota_data.write_json_file("{}{}".format(base_filepath, '{}.json'.format(str(i + 1))), match_subset)

    def _chunck_matches(self, filename):
        dota_data = DotaData()

        match_ids = dota_data.read_json_file("./Data/Matches_By_Id/chunked/{}.json".format(filename))
        matches = []
        for mid in match_ids:
            status, data = dota_data.get("matches/{}".format(mid))

            if(status == 200):
                print(str(mid))
                matches.append(data)
            else:
                print('bad status')
            sleep(1.1)  # the opendota api requests that this endpoint only be hit 1/s
        matches = dota_data.shorten_data(matches, {'players': ['isRadiant', 'hero_id'], 'radiant_win': None})
        dota_data.write_json_file("./Data/Matches/chunked/{}.json".format(filename), matches)

        try:
            import time
            from pygame import mixer
            mixer.init()
            alert=mixer.Sound('boom.wav')
            alert.play()
            time.sleep(1)
            alert.play()
            time.sleep(1)
            alert.play()
            time.sleep(1)
        except ImportError:
            pass



class BasicHeroData(DotaData):
    '''
    Basic Usage: Get dota data however you can, be it loaded from a file or directly from the api
        then call load_data on it
    '''

    def __init__(self):
        super(BasicHeroData, self).__init__()
        self.hero_features, self.hero_id_index_map = self.heroes()
        self.target_labels = ['radiant_win']

    def heroes(self):
        # heroes = self.read_json_file('./Data/heroes.json')['heroes']
        heroes = self.read_json_file('./Data/heroesV3.json')['heroes']
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
        self.raw_targets = targets
        self.data = self.np_ize(data, True)
        self.targets = self.np_ize(targets, True)

    def load_hero_data(self):
        r = requests.get("{}".format('https://api.opendota.com/api/heroStats'))
        shortened_heroe_data = self.shorten_data(r.json(), { 'id': None, "name": None, "localized_name": None, })
        self.write_json_file('./Data/heroesV3.json', shortened_heroe_data)


def gatherdata(write_path, read_path):
    h = BasicHeroData()

    matches_by_id = h.read_json_file(read_path)
    # matches_by_id = h.read_json_file('./Data/Matches_By_Id/10_matches.json')
    # features, _data = h.np_ize(matches_by_id)
    matches = []
    match_ids = [datum['match_id'] for datum in matches_by_id]

    t = 0

    for mid in match_ids:
        # for mid in match_ids:
        print('Getting: ' + str(mid) + ": " + str(t))
        t = t + 1

        status, chunk = h.get("matches/{}".format(mid))

        if(status == 200):
            print('appending')
            matches.append(chunk)
        else:
            print('bad status')

        # matches.append(h.get("matches/{}".format(mid)))
        sleep(1.1)  # the opendota api requests that this endpoint only be hit 1/s
    # h.write_json_file('./Data/Matches/500_matches_full.json', matches)

    h.load_data(matches)

    features = h.hero_features
    data = h.data
    target_labels = h.target_labels
    targets = h.targets

    # assert len(data[0]) == len(features)
    # assert len(targets[0]) == len(target_labels)

    h.write_json_file(write_path, h.shortened_data)

def gather_chunked_data(write_path, read_path_partial, max, outset = None):
    h = BasicHeroData()
    match_arry = []
    for i in range(1, max):
        print("Getting: " + read_path_partial +'{}.json'.format(i))
        match_arry = match_arry + h.read_json_file(read_path_partial +'{}.json'.format(i))
    # print("Getting: " + (read_path_partial + outset))
    # match_arry = match_arry + json.loads(h.read_json_file((read_path_partial + outset)))
    h.load_data(match_arry)
    h.write_json_file(write_path, h.shortened_data)

def _save_hero_data():
    hero_data = BasicHeroData()
    matches = hero_data.read_json_file('./Data/Matches/40k_matches_short.json')
    hero_data.load_data(matches)
    data = {
        'raw_data': hero_data.raw_data,
        'raw_targets': hero_data.raw_targets,
        'features': hero_data.hero_features,
        'target_labels': hero_data.target_labels
    }
    hero_data.write_json_file('./Data/hero_data/full_40000_plus_data.json', data)



def plot_summed(summed_features):
    _sorted = np.sort(summed_features, axis=-1, kind='mergesort', order=None)
    y_pos = np.arange(len(_sorted))

    plt.figure(figsize=(20, 3))  # width:20, height:3
    # plt.bar(range(len(my_dict)), my_dict.values(), align='edge', width=0.3)

    plt.bar(range(len(summed_features)), summed_features,  align='center', alpha=0.5, width=0.3)

    plt.xticks(range(0, len(summed_features), 10))

    plt.ylabel('Usage')
    plt.title('Dota 2 hero usages in 40k matches')

    plt.show()

def load_saved_hero_data(filepath):
    hero_data = BasicHeroData().read_json_file(filepath)
    hero_data['data'] = np.array(hero_data['raw_data'])
    hero_data['targets'] = np.array(hero_data['raw_targets'])
    del hero_data['raw_targets']
    del hero_data['raw_data']
    return hero_data['data'], hero_data['targets'], hero_data['features'], hero_data['target_labels']

def assess_hero_data(data):
    # print formatting settings
    np.set_printoptions(threshold=np.nan, linewidth=453)
    # http://stackoverflow.com/questions/16468717/iterating-over-numpy-matrix-rows-to-apply-a-function-each
    def sum(x):
        return np.sum(x)
    # find difference between the team 1 and team 2s usage of the player
    def sub(x):
        return abs(x[:113] - x[113:])
    # total number of times (between both teams) a hero is used
    def add(x):
        return np.add(x[:113], x[113:])


    # get sum of each column (find out how much each hero is used)
    feature_details = np.apply_along_axis(sum, axis=0, arr=data)

    summed_features = np.apply_along_axis(add, axis=0, arr=feature_details)

    percentages = np.divide(summed_features.astype('float32'), sum(summed_features))

    '''for i, x in enumerate(percentages):
                    if x < .002:
                        print i'''

    return feature_details, summed_features, percentages


def drop_features(data, targets, features, percentages, threshold):

    column_drop_indexes = []
    for i, x in enumerate(percentages):
        if x < threshold:
            column_drop_indexes.append(i)
            column_drop_indexes.append(i + len(features) / 2)

    print column_drop_indexes

    row_drop_indexes = []
    for index, d in enumerate(data):
        if any([d[i] == 1 for i in column_drop_indexes]):
            row_drop_indexes.append(index)

    data = np.delete(data, row_drop_indexes, 0)
    targets = np.delete(targets, row_drop_indexes, 0)
    features = np.delete(features, column_drop_indexes)

    return data, targets, features

def _write_json_file(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f)

def _save_data_dropped_features(threshold, name):

    data, targets, features, target_labels = load_saved_hero_data('./Data/hero_data/full_40000_plus_data.json')
    print len(data)

    feature_details, summed_features, percentages = assess_hero_data(data)
    #plot_summed(summed_features)
    data, targets, features = drop_features(data, targets, features, percentages, threshold)
    print len(data)
    d = {
        'data': data.tolist(),
        'targets': targets.tolist(),
        'features': features.tolist(),
        'target_labels': target_labels
    }
    _write_json_file('./Data/hero_data/{}'.format(name), d)





if __name__ == "__main__":

    _save_data_dropped_features(.004, 'threshold_004.json')











