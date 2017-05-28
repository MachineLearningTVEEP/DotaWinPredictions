import requests
import numpy as np
from numpy import dstack

from time import sleep
import json
import os
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
        try:
            r = requests.get("{}{}".format(self.base_api, api))

            if(r.status_code == 200):
                return r.status_code, r.json()
            else:
                return r.status_code, None
        # handle all exceptions
        except:
            # return 404 
            return 404, None

    def get_schema(self):
        '''
        The dota api has a schema endpoint. This method parses the json returned from that into a 
            dict with table names as keys and their columns as list values because the format on 
            the endpoint is incomprehensible. 
        '''
        status, schema = self.get('schema')

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


class BasicHeroData(DotaData):
    '''
    Basic Usage: Get dota data however you can, be it loaded from a file or directly from the api
        then call load_data on it

    load_data takes in a list of dicts in the format supplied by matches/ endpoint, shortens the data if it
        is not already shortened, and transforms it into a numpy friendly format

    load_saved_hero_data loads an already numpy friendly format
    '''

    def __init__(self):
        super(BasicHeroData, self).__init__()
        self.hero_features, self.hero_id_index_map = self.heroes()
        self.target_labels = ['radiant_win']

    def heroes(self):
        '''
        gets hero data from the heroes file (which in turn comes from the heroStats/ endpoint)
        maps the supplied ids to indexes because there are some heroes missing (we don't know why)
        returns the id_index_map and a doubled list of hero features, for both radiant and dire teams
        '''
        heroes = self.read_json_file('./Data/heroes.json')
        id_name_map = {h['id']: h['name'] for h in heroes}
        ids = id_name_map.keys()
        id_index_map = {x: i for i, x in enumerate(ids)}
        hero_features = [''] * (2 * (len(id_index_map)))
        for i, id in enumerate(id_index_map):
            hero_features[i] = '{}_{}'.format(id_name_map[id], 'radiant')
            hero_features[i + len(id_index_map)] = '{}_{}'.format(id_name_map[id], 'dire')
        return hero_features, id_index_map

    def process_matches(self):
        '''
        Uses hero_id_index_map to create a list for each match that has ones
            for a hero pick and zeroes for all other heroes 
        For both radiant and dire teams in the match 
        Returns an array of such matches and the outcome of each match in data, targets
        '''
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
                targets.append([int(match['radiant_win'])])
        return data, targets

    def load_data(self, matches):
        '''
        Accepts input data in the format of the matches/ endpoint, 
            shortens data, processes it and sets the raw_data, raw_targets,
            data, targets class variables
        '''
        self.shortened_data = self.shorten_data(matches, {'players': ['isRadiant', 'hero_id'], 'radiant_win': None})
        data, targets = self.process_matches()
        self.raw_data = data
        self.raw_targets = targets
        self.data = self.np_ize(data, True)
        self.targets = self.np_ize(targets, True)

    def _load_heroes(self):
        '''
        Calls the heroStats/ endpoint and returns a shortened list
        '''
        status, heroes = self.get('heroStats')
        shortened_heroe_data = self.shorten_data(heroes, { 'id': None, "name": None, "localized_name": None,})
        self.write_json_file('./Data/heroes.json', shortened_heroe_data)

    def _chunk_match_ids(self):
        '''
        Seperates match ids into distinct files so that they can be processed 
        We were having issues with the dota matches/ endpoint and large, repetitive queries
            because at one call per second, 48,000+ queries was way too much 
        '''
        amount = 1000

        matches = self.read_json_file("./Data/Matches_By_Id/40000_plus_matches.json")
        matches = [match['match_id'] for match in matches]
        iterations = len(matches) / amount
        remainder = len(matches) % amount

        base_filepath = "./Data/Matches_By_Id/chunked/"

        r = matches[:remainder]
        self.write_json_file("{}{}".format(base_filepath, 'remainder.json'), r)
        for i in range(iterations):
            match_subset = matches[remainder + (i * amount):remainder + ((i + 1) * amount)]
            self.write_json_file("{}{}".format(base_filepath, '{}.json'.format(str(i + 1))), match_subset)

    def _chunk_matches(self, filename):
        '''
        With above, gets a match id file from Data/Matches_By_Id/chunked, calls the match endpoint on those ids, 
            saves the shortened results into the corresponding chunked file in Data/Matches/chunked
        This takes a while, so some explosions let you know when it is done ;) (if you have pygame installed)
        Inputs: string filename (example: '1') 
        '''

        match_ids = self.read_json_file("./Data/Matches_By_Id/chunked/{}.json".format(filename))
        matches = self._get_match(match_ids)
        matches = self.shorten_data(matches, {'players': ['isRadiant', 'hero_id'], 'radiant_win': None})
        self.write_json_file("./Data/Matches/chunked/{}.json".format(filename), matches)

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

    def _get_match(self, match_ids):
        '''
        calls the dota matches endpoint with input match_ids
        '''
        matches = []
        for mid in match_ids:
            status, data = self.get("matches/{}".format(mid))
            print mid
            if(status == 200):
                matches.append(data)
            else:
                print "bad status"
            sleep(1.1)  # the opendota api requests that this endpoint only be hit 1/s
        return matches

    def _gather_chunked_data(self, r_max, outfile='40k_matches_short.json'):
        '''
        Saves all the individual chunks in one file
        '''
        matches = []
        for i in range(1, r_max):
            matches += self.read_json_file('./Data/Matches/chunked/{}.json'.format(i))
        matches += self.read_json_file('./Data/Matches/chunked/remainder.json')

        self.write_json_file('./Data/Matches/{}'.format(outfile), matches)

    def _data(self):
        return {
            'raw_data': self.raw_data,
            'raw_targets': self.raw_targets,
            'features': self.hero_features,
            'target_labels': self.target_labels
        }

    def _save_hero_data(self):
        '''
        saves data in np friendly format to be loaded into ML methods
        initial - 40k plus
        '''
        matches = self.read_json_file('./Data/Matches/40k_matches_short.json')
        self.load_data(matches)
        data = self._data()
        self.write_json_file('./Data/hero_data/full_40000_plus_data.json', data)


    def load_saved_hero_data(self, filepath):
        '''
        loads np friendly version of the data for use in ML methods
        '''
        hero_data = self.read_json_file(filepath)
        hero_data['data'] = np.array(hero_data['raw_data'])
        hero_data['targets'] = np.array(hero_data['raw_targets'])
        del hero_data['raw_targets']
        del hero_data['raw_data']
        return hero_data['data'], hero_data['targets'], hero_data['features'], hero_data['target_labels']

    def _assess_hero_data(self, data):
        '''
        Counts the number of times a hero is used over the dataset
        Calculates the percentages of each hero's use for use in _drop_features
        '''
        def sum(x):
            return np.sum(x)
        # find difference between the team 1 and team 2s usage of the player
        def sub(x):
            return abs(x[:113] - x[113:])
        # total number of times (between both teams) a hero is used
        def add(x):
            return np.add(x[:113], x[113:])

        # get sum of each column (find out how much each hero is used)
        self.feature_details = np.apply_along_axis(sum, axis=0, arr=data)

        self.summed_features = np.apply_along_axis(add, axis=0, arr=self.feature_details)

        self.percentages = np.divide(self.summed_features.astype('float32'), sum(self.summed_features))


    def _plot_summed(self):
        '''
        Plots the summed features from _assess_hero_data
        '''
        _sorted = np.sort(self.summed_features, axis=-1, kind='mergesort', order=None)
        y_pos = np.arange(len(_sorted))

        plt.figure(figsize=(20, 3))  # width:20, height:3

        plt.bar(range(len(self.summed_features)), self.summed_features,  align='center', alpha=0.5, width=0.3)

        plt.xticks(range(0, len(self.summed_features), 10))

        plt.ylabel('Usage')
        plt.title('Dota 2 hero usages in 40k matches')

        plt.show()


    def _drop_features(self, data, targets, features, threshold):
        '''
        drops data and features if they do not pass the percentage threshold 
        uses percentage from _assess_hero_data
        '''
        column_drop_indexes = []
        for i, x in enumerate(self.percentages):
            if x < threshold:
                column_drop_indexes.append(i)
                column_drop_indexes.append(i + len(features) / 2)

        row_drop_indexes = []
        for index, d in enumerate(data):
            if any([d[i] == 1 for i in column_drop_indexes]):
                row_drop_indexes.append(index)

        data = np.delete(data, row_drop_indexes, 0)
        targets = np.delete(targets, row_drop_indexes, 0)
        features = np.delete(features, column_drop_indexes)

        return data, targets, features


    def _save_data_dropped_features(self, threshold, name):
        '''
        Loads saved data, assesses the prevalence of heroes and drops features
            and associated data from those that do not pass the threshold
        '''
        data, targets, features, target_labels = self.load_saved_hero_data('./Data/hero_data/full_40000_plus_data.json')

        self._assess_hero_data(data)
        #self._plot_summed()
        data, targets, features = self._drop_features(data, targets, features, threshold)
        
        d = {
            'raw_data': data.tolist(),
            'raw_targets': targets.tolist(),
            'features': features.tolist(),
            'target_labels': target_labels
        }
        self.write_json_file('./Data/hero_data/{}'.format(name), d)



    def _match_id_dict_to_list(self, read_path, write_path):
        matches = self.read_json_file(read_path)
        self.write_json_file(write_path, sorted([m['match_id'] for m in matches]))


    def get_player_rankings(self, infile, outfile):
        '''
        TODO
        '''
        print "A"
        match_ids = self.read_json_file(infile)

        matches = []

        for mid in match_ids[:1]:
            print "B"
            status, match = self.get('/matches/{}'.format(mid))
            #pp(match)
            M = self.shorten_data([match], {'players': ['account_id', 'hero_id'], 'match_id': None})

            #pp(M)
            _M = []
            for player in M[0]['players']:
                status, _player = self.get('/players/{}'.format(player['account_id']))
                #pp(player)
                sleep(1.1)
                #pp(player)
                p = (self.shorten_data([_player], {'solo_competitive_rank':None, 'competitive_rank': None, 'mmr_estimate':None}))
                #pp(p)
                #p.update(player)
                player.update(p[0])

            #pp(M)
            matches.append(M[0])

        #pp(matches)
        matches = {m['match_id']:m['players'] for m in matches}
        #pp(matches)

        self.write_json_file(outfile, matches)

    def get_solo_player_rankings(self, infile, outfile):

        match_ids = self.read_json_file(infile)

        matches = []

        for mid in match_ids[:2]:
            status, match = self.get('/matches/{}'.format(mid))
            #pp(match)
            M = self.shorten_data([match], {'players': ['account_id', 'hero_id', 'solo_competitive_rank'], 'match_id': None})

            pp(M)
            matches.append(M[0])

        pp(matches)
        matches = {m['match_id']:m['players'] for m in matches}
        self.write_json_file(outfile, matches)

def solo():
    '''
    Tanner, Run this on the 16 gb machine you have; it should take 16 hrs like the last one
    '''
    h = BasicHeroData()
    dir_1 = './Data/Matches_By_Id/chunked/'
    dir_2 = './Data/Matches/solo_chunked/'
    if not os.path.isdir(dir_1):
        os.mkdir(dir_1)
    if not os.path.isdir(dir_2):
        os.mkdir(dir_2)
    for i in range(1,47):
        h.get_player_rankings('{}{}.json'.format(dir_1, str(i)), '{}{}.json'.format(dir_2, str(i)))
    h.get_player_rankings('{}remainder.json'.format(dir_1), '{}remainder.json'.format(dir_2))



def run_on_machine(low, high):
    h = BasicHeroData()
    dir_1 = './Data/Matches_By_Id/chunked/'
    dir_2 = './Data/Matches/chunked_players/'
    if not os.path.isdir(dir_1):
        os.mkdir(dir_1)
    if not os.path.isdir(dir_2):
        os.mkdir(dir_2)
    for i in range(low,high):
        h.get_player_rankings('{}{}.json'.format(dir_1, str(i)), '{}{}.json'.format(dir_2, str(i)))
    h.get_player_rankings('{}remainder.json'.format(dir_1), '{}remainder.json'.format(dir_2))   

#TANNER, run one of these on each of the amazon machines

def machine_1():
    run_on_machine(1, 3)
def machine_2():
    run_on_machine(3, 5)
def machine_3():
    run_on_machine(5, 7)
def machine_4():
    run_on_machine(7, 9)
def machine_5():
    run_on_machine(9, 11)
def machine_6():
    run_on_machine(11, 13)
def machine_7():
    run_on_machine(13, 15)
def machine_8():
    run_on_machine(15, 17)
def machine_9():
    run_on_machine(17, 19)
def machine_10():
    run_on_machine(19, 21)


def make_dummy_input_array(features, num_samples):

    X = np.empty((0, features))
    for i in range(0, num_samples):

        arr = np.zeros(shape=(features, 1))

        for i in range(features):
            # arr[i] = np.random.u
            arr[i] = np.random.random_integers(0, 1)
            # arr[i] = np.random.random_integers(0, 9)

        arr = arr.T
        X = np.append(X, arr, axis=0)

    return X

# def switch(original_arr_row):
#     original_arr_feature_size = original_arr_row.shape[0]
#
#     # if ((original_arr_feature_size % 2) == 0
#
#     team_2_starting_index = original_arr_feature_size // 2
#     # print(team_2_starting_index)
#     a = original_arr_row
#     b = np.empty(original_arr_feature_size)
#
#     for i in range(0, team_2_starting_index):
#         b[i] = a[team_2_starting_index + i]
#
#     for i in range(0, team_2_starting_index)
#         b[team_2_starting_index + i] = a[i]
#
#     return b

# def double( original_arr):
#     original_arr_sample_size = original_arr.shape[0]
#     original_arr_feature_size = original_arr.shape[1]

    # return np.matrix()


    # # print(original_arr_sample_size)
    # a2 = original_arr
    #
    # X = np.empty((0, original_arr_feature_size))
    #
    # # print(X.shape)
    # for row in range(0, original_arr_sample_size):
    #     arr = np.zeros(shape=(original_arr_feature_size, 1))
    #     for j in range(0, original_arr_feature_size):
    #         arr[j] = a2[row][j]
    #     arr = arr.T
    #     X = np.append(X, arr, axis=0)
    #
    #     arr = np.zeros(shape=(original_arr_feature_size, 1))
    #     switched = self.switch(a2[row])
    #     for j in range(0, original_arr_feature_size):
    #         arr[j] = switched[j]
    #     arr = arr.T
    #     X = np.append(X, arr, axis=0)

    # return X

def double_inverse_samples(original_arr):
    doubled_arr = np.zeros((original_arr.shape[0] * 2, original_arr.shape[1]))
    j = 0
    for i in range (0, doubled_arr.shape[0], 2):
        doubled_arr[i] = np.copy(original_arr[j])
        # doubled_arr[i+1] = np.copy(original_arr[j])
        doubled_arr[i+1] = [0 if x == True else 1 for x in original_arr[j]]
        j = j + 1
    return doubled_arr
        # A = np.ones((4, 3))
        # B = np.zeros_like(A)
        #
        # C = np.empty((A.shape[0] + B.shape[0], A.shape[1]))
        #
        # C[::2, :] = A
        # C[1::2, :] = B


if __name__ == '__main__':




