import requests
import numpy as np
from time import sleep
import json

class NumpyException(Exception):
	pass

class SubDictException(Exception):
	pass

class DotaData:

	def __init__(self):
		self.base_api = "https://api.opendota.com/api/"

	def get(self, api):
		 r = requests.get("{}{}".format(self.base_api, api))
		 assert r.status_code == 200
		 return r.json()

	def extract_base_features(self, data):
		base_feature_set = set(data[0].keys())
		extras = set()
		for d in data:
			f = set(d.keys())
			extras = extras.union(base_feature_set.symmetric_difference(f))
			base_feature_set = base_feature_set.intersection(f)
				
		return base_feature_set, extras	

	def np_ize(self, data):
		if isinstance(data, list):
			features = data[0].keys()
			if all([d.keys() == features for d in data]):
				l = [[v for k,v in sorted(d.items())] for d in data]
				return features, np.array(l)
			else:
				base_feature_set, extra_features = self.extract_base_features(data)
				l = [[v for k,v in sorted(d.items()) if k in base_feature_set] for d in data]
				return base_feature_set, np.array(l)
		raise NumpyException("Unable to transform data into numpy array")

	def sub_dicts(self, data, desired_keys):
		base_feature_set, extras = self.extract_base_features(data)
		if all(k in base_feature_set for k in desired_keys):
			return [{k:d[k] for k in desired_keys} for d in data]
		raise SubDictException("Unable to extract sub dict. Not all keys exist in every member of the data set.")

	def read_json_file(self, filepath):
		with open(filepath, 'r') as f:
			return json.load(f)


if __name__ == "__main__":
	'''example usage'''
	d = DotaData()
	match_ids = d.read_json_file('./Data/Matches_By_Id/200_matches.json')
	match_ids = [item.values()[0] for item in match_ids]
	matches = []
	for mid in match_ids[:2]:#only getting 2 here so you can see what I'm doing
		matches.append(d.get('/matches/{}'.format(mid)))
	print json.dumps(matches[0], indent=4)
	smaller_dict = d.sub_dicts(matches, ['match_id', 'radiant_win'])
	print smaller_dict
	features, data = d.np_ize(smaller_dict)
	print features
	print data

	


	





	


