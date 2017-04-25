import requests
import numpy as np
from time import sleep

class NumpyException(Exception):
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


if __name__ == "__main__":
	'''example usage'''
	d = DotaData()
	data = d.get('publicMatches')
	features, _data = d.np_ize(data)

	matches = []
	match_ids = [datum['match_id'] for datum in data]

	for mid in match_ids[:2]:# just doing 2 in this example so it doesn't take too long
		matches.append(d.get("matches/{}".format(mid)))
		sleep(1) # the opendota api requests that this endpoint only be hit 1/s

	features, _data = d.np_ize(matches)




	


