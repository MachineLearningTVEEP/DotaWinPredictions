import numpy as np
import json
from pprint import pprint
import requests

class NumpyException(Exception):
	pass

class MatchData:
	def __init__(self):
		self.dire_score = None
		self.radiant_score = None
		self.radiant_win = None
		self.picks = []

class MatchBanPicks:
	def __init__(self, is_pick, hero_id, team):
		self.is_pick = is_pick
		self.hero_id = hero_id
		self.team = team

class DotaData:

	def __init__(self):
		self.base_api = "https://api.opendota.com/api/matches/"
		self.data = None
		self.matches = []

	def loadMatchIds(self, file):
		# Load in json file of matches
		with open(file) as data_file:
			# save json to memory
			self.data = json.load(data_file)

	def getMatchIds(self):
		# return json file
		return self.data


	def loadMatchData(self):
		# get the match ids from json into a single array
		# instead of doing it in the same for loop, I thought this may be easier for now
		match_ids = [datum['match_id'] for datum in self.data]
		for match in match_ids:
			r = requests.get("{}{}".format(self.base_api, str(match)))
			assert r.status_code == 200
			# print(r.json())

			match = MatchData()

			match.dire_score = r.json()['dire_score']
			match.radiant_score = r.json()['radiant_score']
			match.radiant_win = r.json()['radiant_win']
			picks = r.json()['picks_bans']

			for pick in picks:
				picked = MatchBanPicks(pick['is_pick'], pick['hero_id'], pick['team'])
				match.picks.append(picked)

			print(match.picks)

			# print(r.json()['dire_score'])









			#
			# def extract_base_features(self, data):
			# 	base_feature_set = set(data[0].keys())
			# 	extras = set()
			#
			# 	for d in data:
			# 		f = set(d.keys())
			# 		extras = extras.union(base_feature_set.symmetric_difference(f))
			# 		base_feature_set = base_feature_set.intersection(f)
			#
			# 	return base_feature_set, extras
			#
			# def np_ize(self, data):
			# 	if isinstance(data, list):
			# 		features = data[0].keys()
			# 		if all([d.keys() == features for d in data]):
			# 			l = [[v for k,v in sorted(d.items())] for d in data]
			# 			return features, np.array(l)
			# 		else:
			# 			base_feature_set, extra_features = self.extract_base_features(data)
			# 			l = [[v for k,v in sorted(d.items()) if k in base_feature_set] for d in data]
			# 			return base_feature_set, np.array(l)
			# 	raise NumpyException("Unable to transform data into numpy array")


if __name__ == "__main__":
	'''example usage'''
	d = DotaData()
	data = d.loadMatchIds('./Data/Matches_By_Id/200_matches.json')

	d.loadMatchData()


	# features, _data = d.np_ize(data)
    #
	# matches = []
	# match_ids = [datum['match_id'] for datum in data]
    #
	# for mid in match_ids[:2]:# just doing 2 in this example so it doesn't take too long
	# 	matches.append(d.get("matches/{}".format(mid)))
	# 	sleep(1) # the opendota api requests that this endpoint only be hit 1/s
    #
	# features, _data = d.np_ize(matches)




	


