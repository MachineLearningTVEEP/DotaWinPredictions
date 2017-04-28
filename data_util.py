import numpy as np
import json
from pprint import pprint
import requests
import jsonpickle

class NumpyException(Exception):
	pass

# simple wrapper class to make it easier to compute a json to save to file
# class MatchDataBase:
# 	def __init__(self):
# 		self.matches_complete = None

	# convert match data to json
	# def toJSON(self):
	# 	# http://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable
	# 	return json.dumps(self, default=lambda o: o.__dict__,sort_keys=False, indent=2)

# class to old the match data that is needed (excluding what we don't need)
class MatchData:
	def __init__(self):
		# current needed data
		self.dire_score = None
		self.radiant_score = None
		self.radiant_win = None
		self.picks = []

# class to hold the pick_ban data
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

	# load match ids from a chosen json file
	def loadMatchIds(self, file):
		# Load in json file of matches
		with open(file) as data_file:
			# save json to memory
			self.data = json.load(data_file)

	# return the json of match ids
	def getMatchIds(self):
		# return json file
		return self.data

	# using the json of match ids, call the opendota api to get the json for each match using each id
	def loadMatchData(self):
		# get the match ids from json into a single array
		# instead of doing it in the same for loop, I thought this may be easier for now
		match_ids = [datum['match_id'] for datum in self.data]
		# loop all match ids
		for match in match_ids:
			# call api to get json for each match containing all match info
			r = requests.get("{}{}".format(self.base_api, str(match)))
			# check for return status
			assert r.status_code == 200
			# create temp object to hold current match
			current_match = MatchData()
			# set match object's data that is from the json pulled for this current match id
			current_match.dire_score = r.json()['dire_score']
			current_match.radiant_score = r.json()['radiant_score']
			current_match.radiant_win = r.json()['radiant_win']
			picks = r.json()['picks_bans']
			# the match json contains a array of picks and bans
			# loop this array and set an array of the MatchBanPick object which the needed data from that
			for pick in picks:
				# set new object with the 3 parameters we need
				picked = MatchBanPicks(pick['is_pick'], pick['hero_id'], pick['team'])
				# append to array
				current_match.picks.append(picked)
			# append the curent match with all the needed data to a new array
			self.matches.append(current_match)

	def matchesToJson(self):
		# create base class (since we cna turn a class into json, we needed a wrapper class to wrap the array of all the matches
		# to parse into json. this makes it a lot easier for now)
		# base = MatchDataBase()
		# set class array to array of matches
		# base.matches_complete = self.matches
		match_json = jsonpickle.encode(self.matches, unpicklable=False)

		# output_file = open("/Data/Matches/match.json", "w")
		output_file = open("match.json", "w")
		output_file.write(match_json)
		output_file.close()
		# print out json DEBUG
		# print(base.toJSON())
		# print(json.dumps([ob.__dict__ for ob in self.matches]))

    #
	# with open('data.txt', 'w') as outfile:
	# 	json.dump(data, outfile)


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
	data = d.loadMatchIds('./Data/Matches_By_Id/10_matches.json')
	d.loadMatchData()
	d.matchesToJson()

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




	


