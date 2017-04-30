import requests
import numpy as np
from time import sleep
import json


class NumpyException(Exception):
	pass


class DotaData:
	def __init__(self):
		self.base_api = "https://api.opendota.com/api/"

	# load match ids from a chosen json file
	def loadMatches(self, file_path):
		# Load in json file of matches
		with open(file_path) as data_file:
			# save json to memory
			return json.load(data_file)

	# get json from api with parameter
	def get(self, api):
		# get request from api
		# https://pyformat.info/
		r = requests.get("{}{}".format(self.base_api, api))
		# make sure request has good status code
		assert r.status_code == 200
		# return json of request object
		return r.json()


	def extract_base_features(self, data):
		# get keys of the first data element as the form of a set
		base_feature_set = set(data[0].keys())
		# create an empty set for the extra features that do not belong to all the data (matches)
		extras = set()
		# loop all data
		for d in data:
			# create temp set using the keys from current match
			f = set(d.keys())
			# symmetric_difference: new set with elements in either s or t but not both
			# https://docs.python.org/2/library/sets.html
			# basically, get the features that do not belong to the base feature set
			# and add them to the current set of other features that do not belong to the base feature set
			extras = extras.union(base_feature_set.symmetric_difference(f))
			# intersection: new set with elements common to s and t
			# set base features of what features bwlong to both the base feature set and the temp f set
			base_feature_set = base_feature_set.intersection(f)
		# return the common features and the features that are unique
		return base_feature_set, extras

	def np_ize(self, data):
		# check if data is a list
		if isinstance(data, list):
			# use any element to get the keys (all elements may or may not have same features)
			features = data[0].keys()
			# Return True if all elements of the iterable are true (or if the iterable is empty).
			# https://docs.python.org/2/library/functions.html#all
			# checks if all keys of the element are equal for all the other data elements
			if all([d.keys() == features for d in data]):
				# sort items: sorted(d.items()) for each d in data
				# for k, v in sorted(d.items()): get key and vlaue for each sorted d
				# get v as the value to put into list
				# we use k, v so seperate the key ans vlaue or else it would display both
				# now we have list of features (values)
				l = [[v for k, v in sorted(d.items())] for d in data]
				# return the feautres (keys) and a np array (vector) of the list of features (values)
				return features, np.array(l)
			# there are features that do no belong to all api calls (all data / all matches)
			else:
				# get the common features and unique features
				base_feature_set, extra_features = self.extract_base_features(data)
				# loop the data : d
				# get all values in d
				# if that value belongs to the base_feature_set
				# add to list
				l = [[v for k, v in sorted(d.items()) if k in base_feature_set] for d in data]
				# return the common features and a np vector of the data of that feature set
				return base_feature_set, np.array(l)
		raise NumpyException("Unable to transform data into numpy array")


if __name__ == "__main__":
	'''example usage'''
	d = DotaData()
	# data = d.get('publicMatches')
	data = d.loadMatches('./Data/Matches_By_Id/10_matches.json')
	features, _data = d.np_ize(data)
	matches = []
	match_ids = [datum['match_id'] for datum in data]
	for mid in match_ids[:5]:# just doing 2 in this example so it doesn't take too long
	# for mid in match_ids:
		matches.append(d.get("matches/{}".format(mid)))
		sleep(1) # the opendota api requests that this endpoint only be hit 1/s


	print(matches)


	#
	# features, _data = d.np_ize(matches)
    #
    # print(features)
	print()
	print()
	print()
	print()
	print()
	# print(_data)






