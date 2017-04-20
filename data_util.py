import requests
import numpy as np

class DotaData:
	def __init__(self):
		self.base_api = "https://api.opendota.com/api/"

	def get(self, api):
		 r = requests.get("{}{}".format(self.base_api, api))
		 assert r.status_code == 200
		 return r

	def np_ize(self, data, existing_data=None):
		if existing_data:
			if isinstance(data, dict):
				return 0
		else:
			if isinstance(data, list):
				l = [[v for k,v in sorted(d.items())] for d in data]
				features = data[0].keys()
				if all([d.keys() == features for d in data]):
					return features, np.array(l)
				return -1
		return -1


if __name__ == "__main__":
	'''example usage'''
	d = DotaData()
	data = d.get('proMatches').json()#list data
	#for datum in data:
		#match_data = d.get('matches/{}'.format(datum['match_id'])).json()#dict data
	features, _data = d.np_ize(data)
	print features 
	print _data


