import csv
import uuid
import os
import pickle
import json

from data_util import BasicHeroData
from datetime import datetime
from keras.models import Sequential

class ModelOutput(object):
	def __init__(self, data_filename, model_folder, model_name, batch_size=64, epochs=10):
		self.data_filename = data_filename
		self.model_folder = model_folder
		self.model_name = model_name
		self.hero_data = BasicHeroData()
		data, targets, features, target_labels = self.hero_data.load_saved_hero_data(self.data_filename)
		self.data_shape = data.shape 
		self.targets_shape = targets.shape
		self.metrics, self.model = self.run_model(data, targets, batch_size, epochs)
		self.save_model()

	def run_model(self, data, targets, batch_size=64, epochs=10):
		'''
		uses self.data, self.targets to run model
		Return metrics, model
		metrics should contain a dict of metrics you want to save, 
			as well as any other information you want saved
		if the model is a keras Sequential model, it will be saved in h5 format
		else it will be saved with pickle
		'''
		raise NotImplmentedError

	def save_model(self):
		'''
		saves model in model_folder specified
		updates model_manifest.csv with model attributes, loss, accuracy, date, etc
		'''
		metrics = ['{}: {}'.format(k,v) for k,v in self.metrics.items()]

		directory = './models/{}'.format(self.model_folder)
		if not os.path.isdir(directory):
			os.mkdir(directory)

		if isinstance(self.model, Sequential):
			filename = '{}/{}_{}.h5'.format(directory, self.model_name, self._rand_str())
			self.model.save(filename)
			for i, layer in enumerate(self.model.layers):
				metrics.append('layer_{}: {}'.format(i + 1, json.dumps(layer.get_config())))
		else:
			filename = '{}/{}_{}.sav'.format(directory, self.model_name, self._rand_str())
			pickle.dump(self.model, open(filename, 'wb'))

		

		with open('model_manifest.csv', 'a') as f:
			writer = csv.writer(f)
			writer.writerow([
				str(datetime.utcnow()), 
				self.data_filename, 
				filename,
				self.data_shape, 
				self.targets_shape, 
				]+metrics)

	def _rand_str(self):
		return str(uuid.uuid4()).upper().replace('-', '')