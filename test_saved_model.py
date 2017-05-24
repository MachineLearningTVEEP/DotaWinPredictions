from keras.models import load_model
from data_util import BasicHeroData
from keras.utils import np_utils

model = load_model('./models/cnn_1.h5')
h = BasicHeroData()
#match_ids = h.read_json_file('./Data/model_data/new_match_ids_1.json')
#match_ids = [m['match_id'] for m in match_ids]
#matches = h._get_match(match_ids)
#h.write_json_file('./Data/model_data/new_matches_1.json', matches)
#h.load_data(matches)
#h.write_json_file('./Data/model_data/new_data_1.json', h._data())
test_data, test_targets, features, labels = h.load_saved_hero_data('./Data/model_data/new_data_1.json')
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
test_targets = np_utils.to_categorical(test_targets, 2)


loss, accuracy = model.evaluate(test_data, test_targets)

print "Loss: {}".format(loss)
print "Accuracy: {}".format(accuracy)