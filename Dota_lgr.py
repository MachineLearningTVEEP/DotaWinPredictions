# The modules we're going to use
from __future__ import print_function

from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_util import double_inverse_samples
from model_output import ModelOutput

# display intercept
def p_intercept(intercept):
	print("The intercept is {}".format(intercept))
# display coefficients
def p_coefficients(coefficients):
	print("The coefficients are {}".format(", ".join(str(x) for x in coefficients.flatten().tolist())))

class LgrModel(ModelOutput):
	def run_model(self, data, targets, batch_size, epochs):
		#get the data from function
		data = double_inverse_samples(data)
		targets = double_inverse_samples(targets)
		# split the data up into multiple sets: training, testing validation
		train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.3, random_state=42)
		# set up logistic regression object
		lgr = linear_model.LogisticRegression()
		# http://stackoverflow.com/questions/34337093/why-am-i-getting-a-data-conversion-warning
		# fixed data to avoid compiler warning
		n = train_target.shape[0]
		y = train_target.reshape((n,))
		# print data shapes
		print(train_data.shape)
		print(y.shape)
		# fit data to model
		lgr.fit(train_data, y)
		# make prediciton on test and training data
		test_predict_1 = lgr.predict(test_data)
		train_predict_1 = lgr.predict(train_data)
		print()
		print()
		# get acc on train and test data
		train_score = str(lgr.score(train_data, train_target))
		train_accuracy = str(accuracy_score(train_target, train_predict_1))
		test_score = str(lgr.score(test_data, test_target))
		test_accuracy = str(accuracy_score(test_target, test_predict_1))
		#print acc
		print("Mean Accuracy (Training Data (Data / True Target) /  sklearn.linear_model.LogisticRegression.Score): " + str(lgr.score(train_data, train_target)))
		print()
		print("Accuracy (Training Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " +  str(accuracy_score(train_target, train_predict_1)))
		print()
		print("Mean Accuracy (Testing Data (Data / True Target) /  sklearn.linear_model.LogisticRegression.Score): " + str(lgr.score(test_data, test_target)))
		print()
		print("Accuracy (Testing Data (Data / Predicted Target) / sklearn.metrics.accuracy_score): " +  str(accuracy_score(test_target, test_predict_1)))
		# set metrics object
		metrics = {
			'train_score': train_score,
			'train_accuracy': train_accuracy,
			'test_score': test_score,
			'test_accuracy': test_accuracy
		}
		# return object and metrics
		return metrics, lgr

if __name__ == '__main__':
	print()
	# fun machine learning on dataset
    # LgrModel('./Data/hero_data/full_40000_plus_data.json', 'lgr', 'lgr', None, None)