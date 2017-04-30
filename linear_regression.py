from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

from data_util import BasicHeroData

def p_intercept(intercept):
	print("The intercept is {}".format(intercept))

def p_coefficients(coefficients):
	print("The coefficients are {}".format(", ".join(str(x) for x in coefficients.flatten().tolist())))

def mse(target, predictions):
	'''mean squared error according to slides (ie, div by 2N)'''
	return ((target - predictions)**2).mean() / 2

def p_errors(test_target, train_target, test_predictions, train_predictions):
	print("The mean squared error of the training dataset is {}".format(mse(train_target, train_predictions)))
	print("The mean squared error of the testing dataset is {}".format(mse(test_target, test_predictions)))

def plot_fig(target, predictions):
	plt.scatter(target, predictions)
	plt.xlabel("Real Prices")
	plt.ylabel("Predicted Prices")
	plt.show() 

def based_on_homework(filepath):
	h = BasicHeroData()
	matches = h.read_json_file(filepath)
	h.load_data(matches)

	data = h.data
	targets = h.targets

	train_data,test_data,train_target,test_target = train_test_split(data,targets, test_size=0.2, random_state=42)

	r = linear_model.LinearRegression()
	r.fit(train_data, train_target)

	p_intercept(r.intercept_[0])
	p_coefficients(r.coef_)
          
	test_predict_1 = r.predict(test_data)
	train_predict_1 = r.predict(train_data)
	p_errors(test_target, train_target, test_predict_1, train_predict_1)

	plot_fig(test_target, test_predict_1)

if __name__ == '__main__':
	based_on_homework('./Data/Matches/5_matches.json')
