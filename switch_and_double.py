import numpy as np


def make_dummy_input_array(features, num_samples):

    X = np.empty((0, features))
    for i in range(0,num_samples):

        arr = np.zeros(shape=(features, 1))

        for i in range(features):
            arr[i] = i

        arr = arr.T
        X = np.append(X, arr, axis=0)

    return X


def switch(original_arr_row):
    original_arr_feature_size = original_arr_row.shape[0]
    if((original_arr_feature_size % 2) == 0):
        team_2_starting_index = original_arr_feature_size // 2
        #print(team_2_starting_index)
        a = original_arr_row
        b = np.empty(original_arr_feature_size)

        for i in range(0,team_2_starting_index):
            b[i] = a[team_2_starting_index + i]

        for i in range(0,team_2_starting_index):
            b[team_2_starting_index + i] = a[i]

        return b

def double(original_arr):
    original_arr_sample_size = original_arr.shape[0]
    original_arr_feature_size = original_arr.shape[1]
    #print(original_arr_sample_size)
    a2 = original_arr

    X = np.empty((0, original_arr_feature_size))

    #print(X.shape)
    for row in range(0,original_arr_sample_size):
        arr = np.zeros(shape=(original_arr_feature_size, 1))
        for j in range(0, original_arr_feature_size):
            arr[j] = a2[row][j]
        arr = arr.T
        X = np.append(X, arr, axis=0)


        arr = np.zeros(shape=(original_arr_feature_size, 1))
        switched = switch(a2[row])
        for j in range(0, original_arr_feature_size):
            arr[j] = switched[j]
        arr = arr.T
        X = np.append(X, arr, axis=0)

    return X


features = 10
num_samples = 5

a = make_dummy_input_array(features, num_samples)

print(a)
print()

print(double(a))

print(double(a).shape)