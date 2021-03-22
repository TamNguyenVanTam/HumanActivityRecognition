# encoding: utf-8
#Authors: TamNV
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from augmented_data import DA_Scaling, DA_Rotation, DA_Permutation

def normalize_data(x_train, x_valid, x_test):
	num_trains, num_valids, num_tests = x_train.shape[0], x_valid.shape[0], x_test.shape[0] 
	num_time_steps, num_feas = x_train.shape[1], x_train.shape[2]
	scaler = MinMaxScaler(feature_range=(-1, 1))

	x_train = np.reshape(x_train, (num_trains * num_time_steps, num_feas))
	x_valid = np.reshape(x_valid, (num_valids * num_time_steps, num_feas))
	x_test = np.reshape(x_test, (num_tests * num_time_steps, num_feas))

	x_train = scaler.fit_transform(x_train)
	x_valid = scaler.transform(x_valid)
	x_test = scaler.transform(x_test)

	x_train = np.reshape(x_train, (num_trains, num_time_steps, num_feas))
	x_valid = np.reshape(x_valid, (num_valids, num_time_steps, num_feas))
	x_test = np.reshape(x_test, (num_tests, num_time_steps, num_feas))

	return x_train, x_valid, x_test

def convert2onehot(y, num_classes):
	"""
	Convert y to onehot form

	params:
		y: numpy array 
			labels
		num_classes: Integer
			number of classes
	returns:
		One-hoted label 
	"""
	y = np.reshape(y, -1)
	num_sams = y.shape[0]
	_y = np.zeros((num_sams, num_classes), np.int32)
	
	for i in range(num_sams):
		_y[i, y[i]] = 1
	
	return _y


def get_augmented_data(X, Y):
	num_sams = X.shape[0]

	aug_X, aug_Y = [], []

	for i in range(num_sams):
		flag = (np.random.rand() <0.2)
		if flag:
			scaled = DA_Scaling(X[i])
			aug_X.append(scaled)
			aug_Y.append(Y[i])

		flag = (np.random.rand() <0.2)
		if flag:
			rotated = DA_Rotation(X[i])
			aug_X.append(rotated)
			aug_Y.append(Y[i])

		# flag = (np.random.rand() <0.2)
		# if flag:
		# 	permuted = DA_Permutation(X[i])
		# 	aug_X.append(permuted)
		# 	aug_Y.append(Y[i])

	aug_X = np.array(aug_X)
	aug_Y = np.array(aug_Y)
	return aug_X, aug_Y



def get_smartphone_dataset(dataname="SmartPhoneDataset"):
	"""
	Read pickle file

	Params:
		None
	Returns:
		x_train, y_train, x_test, y_test
	"""
	file = open("../data/{}".format(dataname), "rb")
	(x_train, y_train), (x_test, y_test) = pickle.load(file, encoding="bytes")
	file.close()
	x_train = x_train[:, :, 0:6]
	x_test = x_test[:, :, 0:6]
	
	#seperate train, valid
	np.random.seed(21) #21
	idxs = np.arange(x_train.shape[0])
	np.random.shuffle(idxs)

	num_valid_sams = int(0.3 * x_train.shape[0])
	x_valid = x_train[idxs[0:num_valid_sams]]
	y_valid = y_train[idxs[0:num_valid_sams]]

	x_train = x_train[idxs[num_valid_sams : ]]
	y_train = y_train[idxs[num_valid_sams : ]]



	x_train, x_valid, x_test = normalize_data(x_train, x_valid, x_test)
	num_classes = np.unique(y_train).shape[0]

	y_train = convert2onehot(y_train, num_classes)
	y_valid = convert2onehot(y_valid, num_classes)
	y_test = convert2onehot(y_test, num_classes)


	print("Number Training {}".format(x_train.shape[0]))
	print("Number Validation {}".format(x_valid.shape[0]))
	print("Number Testing {}".format(x_test.shape[0]))

	return x_train, y_train, x_valid, y_valid, x_test, y_test

def get_class_distribution(Y, num_classes):
	freqs = np.zeros(num_classes)

	for c in Y:
		freqs[c] += 1

	freqs = freqs*100.0 / Y.shape[0]
	return freqs

LABELS = [
	"WALKING",
	"WALKING_UPSTAIRS",
	"WALKING_DOWNSTAIRS",
	"SITTING",
	"STANDING",
	"LAYING"
]

def get_class_proportion(labels):
	num_samples = labels.shape[0]
	for idx, name in enumerate(LABELS):
		num = np.sum(labels == idx)
		print("Class {}: Number Samples {}, Class Proportion {:.2f}"
			.format(name, num, num*1.0 / num_samples))

if __name__ == "__main__":
	x_train, y_train, x_valid, y_valid, x_test, y_test = get_smartphone_dataset()
	y_train = np.argmax(y_train, axis=1)

	y_valid = np.argmax(y_valid, axis=1)
	
	print("Training Set!")
	get_class_proportion(y_train.reshape(-1))
	print("Validation Set!")
	get_class_proportion(y_valid.reshape(-1))