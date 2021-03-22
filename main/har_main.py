"""
Authors: TamNV
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.insert(0, "../model")
sys.path.insert(0, "../base")

from vgg import VGG8
from vgg_center_loss import VGG8CenterLoss
from cnn_lstm import CNNLSTM

from utils import *
from read_data import get_smartphone_dataset

from har_trainer import HARTrainer
from har_predictor import HARPredictor

from abstract_data import DataManager

def read_json_file(filename):
	with open(filename) as file:
		contents = json.load(file)
	if contents == None:
		print("Meeting wrong went read {}".format(filename))
		exit(0)
	return contents

def do_training(model, data, sess, exe_config):
	"""
	Perform training

	Params:
		model: Model instance
		data: DataManager instance
		sess: tf.Session()
		exe_config: Dictionary
	Returns:
		None
	"""
	sess.run(tf.global_variables_initializer())
	trainer = HARTrainer(exe_config)
	trainer.train(model, data, sess)

def do_predict(model, data, sess, exe_config):
	"""
		Perform the predicting process

		Params:
			model: Substance of Model
			data
	"""
	predictor = HARPredictor(exe_config)
	predictor.predict(model, data, sess)

import argparse
parser = argparse.ArgumentParser(description="arguments of traffic jam project")
parser.add_argument("--model_config", dest="model_config", default="configuration/model_config.json")
parser.add_argument("--exe_config", dest="exe_config", default="configuration/exe_config.json")
parser.add_argument("--phase", dest="phase", default="predict")
args = parser.parse_args()

if __name__ == "__main__":
	model_config = read_json_file(args.model_config)
	exe_config = read_json_file(args.exe_config)

	model_config.update({KEY_MODEL_FACTORY:VGG8CenterLoss})

	x_train, y_train, x_valid, y_valid, x_test, y_test = get_smartphone_dataset(exe_config["dataname"])
	
	placeholders = {
		"features": tf.placeholder(tf.float32,
						shape=(
								None,
								model_config[KEY_INPUT_SIZE],
								model_config[KEY_NUM_INPUT_CHANNELS]
						)),
		"labels": tf.placeholder(tf.float32,
						shape=(
								None,
								model_config[KEY_NUM_CLASSES]
						)),
		"dropout": tf.placeholder(tf.float32),
		"learning_rate": tf.placeholder(tf.float32),
		"weight_decay": tf.placeholder(tf.float32)
	}

	model = model_config[KEY_MODEL_FACTORY](placeholders)
	
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
	data = DataManager(x_train, y_train, x_valid, y_valid, x_test, y_test)
	
	if args.phase == "train":
		do_training(model, data, sess, exe_config)
	
	elif args.phase == "predict":
		do_predict(model, data, sess, exe_config)

	else:
		print("{} option doesn't support".format(args.phase))
		exit(0)
