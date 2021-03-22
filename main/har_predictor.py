"""
@Authors: TamNV
Implement traffic jam predictor
"""
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

sys.path.insert(0, "../../base")

from predictor import Predictor
from utils import *

LABELS = [
	"WALKING",
	"W_UPSTAIRS",
	"W_DOWNSTAIRS",
	"SITTING",
	"STANDING",
	"LAYING"
]

def get_recall_each_classes(labels, preds):
	
	for i, name in enumerate(LABELS):
		a = np.sum(np.logical_and(labels==i, preds==i))
		b = np.sum(labels==i)
		rec = a * 1.0 / b
		print("{} : recall {:.3f}".format(name, rec))

def plot_confusion_matric(conf):
	fig, ax = plt.subplots()
	im = ax.imshow(conf)

	# We want to show all ticks...
	ax.set_xticks(np.arange(len(LABELS)))
	ax.set_yticks(np.arange(len(LABELS)))
	# ... and label them with the respective list entries
	ax.set_xticklabels(LABELS)
	ax.set_yticklabels(LABELS)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(len(LABELS)):
	    for j in range(len(LABELS)):
	        text = ax.text(j, i, conf[i, j],
	                       ha="center", va="center", color="w")
	# ax.set_title("")
	fig.tight_layout()
	plt.savefig("normal.png")

class HARPredictor(Predictor):
	"""
	modify predict solution
	"""
	def __init__(self, exe_config):
		"""
		Initilize method

		Params:
			exe_config: dictionary
				configuration of predicting process
		Returns:
			none
		"""
		self.exe_config = exe_config

	def _predict(self):
		"""
		overwritting predicting method
		Params:
			None
		Returns:
			None
		"""
		test_log = []
		labels, preds = [], []
		for i in range(self.num_test_iters):
			batch_x, batch_y = self.data.get_next_batch_test(
										self.exe_config[KEY_BATCH_SIZE]
								)

			feed_dict = {
							self.model.inputs: batch_x,
							self.model.labels: batch_y,
							self.model.dropout: 0.0,
						}
			
			[acc, pred] = self.sess.run([self.model.accuracy, self.model.outputs], feed_dict=feed_dict)

			test_log.append([acc])
			labels.append(batch_y)
			preds.append(pred)
		
		test_log = np.array(test_log)

		labels = np.concatenate(labels, axis=0)
		preds = np.concatenate(preds, axis=0)

		
		labels = np.argmax(labels, axis=1)
		preds = np.argmax(preds, axis=1)

		
		acc = accuracy_score(labels, preds)
		f1 = f1_score(labels, preds,average='macro')
		conf= confusion_matrix(labels, preds)
		pres = precision_score(labels, preds, average='macro')
		rec = recall_score(labels, preds, average='macro')

		print("F1 Score {:.3f}".format(f1))
		print("Acc {:.3f}, Precision {:.3f}, Recall {:.3f}".
			format(acc, pres, rec))
		# get_recall_each_classes(labels, preds)
		# print(conf)
		# plot_confusion_matric(conf)