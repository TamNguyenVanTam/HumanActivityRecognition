"""
@Authors: TamNV
this file implements some modified architectures of VGG
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, "../base")

from layers import Dense, Conv1D, Flatten
from layers import LSTM, MaxPooling1D, leak_relu, CenterLoss
from abstract_model import Model
from utils import *
from metrics import *

class CNNLSTM(Model):
	"""
	Implement CNN LSTM Model
	"""
	def __init__(self, placeholders, **kwargs):
		"""
		Initialize method

		Params:
			placeholders: list of placeholders
				which are used for building LSTM model
		Returns:
			None
		"""
		super(CNNLSTM, self).__init__(**kwargs)

		self.inputs = placeholders["features"]
		self.labels = placeholders["labels"]
		self.learning_rate = placeholders["learning_rate"]
		self.dropout = placeholders["dropout"]
		self.decay_factor = placeholders["weight_decay"]

		self.num_classes = self.labels.get_shape().as_list()[1]
		self.num_time_steps = self.inputs.get_shape().as_list()[1]
		self.num_input_channels = self.inputs.get_shape().as_list()[2]

		self.optimizer = tf.train.AdamOptimizer(
							learning_rate=1e-3
						)
		self.build()

	def build(self):
		"""
		Wrapper for _build
		"""
		with tf.variable_scope(self.name):
			self._build()
		# Build sequential layer model
		# CNN Layers
		fea1 = self.layers[0](self.inputs)
		# CNN Layers
		fea2 = self.layers[1](fea1)
		
		tem_feas = tf.unstack(fea2, self.num_time_steps, 1)

		for fea in tem_feas:
			hidden_state, _ = self.layers[2](fea)

		self.embedded_features = self.layers[3](hidden_state)

		self.center_opt = self.layers[4]([self.embedded_features,
										self.labels])

		self.center = self.layers[4].vars["center"]

		self.outputs = self.layers[5](self.embedded_features)
		# Define loss and optimizer
		self._loss()
		self._accuracy()
		self.opt_op = self.optimizer.minimize(self.loss)



	def _loss(self):
		"""
		Define the loss function

		Params:
			None
		Returns:
			None
		"""
		# regulazation loss
		self.reg_loss = 0
		
		print("||||||||||||||||||||||||")
		for layer in self.layers:
			if type(layer) is Dense:
				self.reg_loss += tf.nn.l2_loss(layer.vars['weights'])
		
		# Cross entropy loss
		self.cross_entropy_loss = get_softmax_cross_entropy(self.outputs, self.labels)
		
		# caculate center loss
		self.center_loss = get_center_loss(self.embedded_features,
										self.labels, self.center)

		self.loss = 0.02 * self.reg_loss + self.cross_entropy_loss + 0.01 * self.center_loss
	

	def _accuracy(self):
		"""
		Caculate accuracy

		Params:
			None
		Returns:
			None
		"""
		self.accuracy = get_accuracy(self.outputs, self.labels)

	def _build(self):
		"""
		Build model
		Input's form is BatchSize x Num_Time_Steps x Num_Channels
		
		Params:
			None
		Returns:
			None
		"""
		self.layers.append(
			Conv1D(num_in_channels=self.num_input_channels,
				num_out_channels=16,
				filter_size=3,
				strides=1,
				padding="SAME",
				dropout=0.0,
				bias=True,
				act=leak_relu
			)
		)

		self.layers.append(
			Conv1D(num_in_channels=16,
				num_out_channels=32,
				filter_size=3,
				strides=1,
				padding="SAME",
				dropout=0.0,
				bias=True,
				act=leak_relu
			)
		)

		self.layers.append(
			LSTM(input_dim=32,
				num_units=128,
				length=self.num_time_steps,
				batch_size=32,
				return_sequece=False,
				bias=True
			)
		)
		
		self.layers.append(
			Dense(input_dim=128,
				output_dim=64,
				dropout=0.0,
				sparse_inputs=False,
				act=leak_relu,
				bias=True
			)
		)

		self.layers.append(CenterLoss(num_classes=self.num_classes,
									num_feas=64, learning_rate=0.5))

		
		self.layers.append(
			Dense(input_dim=64,
				output_dim=self.num_classes,
				dropout=0.0,
				sparse_inputs=False,
				act=leak_relu,
				bias=True
			)
		)








