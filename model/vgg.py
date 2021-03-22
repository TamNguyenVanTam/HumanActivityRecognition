"""
@Authors: TamNV
this file implements some modified architectures of VGG
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, "../base")

from layers import Dense, Conv1D, Flatten, MaxPooling1D, leak_relu
from abstract_model import Model
from utils import *
from metrics import *

class ModifiedModel(Model):
	def __init__(self, placeholders, **kwargs):
		"""
		Initialize method

		Params:
			placeholders: list of placeholders
				which are used for building LSTM model
		Returns:
			None
		"""
		super(ModifiedModel, self).__init__(**kwargs)

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
		#Build sequential layer model
		self.activations.append(self.inputs)
		for layer in self.layers:
			hidden = layer(self.activations[-1])
			self.activations.append(hidden)
		
		print("Modeling sucessfully")
		self.outputs = self.activations[-1]

		#Store model variables for easy access
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
		reg_loss = 0
		print("||||||||||||||||||||||||")
		for layer in self.layers:
			if (type(layer) is Dense):
				reg_loss += tf.nn.l2_loss(layer.vars['weights'])

		# category cross entropy loss
		self.entropy_loss = get_softmax_cross_entropy(self.outputs, self.labels)

		self.loss = self.entropy_loss + 0.02 * reg_loss

	def _accuracy(self):
		"""
		Caculate accuracy

		Params:
			None
		Returns:
			None
		"""
		self.accuracy = get_accuracy(self.outputs, self.labels)

	def predict(self):
		"""
		Perform predicting
		"""
		return tf.argmax(self.outputs)



class VGG8(ModifiedModel):
	def __init__(self, placeholders, **kwargs):
		"""
		Initialize method

		Params:
			placeholders: List of placeholders
				Which are used for building LSTM model
		Returns:
			None
		"""
		super(VGG8, self).__init__(placeholders, **kwargs)

	def _build(self):
		"""
		Build model
		Input's form is BatchSize x Num_Time_Steps x Num_Channels

		Params:
			None
		Returns:
			None
		"""
		self.layers.append(Conv1D(num_in_channels=self.num_input_channels,
								num_out_channels=64,
								filter_size=3,
								strides=1,
								padding="SAME",
								dropout=0.0,
								bias=True,
								act=leak_relu))
		
		self.layers.append(Conv1D(num_in_channels=64,
								num_out_channels=64,
								filter_size=3,
								strides=1,
								padding="SAME",
								dropout=0.0,
								bias=True,
								act=leak_relu))
		
		self.layers.append(MaxPooling1D(ksize=2,
										strides=2,
										padding="VALID"))

		self.layers.append(Conv1D(num_in_channels=64,
								num_out_channels=128,
								filter_size=3,
								strides=1,
								padding="SAME",
								dropout=0.0,
								bias=True,
								act=leak_relu))
		
		self.layers.append(Conv1D(num_in_channels=128,
								num_out_channels=128,
								filter_size=3,
								strides=1,
								padding="SAME",
								dropout=0.0,
								bias=True,
								act=leak_relu))
		
		self.layers.append(Conv1D(num_in_channels=128,
								num_out_channels=128,
								filter_size=3,
								strides=1,
								padding="SAME",
								dropout=0.0,
								bias=True,
								act=leak_relu))

		self.layers.append(MaxPooling1D(ksize=2,
										strides=2,
										padding="VALID"))
		
		self.layers.append(Flatten(num_dims=int(self.num_time_steps/4) * 128))

		self.layers.append(Dense(input_dim=int(self.num_time_steps/4) * 128,
								output_dim=512,
								dropout=0.0,
								act=leak_relu,
								bias=True))

		self.layers.append(Dense(input_dim=512,
								output_dim=256,
								dropout=0.0,
								act=leak_relu,
								bias=True))

		self.layers.append(Dense(input_dim=256,
								output_dim=64,
								dropout=0.0,
								act=leak_relu,
								bias=True))

		self.layers.append(Dense(input_dim=64,
								output_dim=self.num_classes,
								dropout=0.0,
								act=lambda x:x,
								bias=True))
