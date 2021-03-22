"""
#Authors: TamNV
This file implements 4 basic layers in tensorlow 
	+Fully connected, LSTM, Graph Convolutional Neural Network
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from inits import uniform, glorot, zeros, ones
from abstract_layer import *

def leak_relu(x):
	return tf.maximum(x*0.2, x)

class Dense(Layer):
    """
    Dense Layer.
    """
    def __init__(self, input_dim, output_dim, dropout=0.0, sparse_inputs=False,
                    act=tf.nn.relu, bias=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.input_dim = input_dim

        #helper variable for sparse dropout
        self.num_features_nonzero = input_dim

        with tf.variable_scope("{}_vars".format(self.name)):
            self.vars['weights'] = glorot([input_dim, output_dim],\
                                            name='weights')
            if self.bias:
                self.vars["bias"] = zeros([output_dim], name="bias")
        
        if self.logging:
            self._log_vars()
        
    def _call(self, inputs):
        x = inputs
       
        # x = tf.nn.dropout(x, rate=0.2)

        outputs = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        #Bias
        if self.bias:
            outputs += self.vars["bias"]
        
        return self.act(outputs)


class LSTM(Layer):
	"""
	define LSTM layer
	"""
	def __init__(self, input_dim, num_units,
				length, batch_size, return_sequece,
				bias, **kwargs):
		"""
		initialize method
		params:
			input_dim: integer
				the dimension of inputs
			num_units: integer
				the number of hiddens
			bias: Boolean
				the boolean number
		returns:
			none
		"""
		super(LSTM, self).__init__(**kwargs)

		self.input_dim = input_dim
		self.num_units = num_units
		self.return_sequece = return_sequece
		self.length = length
		self.batch_size = batch_size
		self.bias = bias

		with tf.variable_scope("{}_vars".format(self.name)):
			self.vars["fg"] = glorot([self.input_dim+self.num_units, self.num_units], name="forget_weight")
			self.vars["il"] = glorot([self.input_dim+self.num_units, self.num_units], name="input_weight")
			self.vars["ol"] = glorot([self.input_dim+self.num_units, self.num_units], name="output_weight")
			self.vars["Cl"] = glorot([self.input_dim+self.num_units, self.num_units], name="cell_weight")

			if self.bias:
				self.vars["fgb"] = zeros([self.num_units], name="forget_bias")
				self.vars["ilb"] = zeros([self.num_units], name="input_bias")
				self.vars["olb"] = zeros([self.num_units], name="output_bias")
				self.vars["Clb"] = zeros([self.num_units], name="cell_bias")

		self.hidden_state, self.cell_state = self.init_hidden(self.batch_size)

		if self.logging:
			self._log_vars()

	def perform(self, input, hidden_state, cell_state):
		"""
		perform one lstm step
		params:
			input: tensor object
			hidden_state: previous hidden state
			cell_state: previous cell state
		returns:
			next_hidden_state, next_cell_state 
		"""
		x = input
		combined = tf.concat([x, hidden_state], axis=-1)

		f = tf.matmul(combined, self.vars["fg"])
		i = tf.matmul(combined, self.vars["il"])
		o = tf.matmul(combined, self.vars["ol"])
		C = tf.matmul(combined, self.vars["Cl"])
		if self.bias:
			f += self.vars["fgb"]
			i += self.vars["ilb"]
			o += self.vars["olb"]
			C += self.vars["Clb"]
		f = tf.nn.sigmoid(f)
		i = tf.nn.sigmoid(i)
		o = tf.nn.sigmoid(o)
		C = tf.nn.tanh(C)

		cell_state = f*cell_state + i*C
		hidden_state = o*tf.nn.tanh(cell_state)
		return hidden_state, cell_state
	
	def init_hidden(self, batch_size=-1):
		hidden_state = zeros([batch_size, self.num_units], name="hidden_state")
		cell_state = zeros([batch_size, self.num_units], name="cell_state")
		return hidden_state, cell_state

	def _call(self, inputs):
		"""
		perform LSTM layer for one time step
	
		params:
			inputs: Tensor object
				only one time step
		Returns:
			outputs: Tensor object	
		"""
		x = inputs

		self.hidden_state, self.cell_state = self.perform(x, self.hidden_state, self.cell_state)
		
		return self.hidden_state, self.cell_state


class Conv1D(Layer):
	"""
	Convolution 1D
	"""
	def __init__(self, num_in_channels, num_out_channels, filter_size, 
					strides, padding, dropout, bias, act, **kwargs):

		super(Conv1D, self).__init__(**kwargs)

		self.num_in_channels = num_in_channels
		self.num_out_channels = num_out_channels
		self.filter_size = filter_size
		self.strides = strides
		self.padding = padding
		self.dropout = dropout

		self.num_features_nonzero = num_in_channels
		self.bias = bias
		self.act = act

		with tf.variable_scope("{}_vars".format(self.name)):
			self.vars["weights"] = glorot([filter_size, self.num_in_channels,
											self.num_out_channels], name="weights")
			if self.bias:
				self.vars["bias"] = zeros([self.num_out_channels], name="bias")

	def _call(self, inputs):
		"""
		Perform convolution 1D

		Params:
			inputs: Tensor object
				[batch, in_width, in_channels]
		Returns:
			outputs: Tensor object
		"""
		

		x = inputs

		x = tf.nn.conv1d(x, self.vars["weights"],
						[1, self.strides, 1], self.padding)
		if self.bias:
			x += self.vars["bias"]

		outputs = self.act(x)
		return outputs


class MaxPooling1D(Layer):
	"""
	Maxpooling 1D
	"""
	def __init__(self, ksize, strides, padding, **kwargs):
		super(MaxPooling1D, self).__init__(**kwargs)

		self.ksize = ksize
		self.strides = strides
		self.padding = padding
	
	def _call(self, inputs):
		"""
		Perform maxpooling 1D operation
		"""
		x = tf.expand_dims(inputs, axis=-1)

		outputs = tf.nn.max_pool2d(x, [1, self.ksize, 1, 1],
									[1, self.strides, 1, 1],
									self.padding)
		outputs = tf.squeeze(outputs, axis=-1)

		return outputs

class Flatten(Layer):
	def __init__(self, num_dims, **kwargs):
		super(Flatten, self).__init__(**kwargs)
		self.num_dims = num_dims

	def _call(self, inputs):
		outputs = tf.reshape(inputs, (-1, self.num_dims))
		return outputs

class CenterLoss(Layer):
	"""
	Perform center loss
	"""
	def __init__(self, num_classes, num_feas, learning_rate, **kwargs):
		super(CenterLoss, self).__init__(**kwargs)		
		
		self.num_classes = num_classes
		self.num_feas = num_feas
		self.learning_rate = learning_rate

		# Declare variables
		with tf.variable_scope("{}_vars".format(self.name)):
			self.vars["center"] = zeros(shape=[self.num_classes, self.num_feas], 
										trainable=False)	

	def _call(self, inputs):
		"""
		Perform center loss layer

		Params:
			inputs: Tensor object
				Embedding features:	N_Classes x N_Embedding
			labels: Tensor object
				Labels of this batchs: N_Samples x N x Classes 
		Returns:
			center loss optimizer
		"""
		embeded_preds = inputs[0]
		labels = inputs[1]


		_labels = tf.cast(labels, tf.float32)

		embeded_labels = tf.matmul(_labels, self.vars["center"])

		diff = embeded_labels - embeded_preds

		_labels = tf.transpose(_labels)
		grad = tf.matmul(_labels, diff)

		center_counts = tf.reduce_sum(_labels, axis=1, keepdims=True) + 1.0
		
		grad /= center_counts

		updated_center = self.vars["center"] - self.learning_rate * grad
		
		center_loss_opt = tf.assign(self.vars["center"], updated_center)

		return center_loss_opt
