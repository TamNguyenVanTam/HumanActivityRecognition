"""
Authors: TamNV
"""
import sys
import numpy as np

sys.path.insert(0, "../../base")

from trainer import Trainer
from utils import *

class HARTrainer(Trainer):
	"""
	perform traffic jam training process
	"""
	def __init__(self, exe_config, **kwargs):
		self.exe_config = exe_config
	
	def _train(self):
		"""
		Minimize Model's loss function

		Params:
			None
		Returns:
			None
		"""
		best_acc = 0.0
		learning_rate = 1e-3
		num_no_improve = 0
		
		for epoch in range(self.exe_config[KEY_NUM_EPOCHS]):
			print("epoch {}".format(epoch))
			# perfoming training
			train_log, valid_log = [], []
			reset = True
			for i in range(self.num_train_iters):
				batch_x, batch_y =  self.data.get_next_batch_train(
										self.exe_config[KEY_BATCH_SIZE],
										reset
									)
				feed_dict = {
								self.model.inputs: batch_x,
								self.model.labels: batch_y,
								self.model.learning_rate: learning_rate,
								self.model.dropout: 0.0,
								self.model.decay_factor: 0.0
							}
				_, loss, acc = self.sess.run([self.model.opt_op, self.model.loss, self.model.accuracy], feed_dict=feed_dict)

				if hasattr(self.model, 'center_opt'):
					_ = self.sess.run(self.model.center_opt, feed_dict = feed_dict)
				
				train_log.append([loss, acc])
				reset=False
			
			reset = True
			for i in range(self.num_valid_iters):

				batch_x, batch_y = self.data.get_next_batch_valid(
										self.exe_config[KEY_BATCH_SIZE],
										reset
									)
				feed_dict = {
								self.model.inputs: batch_x,
								self.model.labels: batch_y,
								self.model.dropout: 0.0,
								self.model.decay_factor:0.0
							}

				loss, acc = self.sess.run([self.model.loss, self.model.accuracy],
											feed_dict=feed_dict
										)
				valid_log.append([loss, acc])
				reset = False
			
			train_log, valid_log = np.array(train_log), np.array(valid_log)
		
			print("Training Set Total loss {:.3f}, acc {:.3f}".
				format(np.mean(train_log[:, 0]), np.mean(train_log[:, 1]) * 100))
			
			print("Validation Set Total loss {:.3f}, acc {:.3f}"
				.format(np.mean(valid_log[:, 0]), np.mean(valid_log[:, 1]) * 100))
			
			if best_acc < np.mean(valid_log[:, 1]):
				best_acc = np.mean(valid_log[:, 1])
				self.model.save(self.exe_config[KEY_CHECKPOINT], self.sess)
				num_no_improve = 0
			else:
				num_no_improve += 1
			if num_no_improve == 5:
				learning_rate = learning_rate * 0.99
				print("Learning Rate Decay: {:.5f}".format(learning_rate))

		print("Optimizal point. Accuracy {:.3f}".format(best_acc))
