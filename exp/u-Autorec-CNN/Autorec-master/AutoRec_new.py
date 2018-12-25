import tensorflow as tf
import time
import numpy as np
import os
import math


class U_AutoRec():
	def __init__(self, sess, args,
				 num_users, num_items,
				 R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings,
				 user_train_set, item_train_set, user_test_set, item_test_set,
				 result_path):

		self.sess = sess
		self.args = args

		self.num_users = num_users
		self.num_items = num_items

		self.R = R
		self.mask_R = mask_R
		self.C = C
		self.train_R = train_R
		self.train_mask_R = train_mask_R
		self.test_R = test_R
		self.test_mask_R = test_mask_R
		self.num_train_ratings = num_train_ratings
		self.num_test_ratings = num_test_ratings

		self.user_train_set = user_train_set
		self.item_train_set = item_train_set
		self.user_test_set = user_test_set
		self.item_test_set = item_test_set

		self.hidden_neuron = args.hidden_neuron
		self.train_epoch = args.train_epoch
		self.batch_size = args.batch_size
		self.l = 0#self.batch_size
		self.num_batch = int(math.ceil(self.num_users / float(self.batch_size)))

		self.base_lr = args.base_lr
		self.optimizer_method = args.optimizer_method
		self.display_step = args.display_step
		self.random_seed = args.random_seed

		self.global_step = tf.Variable(0, trainable=False)
		self.decay_epoch_step = args.decay_epoch_step
		self.decay_step = self.decay_epoch_step * self.num_batch
		self.lr = tf.train.exponential_decay(self.base_lr, self.global_step,
											 self.decay_step, 0.96, staircase=True)
		self.lambda_value = args.lambda_value

		self.train_cost_list = []
		self.test_cost_list = []
		self.test_rmse_list = []

		self.result_path = result_path
		self.window_size = args.window_size
		self.step_size = args.step_size
		self.grad_clip = args.grad_clip

	def run(self):
		self.prepare_model()
		init = tf.global_variables_initializer()
		self.sess.run(init)
		for epoch_itr in range(self.train_epoch):
			self.train_model(epoch_itr)
			self.test_model(epoch_itr)
		self.make_records()

	def prepare_model(self):
		self.input_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R")
		self.input_mask_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_mask_R")

		V = tf.get_variable(name="V", initializer=tf.truncated_normal(shape=[self.num_items, self.hidden_neuron],
																	  mean=0, stddev=0.03), dtype=tf.float32)
		W = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[self.hidden_neuron, self.num_items],
																	  mean=0, stddev=0.03), dtype=tf.float32)
		mu = tf.get_variable(name="mu", initializer=tf.zeros(shape=self.hidden_neuron), dtype=tf.float32)
		b = tf.get_variable(name="b", initializer=tf.zeros(shape=self.num_items), dtype=tf.float32)
		#
		W_1 = tf.get_variable(name="W_1", initializer=tf.truncated_normal(shape=[452, self.hidden_neuron ],
																	  mean=0, stddev=0.03), dtype=tf.float32)
		mu_1 = tf.get_variable(name="mu_1", initializer=tf.zeros(shape=self.hidden_neuron), dtype=tf.float32)
		# pre_Encoder = tf.matmul(self.input_R, V) + mu
		# self.Encoder = tf.nn.sigmoid(pre_Encoder)

		# =====maxpooling=======
		pre_Encoder = tf.nn.sigmoid(tf.matmul(self.input_R, V) + mu)
		pre_Encoder_1 = tf.expand_dims(tf.expand_dims(pre_Encoder, 0), 3)
		pooled_tensor_1, max_indices_1 = tf.nn.max_pool_with_argmax(pre_Encoder_1, ksize=[1, 1, self.window_size, 1],
																strides=[1, 1, self.step_size, 1],
																padding='VALID')
		pooled_tensor = tf.squeeze(pooled_tensor_1)
		max_indices = tf.reshape(tf.to_float(max_indices_1), shape=tf.shape(pooled_tensor))
		Encoder_1 = tf.concat([pooled_tensor, max_indices],
									 axis=-1, name='interaction')
		self.Encoder = tf.nn.sigmoid(tf.matmul(Encoder_1, W_1) + mu_1)
		# pooling = tf.nn.max_pool(pre_Encoder_1, [1, 1, self.window_size, 1],
		# 						 [1, 1, self.step_size, 1],
		# 						 padding='SAME')
		# unpooled_tensor = self.unpool_with_with_argmax(pooled_tensor, max_indices,[1, 1, self.window_size, 1])
		# self.Encoder = tf.layers.dense(inputs=interaction,
		# 								  units=self.hidden_neuron,
		# 								  activation= tf.nn.sigmoid,
		# 									kernel_initializer=tf.contrib.layers.xavier_initializer(),
		# 								  name='encoder')
		# self.Encoder = tf.squeeze(pooled_tensor)
		# ref = tf.Variable(tf.zeros(shape=[None, self.num_items]))
		# =====================

		pre_Decoder = tf.matmul(self.Encoder, W) + b
		self.Decoder = tf.identity(pre_Decoder)

		pre_rec_cost = tf.multiply((self.input_R - self.Decoder), self.input_mask_R)
		rec_cost = tf.square(self.l2_norm(pre_rec_cost))
		pre_reg_cost = tf.square(self.l2_norm(W)) + tf.square(self.l2_norm(V))
		reg_cost = self.lambda_value * 0.5 * pre_reg_cost

		self.cost = rec_cost + reg_cost

		if self.optimizer_method == "Adam":
			optimizer = tf.train.AdamOptimizer(self.lr)
		elif self.optimizer_method == "RMSProp":
			optimizer = tf.train.RMSPropOptimizer(self.lr)
		else:
			raise ValueError("Optimizer Key ERROR")

		if self.grad_clip:
			gvs = optimizer.compute_gradients(self.cost)
			capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
			self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
		else:
			self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)


	def unpool_with_with_argmax(self,pooled, ind, ksize):
		input_shape = tf.shape(pooled)
		output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
		pooled_ = tf.reshape(pooled, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])
		batch_range = tf.cast(tf.reshape(tf.range(output_shape[0]), shape=[input_shape[0], 1, 1, 1]), tf.int64)
		b = tf.ones_like(ind) * batch_range
		b_ = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
		ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
		ind_ = tf.concat([b_, ind_], 1)
		ref = tf.Variable(tf.zeros([input_shape[0], self.l * self.hidden_neuron]))
		unpooled_ = tf.scatter_nd_update(ref, ind_, pooled_)
		unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
		return unpooled

	def train_model(self, itr):
		start_time = time.time()
		random_perm_doc_idx = np.random.permutation(self.num_users)

		batch_cost = 0
		for i in range(self.num_batch):
			if i == self.num_batch - 1:
				self.l = len(random_perm_doc_idx) - i * self.batch_size
				batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
			elif i < self.num_batch - 1:
				self.l = self.batch_size
				batch_set_idx = random_perm_doc_idx[i * self.batch_size: (i + 1) * self.batch_size]

			_, Cost = self.sess.run(
				[self.optimizer, self.cost],
				feed_dict={self.input_R: self.train_R[batch_set_idx, :],
						   self.input_mask_R: self.train_mask_R[batch_set_idx, :]})

			batch_cost = batch_cost + Cost
		self.train_cost_list.append(batch_cost)

		if (itr + 1) % self.display_step == 0:
			print("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(batch_cost),
				  "Elapsed time : %d sec" % (time.time() - start_time))

	def test_model(self, itr):
		start_time = time.time()
		self.l = len(self.test_R)
		Cost, Decoder = self.sess.run(
			[self.cost, self.Decoder],
			feed_dict={self.input_R: self.test_R,
					   self.input_mask_R: self.test_mask_R})

		self.test_cost_list.append(Cost)

		if (itr + 1) % self.display_step == 0:
			Estimated_R = Decoder.clip(min=1, max=5)
			unseen_user_test_list = list(self.user_test_set - self.user_train_set)
			unseen_item_test_list = list(self.item_test_set - self.item_train_set)

			for user in unseen_user_test_list:
				for item in unseen_item_test_list:
					if self.test_mask_R[user, item] == 1:  # exist in test set
						Estimated_R[user, item] = 3

			pre_numerator = np.multiply((Estimated_R - self.test_R), self.test_mask_R)
			numerator = np.sum(np.square(pre_numerator))
			denominator = self.num_test_ratings
			RMSE = np.sqrt(numerator / float(denominator))

			self.test_rmse_list.append(RMSE)

			print("Testing //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(Cost),
				  " RMSE = {:.5f}".format(RMSE),
				  "Elapsed time : %d sec" % (time.time() - start_time))
			print("=" * 100)

	def make_records(self):
		if not os.path.exists(self.result_path):
			os.makedirs(self.result_path)

		basic_info = self.result_path + "basic_info.txt"
		train_record = self.result_path + "train_record.txt"
		test_record = self.result_path + "test_record.txt"

		with open(train_record, 'w') as f:
			f.write(str("Cost:"))
			f.write('\t')
			for itr in range(len(self.train_cost_list)):
				f.write(str(self.train_cost_list[itr]))
				f.write('\t')
			f.write('\n')

		with open(test_record, 'w') as g:
			g.write(str("Cost:"))
			g.write('\t')
			for itr in range(len(self.test_cost_list)):
				g.write(str(self.test_cost_list[itr]))
				g.write('\t')
			g.write('\n')

			g.write(str("RMSE:"))
			for itr in range(len(self.test_rmse_list)):
				g.write(str(self.test_rmse_list[itr]))
				g.write('\t')
			g.write('\n')

		with open(basic_info, 'w') as h:
			h.write(str(self.args))

	def l2_norm(self, tensor):
		return tf.sqrt(tf.reduce_sum(tf.square(tensor)))
