import tensorflow as tf

# '''
# 二 反池化操作
# '''
#
#
# def unpool_with_with_argmax(pooled, ind,ksize=[1, 1, 2, 1]):
# 	input_shape = tf.shape(pooled)
# 	output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
# 	pooled_ = tf.reshape(pooled, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])
# 	batch_range = tf.cast(tf.reshape(tf.range(output_shape[0]), shape=[input_shape[0], 1, 1, 1]),tf.int64)
# 	b = tf.ones_like(ind) * batch_range
# 	b_ = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
# 	ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
# 	ind_ = tf.concat([b_, ind_], 1)
# 	ref = tf.Variable(tf.zeros([1,a*4*1]))
# 	unpooled_ = tf.scatter_nd_update(ref, ind_, pooled_)
# 	unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
# 	return unpooled
#
#
# a=3
original_tensor_1 = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="original_tensor")
t = tf.placeholder(dtype=tf.float32, shape=[1], name="t")
original_tensor = tf.expand_dims(tf.expand_dims(original_tensor_1, 0), 3)
pooled_tensor, max_indices = tf.nn.max_pool_with_argmax(original_tensor, ksize=[1,1, 2, 1], strides=[1, 1, 2, 1],
														padding='SAME')
pooled_tensor_1 = tf.squeeze(pooled_tensor)
max_indices_1 = tf.reshape(tf.to_float(max_indices),shape=tf.shape(pooled_tensor_1))
interaction = tf.concat([pooled_tensor_1, max_indices_1],
									 axis=-1, name='interaction')
# pooled_tensor_1 = tf.squeeze(pooled_tensor)
# unpooled_tensor = unpool_with_with_argmax(pooled_tensor, max_indices,t)
with tf.Session() as sess:
	# print(sess.run(original_tensor))
	# print(sess.run(pooled_tensor))
	# print(sess.run(max_indices))
	# sess.run(tf.global_variables_initializer())
	# print(sess.run(unpooled_tensor).shape)
	sess.run(tf.global_variables_initializer())
	a=[[1, 2, 3,1], [1,2, 1, 1], [1,2, 2, 2]]
	pooled_tensor = sess.run([interaction], feed_dict={original_tensor_1: a})
	print(pooled_tensor)










# def max_unpool_2x2(x, output_shape):
#     out = tf.concat_v2([x, tf.zeros_like(x)], 3)
#     out = tf.concat_v2([out, tf.zeros_like(out)], 2)
#     out_size = output_shape
#     return tf.reshape(out, out_size)
#
# def max_unpool_2x2(x, shape):
#     inference = tf.image.resize_nearest_neighbor(x, tf.pack([shape[1]*2, shape[2]*2]))
#     return inference
# # PI is the 4-dimension Tensor from above layer
# unpool1 = tf.image.resize_images(PI, size = [resize_width, resize_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# def max_pool_with_argmax(net, stride):
# 	'''
# 	重定义一个最大池化函数，返回最大池化结果以及每个最大值的位置(是个索引，形状和池化结果一致)
#
# 	args:
# 		net:输入数据 形状为[batch,in_height,in_width,in_channels]
# 		stride：步长，是一个int32类型，注意在最大池化操作中我们设置窗口大小和步长大小是一样的
# 	'''
# 	# 使用mask保存每个最大值的位置 这个函数只支持GPU操作
# 	_, mask = tf.nn.max_pool_with_argmax(net, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],
# 										 padding='SAME')
# 	# 将反向传播的mask梯度计算停止
# 	mask = tf.stop_gradient(mask)
# 	# 计算最大池化操作
# 	net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
# 	# 将池化结果和mask返回
# 	return net, mask
#
#
# def un_max_pool(net, mask, stride):
# 	'''
# 	定义一个反最大池化的函数，找到mask最大的索引，将max的值填到指定位置
# 	args:
# 		net:最大池化后的输出，形状为[batch, height, width, in_channels]
# 		mask：位置索引组数组，形状和net一样
# 		stride:步长，是一个int32类型，这里就是max_pool_with_argmax传入的stride参数
# 	'''
# 	ksize = [1, stride, stride, 1]
# 	input_shape = net.get_shape().as_list()
# 	#  calculation new shape
# 	output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
# 	# calculation indices for batch, height, width and feature maps
# 	one_like_mask = tf.ones_like(mask)
# 	batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
# 	b = one_like_mask * batch_range
# 	y = mask // (output_shape[2] * output_shape[3])
# 	x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
# 	feature_range = tf.range(output_shape[3], dtype=tf.int64)
# 	f = one_like_mask * feature_range
# 	# transpose indices & reshape update values to one dimension
# 	updates_size = tf.size(net)
# 	indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
# 	values = tf.reshape(net, [updates_size])
# 	ret = tf.scatter_nd(indices, values, output_shape)
# 	return ret
#
#
# # def max_unpool_2x2(x, shape):
# # 	inference = tf.image.resize_nearest_neighbor(x, tf.pack([shape[1] * 2, shape[2] * 2]))
# # 	return inference
#
#
# # 定义一个形状为4x4x2的张量
# img = tf.constant([
# 	[[0.0, 4.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
# 	[[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],
# 	[[2.0, 6.0], [2.0, 6.0], [2.0, 6.0], [2.0, 6.0]],
# 	[[3.0, 7.0], [3.0, 7.0], [3.0, 7.0], [3.0, 7.0]],
# ])
#
# img = tf.reshape(img, [1, 4, 4, 2])
# # 最大池化操作
# pooling1 = tf.squeeze(tf.nn.max_pool(img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))
# # 带有最大值位置的最大池化操作
# pooling2, mask = max_pool_with_argmax(img, 2)
# # 反最大池化
# img2 = tf.squeeze(un_max_pool(pooling2, mask, 2))
# unpool1 = tf.image.resize_images(pooling2, size=[4, 4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# unpool2 = unpool_with_with_argmax(pooling2, mask )
# with tf.Session() as sess:
# 	print('image:')
# 	image = sess.run(img)
# 	print(image)
#
# 	# 默认的最大池化输出
# 	result = sess.run(pooling1)
# 	print('max_pool:\n', result)
#
# 	# 带有最大值位置的最大池化输出
# 	result, mask2 = sess.run([pooling2, mask])
# 	print('max_pool_with_argmax:\n', result, mask2)
#
# 	# 反最大池化输出
# 	result = sess.run(img2)
# 	print('un_max_pool', result)
# 	unpool1 = sess.run(unpool1)
# 	print('un_max_pool_1', unpool1)
# 	unpool2 = sess.run(unpool2)
# 	print('un_max_pool_2', unpool2)
#
#
# #
# # a = tf.reshape(tf.constant([
# # 	[[1.0, 2.0, 3.0, 4.0],
# # 	 [5.0, 6.0, 7.0, 8.0],
# # 	 [8.0, 7.0, 6.0, 5.0],
# # 	 [4.0, 3.0, 2.0, 1.0]]
# # ]), (4, 4))
# # # b=a.shape
# # a = tf.expand_dims(tf.expand_dims(a, 0), 3)  # tf.reshape(a, [1, 4, 4, 1])
# #
# # pooling = tf.nn.max_pool(a, [1, 1, 2, 1], [1, 1, 1, 1], padding='SAME')
# #
# # # c = tf.stack(pooling,axis = 1)
# # c = tf.squeeze(pooling)
# # # c = tf.reshape(pooling, [4,4])
# # with tf.Session() as sess:
# # 	b = a.shape
# # 	print("image:")
# # 	image = sess.run(a)
# # 	print(image)
# # 	print("reslut:")
# # 	result = sess.run(pooling)
# # 	print(result.shape)
# # 	c = sess.run(c)
# # 	print(c)
# # 	print(c.shape)
# #
# #
# #
# #
# #
# #
# #
# # 	# import tensorflow as tf
# # 	# import time
# # 	# import numpy as np
# # 	# import os
# # 	# import math
# # 	#
# # 	# import tensorflow as tf
# # 	#
# # 	# temp = [0., 0., 1., 0., 0., 0., 1.5, 2.5]
# # 	#
# # 	# # Reshape the tensor to be 3 dimensions.
# # 	# values = tf.reshape(temp, [1, 8, 1])
# # 	# # Use max with this pool.
# # 	# p_max = tf.nn.pool(input=values,
# # 	#     window_shape=[2],
# # 	#     pooling_type="MAX",
# # 	#     padding="SAME")
# # 	# # padding = "SAME")
# # 	# session = tf.Session()
# # 	#
# # 	# # Print our tensors.
# # 	# print("VALUES")
# # 	# print(session.run(values).transpose())
# # 	# print("POOL MAX")
# # 	# print(session.run(p_max).transpose())
# # 	#
# # 	#
# # 	#
# # 	#
# # 	#
# # 	#
# # 	#
# # 	#
# # 	# # '''
# # 	# # 二 反池化操作
# # 	# # '''
# # 	# #
# # 	# #
# # 	#
# # 	# #
# # 	# #
# # 	# # def un_max_pool(net, mask, stride):
# # 	# # 	'''
# # 	# # 	定义一个反最大池化的函数，找到mask最大的索引，将max的值填到指定位置
# # 	# # 	args:
# # 	# # 		net:最大池化后的输出，形状为[batch, height, width, in_channels]
# # 	# # 		mask：位置索引组数组，形状和net一样
# # 	# # 		stride:步长，是一个int32类型，这里就是max_pool_with_argmax传入的stride参数
# # 	# # 	'''
# # 	# # 	ksize = [1, stride, stride, 1]
# # 	# # 	input_shape = net.get_shape().as_list()
# # 	# # 	#  calculation new shape
# # 	# # 	output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
# # 	# # 	# calculation indices for batch, height, width and feature maps
# # 	# # 	one_like_mask = tf.ones_like(mask)
# # 	# # 	batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
# # 	# # 	b = one_like_mask * batch_range
# # 	# # 	y = mask // (output_shape[2] * output_shape[3])
# # 	# # 	x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
# # 	# # 	feature_range = tf.range(output_shape[3], dtype=tf.int64)
# # 	# # 	f = one_like_mask * feature_range
# # 	# # 	# transpose indices & reshape update values to one dimension
# # 	# # 	updates_size = tf.size(net)
# # 	# # 	indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
# # 	# # 	values = tf.reshape(net, [updates_size])
# # 	# # 	ret = tf.scatter_nd(indices, values, output_shape)
# # 	# # 	return ret
# # 	# #
# # 	# #
# # 	# # # 定义一个形状为4x4x2的张量
# # 	# # img = tf.constant([
# # 	# # 	[[0.0, 4.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
# # 	# # 	[[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],
# # 	# # 	[[2.0, 6.0], [2.0, 6.0], [2.0, 6.0], [2.0, 6.0]],
# # 	# # 	[[3.0, 7.0], [3.0, 7.0], [3.0, 7.0], [3.0, 7.0]],
# # 	# # ])
# 	# #
# 	# # img = tf.reshape(img, [1, 4, 4, 2])
# 	# # # 最大池化操作
# 	# # pooling1 = tf.nn.max_pool(img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 	# # # 带有最大值位置的最大池化操作
# 	# # pooling2, mask = max_pool_with_argmax(img, 2)
# 	# # # 反最大池化
# 	# # img2 = un_max_pool(pooling2, mask, 2)
# 	# # with tf.Session() as sess:
# 	# # 	print('image:')
# 	# # 	image = sess.run(img)
# 	# # 	print(image)
# 	# #
# 	# # 	# 默认的最大池化输出
# 	# # 	result = sess.run(pooling1)
# 	# # 	print('max_pool:\n', result)
# 	# #
# 	# # 	# 带有最大值位置的最大池化输出
# 	# # 	result, mask2 = sess.run([pooling2, mask])
# 	# # 	print('max_pool_with_argmax:\n', result, mask2)
# 	# #
# 	# # 	# 反最大池化输出
# 	# # 	result = sess.run(img2)
# 	# # 	print('un_max_pool', result)
