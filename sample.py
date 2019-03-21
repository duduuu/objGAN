import os
import numpy as np
import tensorflow as tf

import models

num_samples = 100
batch_size = 64
output_dir = 'output'
checkpoint_dir = os.path.join(output_dir, 'checkpoints', 'checkpoint_4.ckpt')


z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

fake_x = models.generator(z, isTrain)	
	
samples = np.random.rand(num_samples, 64, 64, 1)

saver = tf.train.Saver()

print('start')
with tf.Session() as sess:
	
	saver.restore(sess, checkpoint_dir)
	
	for i in range(int(num_samples / batch_size)):
		noise = np.random.normal(0, 1, (batch_size, 1, 1, 100))
		samples = sess.run(fake_x, feed_dict = {z: noise, isTrain: False})
		
		with open(os.path.join(output_dir, 'samples', 'samples!.txt'), 'a') as f:
			for i in range(batch_size):
				output = np.zeros((64,64), dtype='float32')
				for j in range(64):
					for k in range(64):
						output[j][k] = samples[i][j][k][0]
						
				f.write(str(output) + "\n")

print('end')
