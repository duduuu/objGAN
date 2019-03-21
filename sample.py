import os
import numpy as np
import tensorflow as tf

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        
        # 1st hidden layer
        conv0 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
        lrelu0 = lrelu(tf.layers.batch_normalization(conv0, training=isTrain), 0.2)

        # 2nd hidden layer
        conv1 = tf.layers.conv2d_transpose(lrelu0, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 3rd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 4th hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # output layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 1, [4, 4], strides=(2, 2), padding='same')
        output = tf.nn.tanh(conv4)

        return output
		
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

fake_x = generator(z, isTrain)	

num_samples = 100
batch_size = 64
checkpoint = 'output/checkpoints/100.ckpt'

samples = np.random.rand(num_samples, 64, 64, 1)

saver = tf.train.Saver()

print('start')
with tf.Session() as sess:
	
	saver.restore(sess, checkpoint)
	
	for i in range(int(num_samples / batch_size)):
		noise = np.random.normal(0, 1, (batch_size, 1, 1, 100))
         samples = sess.run(fake_x, feed_dict = {z: noise, isTrain: False})
            
         with open(os.path.join(output_dir, 'samples', 'samples_{}.txt').format(iteration), 'w') as f:
             for i in range(batch_size):
                 output = np.zeros((64,64), dtype='float32')
                 for j in range(64):
                     for k in range(64):
                         output[j][k] = samples[i][j][k][0]
                            
                 f.write(str(output) + "\n")

print('end')
