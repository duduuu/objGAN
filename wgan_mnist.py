import os, pickle, itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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

def discriminator(x, isTrain=True, reuse=False):
	with tf.variable_scope('discriminator', reuse=reuse):
        
		# 1st hidden layer
		conv0 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
		lrelu0 = lrelu(conv0, 0.2)

        # 2nd hidden layer
		conv1 = tf.layers.conv2d(lrelu0, 256, [4, 4], strides=(2, 2), padding='same')
		lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 3rd hidden layer
		conv2 = tf.layers.conv2d(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
		lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 4th hidden layer
		conv3 = tf.layers.conv2d(lrelu2, 1024, [4, 4], strides=(2, 2), padding='same')
		lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # output layer
		conv4 = tf.layers.conv2d(lrelu3, 1, [4, 4], strides=(1, 1), padding='valid')
		output = tf.nn.sigmoid(conv4)
		
		return output, conv4

# training parameters
batch_size = 32
iters = 100
save_every = 10
    
x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

fake_x = generator(z, isTrain)

D_real = discriminator(x, isTrain)
D_fake = discriminator(fake_x, isTrain, reuse=True)

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
G_loss = tf.reduce_mean(D_fake)


t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if 'discriminator' in var.name]
G_vars = [var for var in t_vars if 'generator' in var.name]

D_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D_loss, var_list=D_vars)
G_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss, var_list=G_vars)

#training_data = np.random.rand(num_examples, 64, 64, 1)
#training_data = np.zeros((num_examples, 64, 64, 1), dtype='float32')


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

def inf_train_gen():
    while True:
        np.random.shuffle(training_data)
        for i in range(0, num_examples - batch_size + 1, batch_size):
            yield np.array(training_data[i:i+batch_size], dtype='float32')
            
# training-loop
print('start')
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    # load MNIST
    training_data = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
    training_data = (training_data - 0.5) / 0.5  # normalization; range: -1 ~ 1
    num_examples = mnist.train.num_examples
    
    gen = inf_train_gen()
    
    for iteration in range(iters):
        print(iteration)
        
        noise = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        batch_xs = next(gen)
            
        _ = sess.run(D_train, feed_dict = {x: batch_xs, z: noise, isTrain: True})
        _ = sess.run(G_train, feed_dict = {z: noise, isTrain: True})

        # save data
        if (iteration+1) % save_every == 0:
            noise = np.random.normal(0, 1, (batch_size, 1, 1, 100))
            samples = sess.run(fake_x, feed_dict = {z: noise, isTrain: False})
            
            size_figure_grid = 5
            fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
            for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

            for k in range(size_figure_grid*size_figure_grid):
                i = k // size_figure_grid
                j = k % size_figure_grid
                ax[i, j].cla()
                ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap='gray')
                
            label = 'Epoch {0}'.format(iteration+1)
            fig.text(0.5, 0.04, label, ha='center')

             
print('end')
