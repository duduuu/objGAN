import os, time, itertools, pickle
import numpy as np
import tensorflow as tf

# G(z)
def generator(x):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('G_w0', [x.get_shape()[1], 256], initializer=w_init)
    b0 = tf.get_variable('G_b0', [256], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    # 2nd hidden layer
    w1 = tf.get_variable('G_w1', [h0.get_shape()[1], 512], initializer=w_init)
    b1 = tf.get_variable('G_b1', [512], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    # 3rd hidden layer
    w2 = tf.get_variable('G_w2', [h1.get_shape()[1], 1024], initializer=w_init)
    b2 = tf.get_variable('G_b2', [1024], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # output hidden layer
    w3 = tf.get_variable('G_w3', [h2.get_shape()[1], 4096], initializer=w_init)
    b3 = tf.get_variable('G_b3', [4096], initializer=b_init)
    o = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)

    return o

# D(x)
def discriminator(x, drop_out):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('D_w0', [x.get_shape()[1], 1024], initializer=w_init)
    b0 = tf.get_variable('D_b0', [1024], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    h0 = tf.nn.dropout(h0, drop_out)

    # 2nd hidden layer
    w1 = tf.get_variable('D_w1', [h0.get_shape()[1], 512], initializer=w_init)
    b1 = tf.get_variable('D_b1', [512], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
    h1 = tf.nn.dropout(h1, drop_out)

    # 3rd hidden layer
    w2 = tf.get_variable('D_w2', [h1.get_shape()[1], 256], initializer=w_init)
    b2 = tf.get_variable('D_b2', [256], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    h2 = tf.nn.dropout(h2, drop_out)

    # output layer
    w3 = tf.get_variable('D_w3', [h2.get_shape()[1], 1], initializer=w_init)
    b3 = tf.get_variable('D_b3', [1], initializer=b_init)
    o = tf.sigmoid(tf.matmul(h2, w3) + b3)

    return o
    
    
# training parameters
num_examples = 320
batch_size = 64
learning_rate = 0.0002
train_epoch = 500

with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, shape=(None, 128))
    fake_x = generator(z)
    
with tf.variable_scope('D') as scope:
    drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
    x = tf.placeholder(tf.float32, shape=(None, 4096))
    D_real = discriminator(x, drop_out)
    scope.reuse_variables()
    D_fake = discriminator(fake_x, drop_out)

eps = 1e-2
D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if 'D_' in var.name]
G_vars = [var for var in t_vars if 'G_' in var.name]

D_train= tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=D_vars)
G_train = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
sess.close()

loss_d_, loss_g_ = 0, 0

# training-loop
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(train_epoch):
        # update discriminator
        batch_xs = np.zeros((batch_size, 4096))
        noise = np.random.normal(0, 1, (batch_size, 128))

        _, loss_d_ = sess.run([D_train, D_loss], feed_dict = {x: batch_xs, z: noise, drop_out: 0.3})
        _, loss_g_ = sess.run([G_train, G_loss], feed_dict = {z: noise, drop_out: 0.3})
        
        # save images
        if epoch == 0 or (epoch + 1) % 10 == 0:
            noise = np.random.normal(0, 1, (batch_size, 128))
            samples = sess.run(fake_x, feed_dict = {z: noise})
            
            print("epoch : {}".format(str(epoch)))
            print(str(samples))
    
print('end')
