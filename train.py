import os
import numpy as np
import tensorflow as tf

import models

# training parameters
num_examples = 300
batch_size = 32
iters = 4
save_every = 2
output_dir = 'output'

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if not os.path.isdir(os.path.join(output_dir, 'checkpoints')):
    os.makedirs(os.path.join(output_dir, 'checkpoints'))

if not os.path.isdir(os.path.join(output_dir, 'samples')):
    os.makedirs(os.path.join(output_dir, 'samples'))
	
    
x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

fake_x = models.generator(z, isTrain)

D_real = models.discriminator(x, isTrain)
D_fake = models.discriminator(fake_x, isTrain, reuse=True)

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
G_loss = tf.reduce_mean(D_fake)

alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1.)
differences = fake_x - x
interpolates = x + (alpha*differences)
gradients = tf.gradients(models.discriminator(interpolates, isTrain, reuse=True), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
D_loss += 10 * gradient_penalty

t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if 'discriminator' in var.name]
G_vars = [var for var in t_vars if 'generator' in var.name]

D_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D_loss, var_list=D_vars)
G_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss, var_list=G_vars)
    
training_data = np.random.rand(num_examples, 64, 64, 1)
#training_data = np.ones((num_examples, 64, 64, 1), dtype='float32')

def inf_train_gen():
    while True:
        np.random.shuffle(training_data)
        for i in range(0, num_examples - batch_size + 1, batch_size):
            yield np.array(training_data[i:i+batch_size], dtype='float32')
            

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
sess.close()

model_saver = tf.train.Saver()

# training-loop
print('start')
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
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
            
            with open(os.path.join(output_dir, 'samples', 'samples_{}.txt').format(iteration+1), 'w') as f:
                for i in range(batch_size):
                    output = np.zeros((64,64), dtype='float32')
                    for j in range(64):
                        for k in range(64):
                            output[j][k] = samples[i][j][k][0]
                            
                    f.write(str(output) + "\n")
                
        # save model
        if (iteration+1) % save_every == 0:
            model_saver.save(sess, os.path.join(output_dir, 'checkpoints', 'checkpoint_{}.ckpt').format(iteration+1))

             
print('end')
