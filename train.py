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
    
    
x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

fake_x = generator(z, isTrain)

D_real = discriminator(x, isTrain)
D_fake = discriminator(fake_x, isTrain, reuse=True)

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
G_loss = tf.reduce_mean(D_fake)

alpha = tf.random_uniform(shape=[1], minval=0, maxval=1.)

differences = fake_x - x
interpolates = x + (alpha*differences)
gradients = tf.gradients(discriminator(interpolates, isTrain, reuse=True), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
D_loss += 10 * gradient_penalty

t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if 'discriminator' in var.name]
G_vars = [var for var in t_vars if 'generator' in var.name]

D_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D_loss, var_list=D_vars)
G_train = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss, var_list=G_vars)

# training parameters
num_examples = 1000
batch_size = 64
iters = 100
save_every = 50
output_dir = 'output'

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if not os.path.isdir(os.path.join(output_dir, 'checkpoints')):
    os.makedirs(os.path.join(output_dir, 'checkpoints'))

if not os.path.isdir(os.path.join(output_dir, 'samples')):
    os.makedirs(os.path.join(output_dir, 'samples'))
    

training_data = np.random.rand(num_examples, 64, 64, 1)
#training_data = np.ones((num_examples, 64, 64, 1), dtype='float32')

def inf_train_gen():
    while True:
        np.random.shuffle(training_data)
        for i in range(0, num_examples - batch_size + 1, batch_size):
            yield np.array(training_data[i:i+batch_size], dtype='float32')
            
# open session and initialize all variables
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
        
        # update discriminator
        noise = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        batch_xs = next(gen)
            
        _, _ = sess.run([D_train, D_loss], feed_dict = {x: batch_xs, z: noise, isTrain: True})
        _, _ = sess.run([G_train, G_loss], feed_dict = {z: noise, isTrain: True})

        # save data
        if (iteration+1) % save_every == 0:
            noise = np.random.normal(0, 1, (batch_size, 1, 1, 100))
            samples = sess.run(fake_x, feed_dict = {z: noise, isTrain: False})
            
            with open(os.path.join(output_dir, 'samples', 'samples_{}.txt').format(iteration), 'w') as f:
                for i in range(batch_size):
                    output = np.zeros((64,64), dtype='float32')
                    for j in range(64):
                        for k in range(64):
                            output[j][k] = samples[i][j][k][0]
                            
                    f.write(str(output) + "\n")
                
        # save model
        if (iteration+1) % save_every == 0:
            model_saver.save(sess, os.path.join(output_dir, 'checkpoints'), global_step=iteration)

             
print('end')
