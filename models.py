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
