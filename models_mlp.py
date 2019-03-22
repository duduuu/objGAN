import tensorflow as tf
import tensorflow.contrib.layers as tcl

def lrelu(x, th=0.2, name="lrelu"):
    return tf.maximum(th * x, x)
	
def generator(x, isTrain=True, reuse=False):
	with tf.variable_scope('generator', reuse=reuse):
	
		output = tcl.fully_connected(x, 512, activation_fn=lrelu, normalizer_fn = tcl.batch_norm)
		output = tcl.fully_connected(x, 64, activation_fn=lrelu, normalizer_fn = tcl.batch_norm)
		output = tcl.fully_connected(x, 64, activation_fn=lrelu, normalizer_fn = tcl.batch_norm)
		output = tcl.fully_connected(x, 64 * 64 * 1, activation_fn=tf.nn.tanh, normalizer_fn = tcl.batch_norm)
		
		return output

def discriminator(x, isTrain=True, reuse=False):
	with tf.variable_scope('discriminator', reuse=reuse):
        
		output = tcl.fully_connected(x, 64 * 64 * 1, activation_fn=tf.nn.relu, normalizer_fn = tcl.batch_norm)
		output = tcl.fully_connected(output, 64, activation_fn=tf.nn.relu, normalizer_fn = tcl.batch_norm)
		output = tcl.fully_connected(output, 64, activation_fn=tf.nn.relu, normalizer_fn = tcl.batch_norm)
		logit = tcl.fully_connected(output, 1, activation_fn=None)
		
		return logit
