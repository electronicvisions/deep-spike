'''
This script is meant to provide a baseline result and a simple working implementation of a tensorflow model.
The dataset considered are the digits od size 8x8.


'''

import tensorflow as tf
from bin.models import ShallowNetwork, ShallowOverwritingNetwork
import bin.data_loader as ff
import numpy as np
from cc import binarized_ops, binarized_grad

########## Parameters
# Number of neurons
n_input = 8 * 8
n_hidden = 192
n_output = 10

weight_quantization = 4 # Number of possible weights values strictly above 0. There is also 0 and the symmetric.

train_prop = .8 # Proportion of data selected for training set
N_step = 200 # Gradient descent iterations
learning_rate = .1 # Learning rate of the gradient descent

sigma = 1. # Initial weight standard deviation
print_every = 20 # Print accuracy every 'print_every' steps
h = binarized_ops.binarized # Activation function

print('Digit classification with weights quantized to {} values'.format(weight_quantization*2 +1))

def quantize_weight(W, precision=weight_quantization):
    '''
    For a given weight matrix, returns weights of values -1, 0 or 1
    :param W:
    :return:
    '''
    W_ = tf.round(W * precision) / precision
    return W_

########## Loading the dataset
x_train, x_test, z_train, z_test = ff.load_small_digits(train_prop, n_output)

########## TensorFlow network models
# A first network is kept for forward pass
# (computing output of each layers with possible distortion: binary weights etc...)
net_forward = ShallowNetwork(n_input, n_hidden, n_output, sigma, h)

# A second network is used for training only
net_backward = ShallowOverwritingNetwork(n_input, n_hidden, n_output, sigma, h)

# Define gradient descient step with TensorFlow built in functions
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(net_backward.loss)

########## Running the model
# TensorFlow initialization
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

sess.run(tf.assign(net_backward.W_hid, quantize_weight(net_backward.W_hid)))
sess.run(tf.assign(net_backward.W_out, quantize_weight(net_backward.W_out)))


for i in range(N_step):
    # Forward Pass with distorted model
    forward_y, forward_out = sess.run([net_forward.y, net_forward.z],
                                      feed_dict={net_forward.x: x_train, net_forward.z_: z_train})

    # Backward pass with imposed hidden and ouput layers outputs
    sess.run(train_step, feed_dict={net_backward.x: x_train, net_backward.forward_y: forward_y,
                                    net_backward.forward_z: forward_out,
                                    net_backward.z_: z_train})

    sess.run(tf.assign(net_backward.W_hid, tf.clip_by_value(net_backward.W_hid,-1,1)))
    sess.run(tf.assign(net_backward.W_out, tf.clip_by_value(net_backward.W_out,-1,1)))

    # Set the quantized weight to the forward pass network
    sess.run(tf.assign(net_forward.W_hid, quantize_weight(net_backward.W_hid)))
    sess.run(tf.assign(net_forward.W_out, quantize_weight(net_backward.W_out)))

    # Set the bias
    sess.run(tf.assign(net_forward.b_hid, net_backward.b_hid))
    sess.run(tf.assign(net_forward.b_out, net_backward.b_out))

    # Printing along iterations
    if np.mod(i, print_every) == 0:
        acc_train = sess.run(net_forward.accuracy, feed_dict={net_forward.x: x_train, net_forward.z_: z_train})
        acc_test = sess.run(net_forward.accuracy, feed_dict={net_forward.x: x_test, net_forward.z_: z_test})
        print('Accuracy \t  step {} \t Training {:.3g} \t Testing {:.3g}'.format(i, acc_train,acc_test))

print(' ---- Final test accuracy: ')
print(sess.run(net_forward.accuracy, feed_dict={net_forward.x: x_test, net_forward.z_: z_test}))
