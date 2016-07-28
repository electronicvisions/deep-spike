'''
This script is meant to provide a baseline result and a simple working implementation of a tensorflow model.
The dataset considered are the digits od size 8x8.


'''

import tensorflow as tf
from models import ShallowNetwork,ShallowOverwritingNetwork
import data_loader as ff
import numpy as np
import numpy.random as rd

# Number of inputs
n_input = 8*8
n_hidden = 192
n_output = 10

train_prop = .8
N_step = 200
learning_rate = .1

sigma = .1
print_every = 50

x_train,x_test,z_train,z_test = ff.load_small_digits(train_prop,n_output)

########## TF model
# Initialisation of place holders
net_forward = ShallowNetwork(n_input, n_hidden, n_output, sigma)
net_backward = ShallowOverwritingNetwork(n_input, n_hidden, n_output, sigma)

########### Optimisation
grads = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(net_backward.accuracy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(N_step):
    # Forward Pass with distorted model
    forward_y, forward_out = sess.run([net_forward.y, net_forward.z],
                                      feed_dict={net_forward.x: x_train, net_forward.z_: z_train})

    # Backward pass with imposed hidden and ouput layers outputs
    sess.run(grads, feed_dict={net_backward.x: x_train, net_backward.forward_y: forward_y,
                                    net_backward.forward_z: forward_out,
                                    net_backward.z_: z_train})
    print(grads)


    sess.run(tf.assign(net_forward.W_hid, tf.clip_by_value(net_forward.W_hid, -1, 1)))
    sess.run(tf.assign(net_forward.W_out, tf.clip_by_value(net_forward.W_out, -1, 1)))

    if np.mod(i,print_every) == 0:
      acc = sess.run(net.accuracy, feed_dict={net.x: x_train, net.z_: z_train})
      print('Training accuracy at step %s: %s' % (i, acc))

print('Testing accuracy: ')
print(sess.run(net.accuracy, feed_dict={net.x: x_test, net.z_: z_test}))
