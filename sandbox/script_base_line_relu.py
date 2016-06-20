'''
This script is meant to provide a baseline result and a simple working implementation of a tensorflow model.
The dataset considered are the digits od size 8x8.


'''

import tensorflow as tf
from models import ShallowNetwork
import utils as ff
import numpy as np
import numpy.random as rd

# Number of inputs
n_input = 8*8
n_hidden = 192
n_output = 10

train_prop = .8
N_step = 500
learning_rate = .1

sigma = .1
print_every = 50

x_train,x_test,z_train,z_test = ff.load_small_digits(train_prop,n_output)

########## TF model
# Initialisation of place holders
net = ShallowNetwork(n_input,n_hidden,n_output,sigma)
tf.scalar_summary('accuracy', net.accuracy)

########### Optimisation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(net.loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(N_step):
  sess.run(train_step, feed_dict={net.x: x_train, net.out_: z_train})
  sess.run(tf.assign(net.W_hid,tf.clip_by_value(net.W_hid,-1,1)))
  sess.run(tf.assign(net.W_out,tf.clip_by_value(net.W_out,-1,1)))

  if np.mod(i,print_every) == 0:
      acc = sess.run(net.accuracy, feed_dict={net.x: x_train, net.out_: z_train})
      print('Training accuracy at step %s: %s' % (i, acc))

print('Testing accuracy: ')
print(sess.run(net.accuracy, feed_dict={net.x: x_test, net.out_: z_test}))
