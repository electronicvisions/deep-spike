import tensorflow as tf
import numpy.random as rd
import utils as ff
import numpy as np
from models import SpikeGatedShallowNetwork,ShallowNetwork

# Number of inputs
n_input = 8*8
n_hidden = 192
n_output = 10

train_prop = .5
N_step = 100
learning_rate = .1

sigma = 1.
h = tf.nn.relu

########## Digit dataset
x_train,x_test,z_train,z_test = ff.load_small_digits(train_prop,n_output)

########## TF model
net_forward = ShallowNetwork(n_input,n_hidden,n_output,sigma)
net = SpikeGatedShallowNetwork(n_input, n_hidden, n_output, sigma)

########### Optimisation step
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(net.loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(N_step):
    # Compute a distorted forward pass
    sess.run(tf.assign(net_forward.W_hid, tf.sign(net.W_hid))) # tf.to_float(tf.abs(net.W_hid) > .5) *
    sess.run(tf.assign(net_forward.W_out, tf.sign(net.W_out)))
    sess.run(tf.assign(net_forward.b_hid, net.b_hid))
    sess.run(tf.assign(net_forward.b_out, net.b_out))

    gate_hidden = sess.run(tf.to_float(net_forward.y > 0), feed_dict={net_forward.x: x_train, net_forward.out_: z_train})

    # Compute gradient and optimization step with the gated network
    sess.run(train_step, feed_dict={net.x: x_train, net.gate_hidden: gate_hidden, net.out_: z_train})
    sess.run(tf.assign(net.W_hid,tf.clip_by_value(net.W_hid,-1,1)))
    sess.run(tf.assign(net.W_out,tf.clip_by_value(net.W_out,-1,1)))


print('Accuracy: ')
print(sess.run(net_forward.accuracy, feed_dict={net_forward.x: x_test, net_forward.out_: z_test}))