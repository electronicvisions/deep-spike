'''
This file contains the models in tensorflow used in the experiment.

'''

import tensorflow as tf
import numpy as np

class ShallowNetwork:
    def __init__(self,n_input,n_hidden,n_output,sigma=1.,h=tf.nn.relu):
        '''
        This tensor flow gives a base line accuracy for a simple ReLu model.
        See the script_base_line_relu.py for usage.

        :param n_input: number of inputs
        :param n_hidden: number of hidden units
        :param n_output: number of outputs
        :param sigma: standard deviation of the weights at initailization
        :param h: tensorflow activation function
        '''
        # Define the place holders, basically the data driven variables
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.out_ = tf.placeholder(tf.float32, [None, n_output])

        # Model parameters
        self.W_hid = tf.Variable(tf.random_normal([n_input, n_hidden],stddev=sigma * 1./ np.sqrt(n_input)))
        self.b_hid = tf.Variable(tf.zeros([n_hidden]))

        self.W_out = tf.Variable(tf.random_normal([n_hidden, n_output],stddev=sigma * 1./ np.sqrt(n_hidden)))
        self.b_out = tf.Variable(tf.zeros([n_output]))

        # build variables
        self.y = h(tf.matmul(self.x, self.W_hid) + self.b_hid)
        self.z = tf.matmul(self.y, self.W_out) + self.b_out

        # loss function
        self.out = tf.nn.softmax(self.z)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.out_ * tf.log(self.out), reduction_indices=[1]))

        self.correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.out_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def all_variables(self):
        return [self.W_hid,self.b_hid,self.W_out,self.b_out]



class SpikeGatedShallowNetwork:
    def __init__(self,n_time,dt,tau_m,tau_reset,n_input,n_hidden,n_output,sigma=1.,b0=1):
        '''
        This tensorflow model is supposed to represent any feedforward network of integrate and fire neuron.
        The corresponding spiking model should have a membrane time constant of tau_m and a reset modeled
        with a self connection decreasing the voltage from threshold (0) to reset (-b) at each spike.

        The attributes spike_x,spike_y, and spike_z are the place holders that will be given the spike trains
        measured in NEST or on the chip.

        :param n_time: number of time steps
        :param dt: time step in arbitrary unit (ex. ms)
        :param tau_m: membrane time constant in a. unit
        :param tau_reset: reset time constant in a. unit
        :param n_input: number of input neurons
        :param n_hidden: number of hidden neurons
        :param n_output: number of output neurons
        :param sigma: variance of the random weights at initialization
        :param b0: size of the bias at initialization
        '''


        # Define the place holders, the gate are value taken from the data
        self.spike_x = tf.sparse_placeholder(tf.float32, [None, 1, n_time, n_input])
        self.spike_y = tf.sparse_placeholder(tf.float32, [None, 1, n_time, n_hidden])
        self.spike_z = tf.sparse_placeholder(tf.float32, [None, 1, n_time, n_output])
        self.out_ = tf.placeholder(tf.float32, [None, n_output])

        # We will need to filter with some PSP
        tt = np.arange(n_time) * dt
        filter_m = np.exp(-tt/tau_m)
        filter_m = filter_m.reshape(1,n_time)

        filter_reset = np.exp(-tt/tau_reset)
        filter_reset = filter_reset.reshape(1,n_time)

        # Model parameters
        self.W_hid = tf.Variable(tf.random_normal([n_input, n_hidden],stddev=sigma * 1./n_input))
        self.b_hid = tf.Variable(tf.ones([n_hidden]))

        self.W_out = tf.Variable(tf.random_normal([n_hidden, n_output],stddev=sigma * 1./n_hidden))
        self.b_out = tf.Variable(tf.ones([n_output]))

        # Build the model of IF neuron with exponential PSP and Exponential Reset
        psp_x = tf.nn.conv2d(self.spike_x,filter_m)
        reset_y = tf.nn.conv2d(self.spike_y,filter_reset)

        ## TODO: Debug this, the product for reset should be done with somekind of outter product
        self.V_y = tf.matmul(psp_x,self.W_hid) - (1 + reset_y) * self.b_hid

        # This is the funny part, by defining this the derivative of the error wrt W_hid and W_out are valid
        # In particular this is equal to spike_y when V_y is 0 at spike time
        # But it is still differentiable wrt W_hid and has the right derivative
        self.differentiable_spike_y = self.spike_y * (self.V_y +1)
        self.psp_y = tf.nn.conv2d(self.differentiable_spike_y,filter_m)
        self.reset_z = tf.nn.conv2d(self.spike_z,filter_reset)

        ## TODO: Debug this, the product for reset should be done with someking of outter product
        self.V_z = tf.matmul(self.psp_y, self.W_out) - (1 + self.reset_z) * self.b_out
        self.differentiable_spike_z = self.spike_z * (self.V_z +1)

        self.z = tf.reduce_sum(self.differentiable_spike_z,reduction_indices=(1,2))

        # loss function
        self.out = tf.nn.softmax(self.z)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.out_ * tf.log(self.out), reduction_indices=[1]))