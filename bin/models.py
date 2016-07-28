'''
This file contains the models in tensorflow used in the experiment.

'''

import tensorflow as tf
import numpy as np
from cc import binarized_ops, binarized_grad
from cc import overwrite_output_ops, overwrite_output_grad
from lib import batch_norm



class ShallowNetwork:
    def __init__(self,n_input,n_hidden,n_output,sigma=1.,h=tf.nn.relu):
        '''
        This tensor flow gives a base line accuracy for a simple ReLu model.
        See the digit_relu.py for usage.

        :param n_input: number of inputs
        :param n_hidden: number of hidden units
        :param n_output: number of outputs
        :param sigma: standard deviation of the weights at initailization
        :param h: tensorflow activation function
        '''

        # Define the place holders, basically the data driven variables
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.z_ = tf.placeholder(tf.float32, [None, n_output])

        # Model parameters
        self.W_hid = tf.Variable(tf.random_normal([n_input, n_hidden],stddev=sigma * 1./ np.sqrt(n_input)))
        self.b_hid = tf.Variable(tf.zeros([n_hidden]))

        self.W_out = tf.Variable(tf.random_normal([n_hidden, n_output],stddev=sigma * 1./ np.sqrt(n_hidden)))
        self.b_out = tf.Variable(tf.zeros([n_output]))

        # Hidden layer
        y_act = tf.matmul(self.x, self.W_hid) + self.b_hid
        self.y = h(y_act)

        # Output softmax layer
        z_act = tf.matmul(self.y, self.W_out) + self.b_out
        self.z = tf.nn.softmax(z_act)

        # loss function
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.z_ * tf.log(self.z), reduction_indices=[1]))

        # Compute accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.z, 1), tf.argmax(self.z_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class ShallowOverwritingNetwork:
    def __init__(self,n_input,n_hidden,n_output,sigma=1.,h=tf.nn.relu):
        '''
        This tensor flow gives a base line accuracy for a simple ReLu model.
        See the digit_relu.py for usage.

        :param n_input: number of inputs
        :param n_hidden: number of hidden units
        :param n_output: number of outputs
        :param sigma: standard deviation of the weights at initailization
        :param h: tensorflow activation function
        '''

        overwrite = overwrite_output_ops.overwrite_output

        # Define the place holders, basically the data driven variables
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.forward_y = tf.placeholder(tf.float32, [None, n_hidden])
        self.forward_z = tf.placeholder(tf.float32, [None, n_output])
        self.z_ = tf.placeholder(tf.float32, [None, n_output])

        # Model parameters
        self.W_hid = tf.Variable(tf.random_normal([n_input, n_hidden],stddev=sigma * 1./ np.sqrt(n_input)))
        self.b_hid = tf.Variable(tf.zeros([n_hidden]))

        self.W_out = tf.Variable(tf.random_normal([n_hidden, n_output],stddev=sigma * 1./ np.sqrt(n_hidden)))
        self.b_out = tf.Variable(tf.zeros([n_output]))

        # Hidden layers
        y_act = tf.matmul(self.x, self.W_hid) + self.b_hid
        y_out = h(y_act)
        self.y = overwrite(y_out, self.forward_y)

        # Output softmax layer
        z_act = tf.matmul(self.y, self.W_out) + self.b_out
        z_out = tf.nn.softmax(z_act)
        self.z = overwrite(z_out, self.forward_z)

        # loss function
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.z_ * tf.log(self.z), reduction_indices=[1]))

        # Compute accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.z, 1), tf.argmax(self.z_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))