from sklearn.datasets import load_digits
from sklearn import cross_validation
from sklearn import preprocessing

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main():
    digits = load_digits()
    x_train, x_test, y_train_, y_test_ = cross_validation.train_test_split(digits.data, digits.target, test_size=0.2,
                                                                           random_state=0)

    lb = preprocessing.LabelBinarizer()
    lb.fit(digits.target)
    y_train = lb.transform(y_train_)
    y_test = lb.transform(y_test_)

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 64])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    w_1 = weight_variable([64, 32])
    b_1 = bias_variable([32])
    h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

    w_2 = weight_variable([32, 10])
    b_2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess.run(tf.initialize_all_variables())
    for i in range(1000):
        train_step.run(feed_dict={x: x_train, y_: y_train})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval(feed_dict={x: x_test, y_: y_test}))


if __name__ == '__main__':
    main()
