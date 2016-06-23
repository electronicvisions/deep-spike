import tensorflow as tf

from binarized_ops import binarized


def testBinarized():
    sess = tf.InteractiveSession()
    result = binarized([-5.5, 4.4, 3.4, -2.3, 1.9])
    with sess.as_default():
        print(result.eval())


if __name__ == "__main__":
    testBinarized()
