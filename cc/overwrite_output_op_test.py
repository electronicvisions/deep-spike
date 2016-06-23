import tensorflow as tf

from cc.overwrite_output_ops import overwrite_output


def testOverwriteOutput():
    sess = tf.InteractiveSession()
    external_input = [0, 1., 0., 1., 1.]
    graph_input = [-5.5, 4.4, 3.4, -2.3, 1.9]
    result = overwrite_output(graph_input, external_input)
    with sess.as_default():
        print(result.eval())


if __name__ == "__main__":
    testOverwriteOutput()
