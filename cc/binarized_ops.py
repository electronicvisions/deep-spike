import tensorflow as tf

_binarized_module = tf.load_op_library('./cc/binarized.so')
binarized = _binarized_module.binarized


@tf.RegisterShape("Binarized")
def _binarized_shape(op):
    """Shape function for the Binarized op.

    This is the unconstrained version of Binarized, which produces an output
    with the same shape as its input.
    """
    return [op.inputs[0].get_shape()]
