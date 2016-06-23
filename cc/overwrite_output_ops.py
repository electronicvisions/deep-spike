import tensorflow as tf

_overwrite_output = tf.load_op_library('./cc/overwrite_output.so')
overwrite_output = _overwrite_output.overwrite_output


@tf.RegisterShape("OverwriteOutput")
def _dummy_shape(op):
    """Shape function for the Dummy op.

    This is the unconstrained version of Dummy, which produces an output
    with the same shape as its input spikes.
    """
    return [op.inputs[0].get_shape()]
