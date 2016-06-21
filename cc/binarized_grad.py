from tensorflow.python.framework import ops
import tensorflow as tf


@ops.RegisterGradient("Binarized")
def _binarized_grad(op, grad):
    """The gradients for `binarized`.

    Args:
      op: The `binarized` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `binarized` op.

    Returns:
      Gradients with respect to the input of `binarized`.
    """
    to_binarize = op.inputs[0]
    to_binarize_grad = grad * tf.maximum(0., 1 - tf.abs(to_binarize))
    return [to_binarize_grad]  # List of one Tensor, since we have one input
