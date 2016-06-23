from tensorflow.python.framework import ops


@ops.RegisterGradient("OverwriteOutput")
def _dummy_grad(op, grad):
    """The gradients for `overwrite_output`.
    This is a trick to do forward pass and backward pass with the same model but with different sources of neuron outputs.

    Args:
      op: The `overwrite_output` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `overwrite_output` op.

    Returns:
      Gradients with respect to the input of `overwrite_output`.
    """
    return [grad, None]  # List of one Tensor, since we have one input
