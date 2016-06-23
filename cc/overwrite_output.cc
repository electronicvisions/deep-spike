#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


REGISTER_OP("OverwriteOutput")
    .Input("graph_input: float")
    .Input("external_input: float")
    .Output("external_input_copy: float");

using namespace tensorflow;

class DummyOp : public OpKernel {
 public:
  explicit DummyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(1);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();

    const int N = input.size();
    for (int i = 0; i < N; i++) {
        output(i) = input(i);
    }
  }
};

REGISTER_KERNEL_BUILDER(
        Name("OverwriteOutput")
        .Device(DEVICE_CPU),
        DummyOp);

