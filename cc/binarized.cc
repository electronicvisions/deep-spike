#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


REGISTER_OP("Binarized")
    .Input("to_binarize: float")
    .Output("binarized: float");

using namespace tensorflow;

class BinarizedOp : public OpKernel {
 public:
  explicit BinarizedOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();

    const int N = input.size();
    for (int i = 0; i < N; i++) {
        if(input(i) >= 0){
            output(i) = 1;
        } else {
            output(i) = 0;
        }
    }
  }
};

REGISTER_KERNEL_BUILDER(
        Name("Binarized")
        .Device(DEVICE_CPU),
        BinarizedOp);

