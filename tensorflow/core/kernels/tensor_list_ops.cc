/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_encode_decode.h"

namespace tensorflow {

namespace internal {

struct TensorList {
  void Encode(VariantTensorData* data) const { data->set_tensors(vec); }

  bool Decode(const VariantTensorData& data) {
    vec = data.tensors();
    return true;
  }

  string TypeName() const { return "tensorflow::TensorList"; }

  std::vector<Tensor> vec;
};

TensorList MakeEmptyList() { return TensorList(); }

Status ExtractTensor(const TensorList& list, Tensor* out) {
  if (list.vec.size() != 1) {
    return errors::InvalidArgument("Input has to be a list of length 1, not ",
                                   list.vec.size());
  }
  *out = list.vec[0];
  return Status::OK();
}

TensorList AppendTensorToList(const TensorList& list, Tensor tensor) {
  TensorList out;
  out.vec.insert(out.vec.begin(), list.vec.begin(), list.vec.end());
  out.vec.push_back(tensor);
  return out;
}

TensorList ConcatLists(const TensorList& l1, const TensorList& l2) {
  TensorList out;
  out.vec.insert(out.vec.end(), l1.vec.begin(), l1.vec.end());
  out.vec.insert(out.vec.end(), l2.vec.begin(), l2.vec.end());
  return out;
}

std::pair<TensorList, TensorList> SplitList(const TensorList& list, int at) {
  if (at >= list.vec.size()) {
    return {list, TensorList()};
  }
  if (at == 0) {
    return {TensorList(), list};
  }

  TensorList out1, out2;

  out1.vec.insert(out1.vec.begin(), list.vec.begin(), list.vec.begin() + at);
  out2.vec.insert(out2.vec.begin(), list.vec.begin() + at, list.vec.end());
  return {out1, out2};
}

}  // end namespace internal

class MakeEmptyList : public OpKernel {
 public:
  explicit MakeEmptyList(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Tensor output(DT_VARIANT, TensorShape({}));
    output.flat<Variant>()(0) = internal::MakeEmptyList();
    context->set_output(0, output);
  }
};

REGISTER_KERNEL_BUILDER(Name("MakeEmptyList").Device(DEVICE_CPU),
                        MakeEmptyList);
REGISTER_KERNEL_BUILDER(
    Name("MakeEmptyList").Device(DEVICE_GPU).HostMemory("output"),
    MakeEmptyList);

class ExtractTensor : public OpKernel {
 public:
  explicit ExtractTensor(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    const Variant v = input.flat<Variant>()(0);
    const internal::TensorList* l = v.get<internal::TensorList>();

    OP_REQUIRES(context, l != nullptr,
                errors::InvalidArgument("Input has to be a list"));

    Tensor out;
    OP_REQUIRES_OK(context, internal::ExtractTensor(*l, &out));

    context->set_output(0, out);
  }
};

REGISTER_KERNEL_BUILDER(Name("ExtractTensor").Device(DEVICE_CPU),
                        ExtractTensor);
REGISTER_KERNEL_BUILDER(
    Name("ExtractTensor").Device(DEVICE_GPU).HostMemory("input"),
    ExtractTensor);

class AppendTensorToList : public OpKernel {
 public:
  explicit AppendTensorToList(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& list = context->input(0);
    const Tensor& tensor = context->input(1);

    const Variant v = list.flat<Variant>()(0);
    const internal::TensorList* l = v.get<internal::TensorList>();
    OP_REQUIRES(context, l != nullptr,
                errors::InvalidArgument("Input has to be a list"));

    Tensor output(DT_VARIANT, TensorShape({}));
    output.flat<Variant>()(0) = internal::AppendTensorToList(*l, tensor);

    context->set_output(0, output);
  }
};

REGISTER_KERNEL_BUILDER(Name("AppendTensorToList").Device(DEVICE_CPU),
                        AppendTensorToList);
REGISTER_KERNEL_BUILDER(Name("AppendTensorToList")
                        .Device(DEVICE_GPU)
                        .HostMemory("tensor_list")
                        .HostMemory("output"),
                        AppendTensorToList);

class ListLength : public OpKernel {
 public:
  explicit ListLength(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& list = context->input(0);

    const Variant v = list.flat<Variant>()(0);
    const internal::TensorList* l = v.get<internal::TensorList>();
    OP_REQUIRES(context, l != nullptr,
                errors::InvalidArgument("Input has to be a list"));

    Tensor output(DT_INT32, TensorShape({}));
    output.flat<int>()(0) = l->vec.size();

    context->set_output(0, output);
  }
};

REGISTER_KERNEL_BUILDER(Name("ListLength").Device(DEVICE_CPU), ListLength);
REGISTER_KERNEL_BUILDER(Name("ListLength")
                        .Device(DEVICE_GPU)
                        .HostMemory("input")
                        .HostMemory("length"),
                        ListLength);

class ConcatLists : public OpKernel {
 public:
  explicit ConcatLists(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input1 = context->input(0);
    const Tensor& input2 = context->input(1);

    const internal::TensorList* l1 =
        input1.flat<Variant>()(0).get<internal::TensorList>();
    const internal::TensorList* l2 =
        input2.flat<Variant>()(0).get<internal::TensorList>();

    OP_REQUIRES(context, l1 != nullptr,
                errors::InvalidArgument("Input 1 has to be a list"));
    OP_REQUIRES(context, l2 != nullptr,
                errors::InvalidArgument("Input 2 has to be a list"));

    Tensor output(DT_VARIANT, TensorShape({}));
    output.flat<Variant>()(0) = internal::ConcatLists(*l1, *l2);

    context->set_output(0, output);
  }
};

REGISTER_KERNEL_BUILDER(Name("ConcatLists").Device(DEVICE_CPU), ConcatLists);
REGISTER_KERNEL_BUILDER(Name("ConcatLists")
                        .Device(DEVICE_GPU)
                        .HostMemory("input1")
                        .HostMemory("input2")
                        .HostMemory("output"),
                        ConcatLists);

class SplitList : public OpKernel {
 public:
  explicit SplitList(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& at_tensor = context->input(1);

    int at = at_tensor.flat<int>()(0);

    const internal::TensorList* l =
        input.flat<Variant>()(0).get<internal::TensorList>();

    OP_REQUIRES(context, l != nullptr,
                errors::InvalidArgument("Input has to be a list"));
    OP_REQUIRES(context, at >= 0,
                errors::InvalidArgument("'at' should be >= 0, but found ", at));

    auto parts = internal::SplitList(*l, at);

    Tensor output1(DT_VARIANT, TensorShape({}));
    output1.flat<Variant>()(0) = parts.first;
    Tensor output2(DT_VARIANT, TensorShape({}));
    output2.flat<Variant>()(0) = parts.second;

    context->set_output(0, output1);
    context->set_output(1, output2);
  }
};

REGISTER_KERNEL_BUILDER(Name("SplitList").Device(DEVICE_CPU), SplitList)
REGISTER_KERNEL_BUILDER(Name("SplitList")
                        .Device(DEVICE_GPU)
                        .HostMemory("input")
                        .HostMemory("at")
                        .HostMemory("output1")
                        .HostMemory("output2"),
                        SplitList);

}  // end namespace tensorflow
