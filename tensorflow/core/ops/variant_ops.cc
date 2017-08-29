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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

// --------------------------------------------------------------------------

// This file contains definitions of ops operating on DT_VARIANT typed Tensors.

// --------------------------------------------------------------------------

// The ops in this section can be composed to create and manipulate lists of
// Tensors. Lists are represented by DT_VARIANT typed Tensors, and the kernels
// implementing these ops do dynamic type-checking to only allow lists of
// Tensors as inputs/outputs.

REGISTER_OP("MakeEmptyList")
    .Output("output: variant")
    .Attr("variant_type: string")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a list of length 0.
)doc");

REGISTER_OP("ExtractTensor")
    .Input("input: variant")
    .Output("output: output_type")
    .Attr("output_type: type")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .Doc(R"doc(
Returns the tensor stored in a list of length 1.
)doc");

REGISTER_OP("AppendTensorToList")
    .Input("tensor_list: variant")
    .Input("input: T")
    .Attr("variant_type: string = 'tensorflow::TensorList'")
    .Attr("T: type")
    .Output("output: variant")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Append a tensor to a list of tensors.
)doc");

REGISTER_OP("ListLength")
    .Input("input: variant")
    .Output("length: int32")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Returns the length of a list.
)doc");

REGISTER_OP("ConcatLists")
    .Input("input1: variant")
    .Input("input2: variant")
    .Output("output: variant")
    .Attr("variant_type: string = 'tensorflow::TensorList'")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Concat two lists.
)doc");

REGISTER_OP("SplitList")
    .Input("input: variant")
    .Input("at: int32")
    .Attr("variant_type: string = 'tensorflow::TensorList'")
    .Output("output1: variant")
    .Output("output2: variant")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Split a list into two lists.
)doc");

}  // end namespace tensorflow
