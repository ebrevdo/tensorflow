# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Python wrappers for Datasets and Iterators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_variant_ops


def _AddVariantType(f):
  def wrapper(*args, **kwargs):
    kwargs['variant_type'] = 'tensorflow::TensorList'
    return f(*args, **kwargs)
  return wrapper


make_empty_list = _AddVariantType(gen_variant_ops.make_empty_list)
append_tensor_to_list = _AddVariantType(gen_variant_ops.append_tensor_to_list)
concat_lists = _AddVariantType(gen_variant_ops.concat_lists)
split_list = _AddVariantType(gen_variant_ops.split_list)


class List(object):
  """Represents a list of Tensors."""

  def __init__(self, dtype=dtypes.float32, initial_list_value=None):
    self._dtype = dtype
    if initial_list_value is not None:
      self._list = initial_list_value
    else:
      self._list = make_empty_list()

  def append(self, tensor):
    tensor = ops.convert_to_tensor(tensor)
    if self._dtype != tensor.dtype:
      raise TypeError('Cannot append a tensor of type {} to a list containing '
                      'elements of type {}.'
                      .format(tensor.dtype, self._dtype))
    return List(self._dtype,
                append_tensor_to_list(self._list, tensor))

  def _extract_tensor(self, output_type=dtypes.float32):
    return gen_variant_ops.extract_tensor(self._list, output_type=output_type)

  def concat(self, other):
    return List(other._dtype,
                concat_lists(self._list, other._list))

  def split(self, at):
    l1, l2 = split_list(self._list, at)
    return List(self._dtype, l1), List(self._dtype, l2)

  def car(self, output_dtype=dtypes.float32):
    return self.split(1)[0]._extract_tensor(output_dtype)

  def length(self):
    return gen_variant_ops.list_length(self._list)

  @staticmethod
  def from_tensor(tensor):
    def cond(l, i):
      return i < array_ops.shape(tensor)[0]

    def body(l, i):
      return append_tensor_to_list(
          l, array_ops.gather(tensor, i)), i + 1

    tensor = ops.convert_to_tensor(tensor)
    l = List(tensor.dtype)._list
    i = 0
    l, i = control_flow_ops.while_loop(
        cond, body, (l, i))
    return List(tensor.dtype, l)

  def map(self, f, input_type=dtypes.float32):
    @function.Defun(input_type)
    def wrapper(*args):
      return f(*args)

    def cond(transformed, orig):
      return gen_variant_ops.list_length(orig) > 0

    def body(transformed, orig):
      first, rest = split_list(orig, 1)
      first = gen_variant_ops.extract_tensor(first, output_type=input_type)
      first = wrapper(first)
      transformed = append_tensor_to_list(transformed, first)
      return transformed, rest

    transformed, _ = control_flow_ops.while_loop(
        cond, body, (make_empty_list(), self._list))
    return List(initial_list_value=transformed)

  def foldl(self, f, z, input_type=dtypes.float32):
    @function.Defun(input_type, input_type)
    def wrapper(*args):
      return f(*args)

    def cond(result, orig):
      return gen_variant_ops.list_length(orig) > 0

    def body(result, orig):
      first, rest = split_list(orig, 1)
      first = gen_variant_ops.extract_tensor(first, output_type=input_type)
      result = wrapper(result, first)
      result.set_shape(tensor_shape.TensorShape([]))
      rest.set_shape(tensor_shape.TensorShape([]))
      return result, rest

    z = ops.convert_to_tensor(z)
    result, _ = control_flow_ops.while_loop(
        cond, body, (z, self._list))
    return result

  def foldr(self, f, z, input_type=dtypes.float32):
    @function.Defun(input_type, input_type)
    def wrapper(*args):
      return f(*args)

    def cond(result, orig):
      return gen_variant_ops.list_length(orig) > 0

    def body(result, orig):
      rest, last = split_list(
          orig,
          gen_variant_ops.list_length(orig) - 1)
      last = gen_variant_ops.extract_tensor(last, output_type=input_type)
      result = wrapper(last, result)
      result.set_shape(tensor_shape.TensorShape([]))
      rest.set_shape(tensor_shape.TensorShape([]))
      return result, rest

    z = ops.convert_to_tensor(z)
    result, _ = control_flow_ops.while_loop(
        cond, body, (z, self._list))
    return result


@ops.RegisterGradient('ExtractTensor')
def _ExtractTensorGrad(_, grad):
  l = List()
  l = l.append(grad)
  return l._list


@ops.RegisterGradient('SplitList')
def _SplitListGrad(op, grad1, grad2):
  if grad1 is None:
    l = grad2
  elif grad2 is None:
    l = grad1
  else:
    l = concat_lists(grad1, grad2)
  if l is None:
    raise ValueError('Both inputs to _SplitListGrad cannot be None')
  return l, None


@ops.RegisterGradient('AppendTensorToList')
def _AppendTensorToListGrad(_, grad):
  l = List(initial_list_value=grad)
  head, rest = l.split(l.length() - 1)
  return head._list, rest._extract_tensor()
