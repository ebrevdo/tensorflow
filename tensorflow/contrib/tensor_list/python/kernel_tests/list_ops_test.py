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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensorflow.contrib.tensor_list.python.ops import list_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ListTest(test.TestCase):

  def testBasic(self):
    with self.test_session() as sess:
      l = list_ops.List()
      l = l.append(42.0)
      self.assertAllEqual(sess.run(l.length()), 1)
      t = l.car()
      self.assertAllEqual(sess.run(t), 42.0)

  def testConcatSplit(self):
    with self.test_session() as sess:
      l1 = functools.reduce(
          lambda l, v: l.append(v), [1., 2.], list_ops.List())
      l2 = functools.reduce(
          lambda l, v: l.append(v), [3., 4.], list_ops.List())
      l = l1.concat(l2)
      rest = l
      for i in range(4):
        v, rest = rest.split(1)
        v = v.car()
        self.assertAllEqual(sess.run(v), i + 1)
      # Empty first half
      l1, l2 = l.split(0)
      self.assertAllEqual(sess.run(l1.length()), 0)
      self.assertAllEqual(sess.run(l2.length()), 4)
      # Empty second half
      l1, l2 = l.split(4)
      self.assertAllEqual(sess.run(l1.length()), 4)
      self.assertAllEqual(sess.run(l2.length()), 0)
      l1, l2 = l.split(1004)
      self.assertAllEqual(sess.run(l1.length()), 4)
      self.assertAllEqual(sess.run(l2.length()), 0)
      with self.assertRaises(errors.InvalidArgumentError):
        l1, l2 = l.split(-1)
        sess.run(l1.car())

  def testLength(self):
    with self.test_session() as sess:
      l = list_ops.List(dtypes.int32)
      self.assertAllEqual(sess.run(l.length()), 0)
      l = functools.reduce(lambda l, v: l.append(v), [1, 2], l)
      self.assertAllEqual(sess.run(l.length()), 2)
      l1, l2 = l.split(0)
      self.assertAllEqual(sess.run(l1.length()), 0)
      self.assertAllEqual(sess.run(l2.length()), 2)

  def testFromTensor(self):
    with self.test_session() as sess:
      rest = list_ops.List.from_tensor([1, 2])
      for i in range(2):
        v, rest = rest.split(1)
        v = v.car(dtypes.int32)
        self.assertAllEqual(sess.run(v), i + 1)
      orig = np.random.rand(4, 4).astype(np.float32)
      rest = list_ops.List.from_tensor(orig)
      for i in range(4):
        v, rest = rest.split(1)
        v = v.car()
        self.assertAllEqual(sess.run(v), orig[i, :])

  def testBasicGradient(self):
    with self.test_session() as sess:
      x = variable_scope.get_variable('x', shape=())
      y = variable_scope.get_variable('y', shape=())
      l = list_ops.List()
      l = l.append(x)
      l = l.append(y)
      a, b = [e.car() for e in l.split(1)]
      z = a * b
      dx, dy = gradients_impl.gradients(z, [x, y])
      variables.global_variables_initializer().run()
      x, y, dx, dy = sess.run([a, b, dx, dy])
      self.assertAllEqual(x, dy)
      self.assertAllEqual(y, dx)

  def testGradientWithMultiUse(self):
    with self.test_session() as sess:
      x = variable_scope.get_variable('x', shape=())
      y = variable_scope.get_variable('y', shape=())
      l = list_ops.List()
      l = l.append(x)
      l = l.append(y)

      a, _ = [e.car() for e in l.split(1)]
      _, b = [e.car() for e in l.split(1)]

      z = a * b
      dx, dy = gradients_impl.gradients(z, [x, y])
      variables.global_variables_initializer().run()
      print(sess.run([a, b, dx, dy]))

  def testMap(self):
    with self.test_session() as sess:
      l = list_ops.List.from_tensor([1.0, 2.0, 3.0, 4.0])
      rest = l.map(lambda x: x + 1)
      for i in range(4):
        v, rest = rest.split(1)
        v = v.car()
        self.assertAllEqual(sess.run(v), i + 2)
      l = list_ops.List()
      l = l.map(lambda x: x)
      self.assertAllEqual(sess.run(l.length()), 0)

  def testFold(self):
    with self.test_session() as sess:
      orig = [1.0, 2.0, 3.0, 4.0]
      l = list_ops.List.from_tensor(orig)
      res = l.foldl(lambda x, y: x + y, 0.0)
      self.assertAllEqual(sess.run(res), sum(orig))
      res = l.foldr(lambda x, y: x - y, 0.0)
      # (1 - (2 - (3 - (4 - 0))))
      self.assertAllEqual(sess.run(res), -2.0)


if __name__ == "__main__":
  test.main()
