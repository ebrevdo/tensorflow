# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""critical section tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.contrib.framework.python.ops import notification_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test

# TODO(ebrevdo): Re-enable once Notification is in core.
# from tensorflow.python.training import saver as saver_lib


class NotificationTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testCreateNotification(self):
    n = notification_ops.Notification(shared_name="notification")
    notifier = n.notifier()
    self.evaluate(notifier)

  @test_util.run_in_graph_and_eager_modes()
  def testNotificationWaitTimesOut(self):
    n = notification_ops.Notification(shared_name="notification")
    for timeout in (0, 1, 5, 10, 1e2, 1e3, 1e6):
      notified = n.wait(timeout_in_us=int(timeout))
      self.assertFalse(self.evaluate(notified))

  def testNotificationWaitTimesOutInParallelThreads(self):
    n = notification_ops.Notification(shared_name="notification")
    w = n.wait(10)
    s = session.Session()
    h = lambda s_: self.assertFalse(s_.run(w))
    ts = [threading.Thread(target=h, args=(s,)) for _ in range(100)]
    for t in ts:
      t.start()
    for t in ts:
      t.join()

  def testNotificationWaitCanBeCancelled(self):
    n = notification_ops.Notification(shared_name="notification")
    options = config_pb2.RunOptions(timeout_in_ms=100)
    for timeout in (None, int(1e6)):
      notified = n.wait(timeout_in_us=timeout)
      with self.test_session() as sess:
        with self.assertRaisesOpError("Timed out"):
          sess.run(notified, options=options)

    options = config_pb2.RunOptions(timeout_in_ms=1000)
    for timeout in (1, 10, 100):
      notified = n.wait(timeout_in_us=timeout)
      with self.test_session() as sess:
        # Run to completion.
        self.assertFalse(sess.run(notified, options=options))

  def testNotificationNotifier(self):
    n = notification_ops.Notification()
    n_wait = notification_ops.Notification()
    flag = resource_variable_ops.ResourceVariable(True, name="v")
    t1 = logging_ops.timestamp()
    with ops.control_dependencies([n.wait()]):
      with ops.control_dependencies([flag.assign(False)]):
        diff = logging_ops.timestamp() - t1

    # Create a while_loop that runs n.notifier() approx every 10ms.
    def _notify_cond(c):
      with ops.control_dependencies([c]):
        return flag.value()

    def _notify_body(c):
      with ops.control_dependencies([c, n_wait.wait(int(1e4))]):  # 10ms
        with ops.control_dependencies([n.notifier()]):
          return c + 1

    # Count the number of times n.notifier() was called.
    notifications = control_flow_ops.while_loop(
        _notify_cond, _notify_body, [0], parallel_iterations=1)

    with self.test_session() as sess:
      sess.run(flag.initializer)
      diff_v, notifications_v = sess.run((diff, notifications))

    # Waited between 10ms and 1s.
    self.assertGreaterEqual(diff_v, 0.01)
    self.assertLess(diff_v, 1.0)

    # Sent between 1 and 5 notifications (5 to be on the safe side).
    self.assertGreaterEqual(notifications_v, 1)
    self.assertLess(notifications_v, 5)

  @test_util.run_in_graph_and_eager_modes()
  def testNotificationDoNotResetImmediately(self):
    n = notification_ops.Notification()
    notifier = n.notifier(immediately_reset=False)
    with ops.control_dependencies([notifier]):
      notified = n.wait(timeout_in_us=0)
    self.assertTrue(self.evaluate(notified))
    # Notification has been set.  Re-execute it.
    notified_again = n.wait(timeout_in_us=1)
    self.assertTrue(self.evaluate(notified_again))
    reset = n.resetter()
    with ops.control_dependencies([reset]):
      not_notified = n.wait(timeout_in_us=1)
      self.assertFalse(self.evaluate(not_notified))

  def testNotificationManySequentialNotifications(self):
    n = notification_ops.Notification()
    n_wait = notification_ops.Notification()
    flag = resource_variable_ops.ResourceVariable(True, name="v")

    t1 = logging_ops.timestamp()
    waiter = []
    for _ in range(20):
      with ops.control_dependencies(waiter):
        waiter = [n.wait()]
    with ops.control_dependencies(waiter):
      with ops.control_dependencies([flag.assign(False)]):
        diff = logging_ops.timestamp() - t1

    # Create a while_loop that runs n.notifier() approx every 10ms.
    def _notify_cond(_):
      return flag.value()

    def _notify_body(c):
      with ops.control_dependencies([c, n_wait.wait(int(1e4))]):
        with ops.control_dependencies([n.notifier()]):
          return c + 1

    # Count the number of times n.notifier() was called.
    notifications = control_flow_ops.while_loop(
        _notify_cond, _notify_body, [0], parallel_iterations=1)

    with self.test_session() as sess:
      sess.run(flag.initializer)
      _, notifications_v = sess.run((diff, notifications))

    # Sent between 20 and 40 notifications (40 to be on the safe side).
    self.assertGreaterEqual(notifications_v, 20)
    self.assertLess(notifications_v, 40)

  def testNotificationManyParallelNotifications(self):
    n = notification_ops.Notification()
    n_wait = notification_ops.Notification()
    flag = resource_variable_ops.ResourceVariable(True, name="v")

    t1 = logging_ops.timestamp()
    waiters = [n.wait() for _ in range(100)]
    with ops.control_dependencies(waiters):
      with ops.control_dependencies([flag.assign(False)]):
        diff = logging_ops.timestamp() - t1

    # Create a while_loop that runs n.notifier() approx every 10ms.
    def _notify_cond(_):
      return flag.value()

    def _notify_body(c):
      with ops.control_dependencies([c, n_wait.wait(int(1e4))]):
        with ops.control_dependencies([n.notifier()]):
          return c + 1

    # Count the number of times n.notifier() was called.
    notifications = control_flow_ops.while_loop(
        _notify_cond, _notify_body, [0], parallel_iterations=1)

    with self.test_session() as sess:
      sess.run(flag.initializer)
      _, notifications_v = sess.run((diff, notifications))

    # Sent at least 1 notification.
    self.assertGreaterEqual(notifications_v, 1)


if __name__ == "__main__":
  test.main()
