"""Check camera-frame history without loading ROS or the motor SDK."""
import ast
from collections import deque
from pathlib import Path
import time
import unittest
from types import SimpleNamespace

import numpy as np


source = Path(__file__).resolve().parents[1] / "isaaclab_parkour_deploy.py"
tree = ast.parse(source.read_text())
body = [node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ParkourPolicy"]
namespace = dict(np=np, deque=deque, time=time, NUM_MOTORS=12,
                 PROPRIO_KEYS=(), projected_gravity_from_wxyz=lambda q: np.array([0., 0., -1.], dtype=np.float32))
exec(compile("from __future__ import annotations\n" +
             ast.unparse(ast.Module(body=body, type_ignores=[])),
             str(source), "exec"), namespace)


class DepthHistoryTest(unittest.TestCase):
    def setUp(self):
        self.policy = namespace["ParkourPolicy"].__new__(namespace["ParkourPolicy"])
        self.policy.history_length = 8
        self.policy.depth_history_length = 37
        self.policy.depth_indices = np.array([1, 6, 11, 16, 21, 26, 31, 36])
        self.policy.encoder_height, self.policy.encoder_width = 18, 32
        self.policy.reset()

    def frame(self, value):
        return np.full((18, 32), value, dtype=np.float32)

    def test_repeated_inference_does_not_advance_history(self):
        first = self.policy._update_depth(self.frame(1), 1)
        for _ in range(10):
            actual = self.policy._update_depth(self.frame(1), 1)
            np.testing.assert_array_equal(actual, first)
        self.assertEqual(len(self.policy.depth_history), 1)
        self.assertEqual(actual.shape, (1, 8, 18, 32))
        # Identical pixels in a new camera frame still count as a new frame.
        self.policy._update_depth(self.frame(1), 2)
        self.assertEqual(len(self.policy.depth_history), 2)

    def test_skipped_camera_ids_are_not_filled(self):
        for index in range(40):
            actual = self.policy._update_depth(self.frame(index), index * 3)
        self.assertEqual(len(self.policy.depth_history), 37)
        np.testing.assert_array_equal(actual[0, :, 0, 0],
                                      [4, 9, 14, 19, 24, 29, 34, 39])

    def test_trace_preserves_actual_model_inputs_and_unclipped_action(self):
        p = self.policy
        p.record_inputs = True
        p.command_limits = np.ones(3)
        p.command_scale = np.ones(3)
        p.ang_vel_scale = p.joint_vel_scale = 1.
        p.default_angles = np.zeros(12)
        p.sdk_to_policy = np.arange(12)
        p.action_clip = .25
        p.encoder_input, p.actor_input = "depth", "obs"
        p._update_proprio = lambda values: np.arange(480, dtype=np.float32)
        seen = {}
        def encode(_, feed):
            seen["encoder"] = feed["depth"].copy()
            return [np.ones((1, 8), dtype=np.float32)]
        def act(_, feed):
            seen["actor"] = feed["obs"].copy()
            return [np.full((1, 12), 2., dtype=np.float32)]
        p.encoder, p.actor = SimpleNamespace(run=encode), SimpleNamespace(run=act)
        action = p.infer(np.zeros(12), np.zeros(12), np.zeros(3),
                         np.array([1, 0, 0, 0]), np.array([2., 0., 0.]),
                         self.frame(.7), 42, {"source_stamp_ns": 123})
        trace = p.last_trace
        np.testing.assert_array_equal(trace["_tensors"]["encoder_input"], seen["encoder"])
        np.testing.assert_array_equal(trace["_tensors"]["actor_input"], seen["actor"])
        np.testing.assert_array_equal(trace["_tensors"]["raw_action"], 2.)
        np.testing.assert_array_equal(action, .25)
        self.assertEqual(trace["encoder_frame_ids"], [42] * 8)
        self.assertEqual(trace["command"], [1., 0., 0.])
        p.reset()
        np.testing.assert_array_equal(trace["_tensors"]["raw_action"], 2.)

    def test_reset_accepts_same_latest_frame(self):
        self.policy._update_depth(self.frame(1), 5)
        self.policy.reset()
        actual = self.policy._update_depth(self.frame(2), 5)
        np.testing.assert_array_equal(actual, np.full((1, 8, 18, 32), 2))
        self.assertEqual(len(self.policy.depth_history), 1)


if __name__ == "__main__":
    unittest.main()
