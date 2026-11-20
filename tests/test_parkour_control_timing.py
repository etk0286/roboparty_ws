"""Exercise the actual controller methods without importing ROS or motor SDKs."""
import ast
from dataclasses import make_dataclass
import json
import math
from pathlib import Path
import threading
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest

import numpy as np
from lowcmd_logger import LowCmdLogger, state_record
from dds_health import DDSHealth

source = Path(__file__).resolve().parents[1] / 'isaaclab_parkour_deploy.py'
tree = ast.parse(source.read_text())
names = {'ParkourDeployController'}
body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in names]
namespace = dict(np=np, threading=threading, time=time, math=math, traceback=traceback, NUM_MOTORS=12, state_record=state_record)
exec(compile('from __future__ import annotations\n' + ast.unparse(ast.Module(body=body, type_ignores=[])), str(source), 'exec'), namespace)
Controller = namespace['ParkourDeployController']


class TimingTest(unittest.TestCase):
    def test_false_write_switches_to_damping_and_keeps_sending(self):
        c = self.make_controller()
        calls = []
        def write(cmd):
            calls.append(cmd.target)
            return len(calls) != 1
        c.publisher.Write = write
        c.start()
        time.sleep(.1)
        self.assertEqual(c.mode, 'damping')
        self.assertGreaterEqual(len(calls), 3)
        self.assertTrue(all(target == 'damping' for target in calls[1:]))
        self.assertEqual(c.dds_health.write_failures, 1)
        self.assertTrue(c._thread.is_alive())

    def test_write_exception_is_logged_and_does_not_kill_sender(self):
        c = self.make_controller()
        calls = []
        def write(cmd):
            calls.append(cmd.target)
            if len(calls) == 1:
                raise RuntimeError('DDS transport failed')
            return True
        c.publisher.Write = write
        c.start()
        time.sleep(.1)
        self.assertEqual(c.mode, 'damping')
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(c.dds_health.write_failures, 1)
        self.assertTrue(c._thread.is_alive())

    def test_enabled_logger_records_each_write_while_inference_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.make_controller()
            c.command_logger = LowCmdLogger({"logging": {"directory": directory}})
            c._state_lock = threading.Lock()
            c._state = c._state_snapshot()
            c._state.timestamp = time.monotonic()
            c._joint_acceleration = None
            c.low_cmd.motor_cmd = [SimpleNamespace(q=0., dq=0., tau=0., kp=40., kd=1., mode=1) for _ in range(12)]
            try:
                c.start()
                self.assertTrue(c.entered.wait(1))
                time.sleep(.24)
            finally:
                c._running.clear()
                c.release.set()
                c._inference_ready.set()
                c._thread.join(2)
                c._inference_thread.join(2)
                c.dds_health.close()
                c.command_logger.close()
            rows = [json.loads(line) for line in (c.command_logger.directory / "control.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), len(c.sent))
            self.assertGreaterEqual(len(rows), 9)
            self.assertTrue(all(row["q_target"] == [0.] * 12 for row in rows))
            self.assertTrue(all(row["timestamp_unix_ns"] > 0 for row in rows))
            self.assertTrue(all(row["publish_end_monotonic_ns"] >= row["publish_monotonic_ns"] for row in rows))

    def make_controller(self):
        c = Controller.__new__(Controller)
        c.command_logger = None
        c.dds_health = DDSHealth(.25, .25)
        c._mode_lock = threading.RLock()
        c._policy_lock = threading.Lock()
        c._running = threading.Event()
        c._inference_ready = threading.Event()
        c._inference_busy = False
        c._inference_request = None
        c._mode_generation = 0
        c._policy_step_counter = 0
        c._target_policy_step_id = None
        c._published_policy_step_id = None
        c._published_mode = "move"
        c._counter = 0
        c.decimation = 1
        c._thread = c._inference_thread = None
        c.dt = .02
        c.mode = 'move'
        c.fault_reason = ''
        c.command = c.target_command = np.zeros(3)
        c.target_dof_pos = c.default_angles = np.zeros(12)
        c.action_scale = np.ones(12)
        c.policy_to_sdk = np.arange(12)
        c.target_lower, c.target_upper = -np.ones(12), np.ones(12)
        c._last_limit_warning_time = 0
        c.depth_stale_timeout = .5
        c.depth_age = 0
        c.depth_node = SimpleNamespace(latest=lambda: (np.zeros(1), c.depth_age, 1), latest_record=lambda: (np.zeros(1), c.depth_age, 1, {}))
        c._state_snapshot = lambda: SimpleNamespace(qpos=np.zeros(12), qvel=np.zeros(12), ang_vel=np.zeros(3), quaternion=np.array([1, 0, 0, 0]), timestamp=time.monotonic())
        c._validate_state_safety = lambda *a, **kw: None
        c.low_cmd = SimpleNamespace(crc=0, target=None)
        c._apply_position = lambda target: setattr(c.low_cmd, 'target', tuple(target))
        c._apply_damping = lambda q: setattr(c.low_cmd, 'target', 'damping')
        c.crc = SimpleNamespace(Crc=lambda cmd: 0)
        c.sent = []
        c.publisher = SimpleNamespace(Write=lambda cmd: c.sent.append((time.monotonic(), cmd.target)))
        c.entered, c.release = threading.Event(), threading.Event()
        def infer(*args):
            c.entered.set()
            c.release.wait(2)
            return np.ones(12) * .5
        c.policy = SimpleNamespace(infer=infer, reset=lambda: None, last_trace={})
        def cleanup():
            c._running.clear()
            c.release.set()
            c._inference_ready.set()
            for thread in (c._thread, c._inference_thread):
                if thread:
                    thread.join(2)
            c.dds_health.close()
        self.addCleanup(cleanup)
        return c

    def test_blocked_inference_repeats_at_50hz_then_updates(self):
        c = self.make_controller()
        c.start()
        self.assertTrue(c.entered.wait(1))
        time.sleep(.24)
        sent = list(c.sent)
        self.assertGreaterEqual(len(sent), 9)
        self.assertTrue(all(target == (0.,) * 12 for _, target in sent))
        rate = (len(sent)-1)/(sent[-1][0]-sent[0][0])
        self.assertAlmostEqual(rate, 50, delta=5)
        c.release.set()
        time.sleep(.08)
        self.assertEqual(c.sent[-1][1], (.5,) * 12)
        self.assertIsNotNone(c._published_policy_step_id)
        print(f'Mocked LowCmd rate during blocked inference: {rate:.1f} Hz')

    def test_damping_discards_inflight_action(self):
        c = self.make_controller()
        c.start()
        self.assertTrue(c.entered.wait(1))
        c.set_damping('test')
        time.sleep(.06)
        self.assertEqual(c.sent[-1][1], 'damping')
        c.release.set()
        time.sleep(.06)
        np.testing.assert_array_equal(c.target_dof_pos, np.zeros(12))
        self.assertEqual(c.sent[-1][1], 'damping')

    def test_stale_depth_stops_replay(self):
        c = self.make_controller()
        c.start()
        self.assertTrue(c.entered.wait(1))
        c.depth_age = 1.
        time.sleep(.06)
        self.assertEqual(c.mode, 'damping')
        self.assertIn('Depth stale', c.fault_reason)
        self.assertEqual(c.sent[-1][1], 'damping')


if __name__ == '__main__':
    unittest.main()
