import json
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lowcmd_logger import LowCmdLogger, estimate_acceleration


@dataclass
class Command:
    motor_cmd: list
    crc: int = 123


class RecordingTest(unittest.TestCase):
    def test_control_snapshot_and_policy_tensors_are_linked(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = LowCmdLogger({"logging": {"directory": directory}})
            motor = SimpleNamespace(q=.5, dq=0., kp=40., kd=1., tau=0., mode=1)
            command = Command([motor] * 20)
            state = SimpleNamespace(timestamp=10., qpos=np.zeros(12), qvel=np.ones(12))
            record = logger.snapshot(command, state)
            record["policy_step_id"] = 7
            motor.q = 99
            logger.submit(record)
            tensor = np.full((1, 8, 18, 32), .5, dtype=np.float32)
            for step in (7, 8):
                logger.submit({"_stream": "policy", "policy_step_id": step,
                               "encoder_frame_ids": [5] * 8,
                               "_tensors": {"encoder_input": tensor},
                               "_frames": [(5, tensor[0, 0], {"source_stamp_ns": 123})]})
            logger.close()
            rows = [json.loads(line) for line in (logger.directory / "control.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["q_target"], [.5] * 12)
            self.assertEqual(rows[0]["policy_step_id"], 7)
            policies = [json.loads(line) for line in (logger.directory / "policy.jsonl").read_text().splitlines()]
            np.testing.assert_array_equal(np.load(logger.directory / policies[0]["tensor_file"])["encoder_input"], tensor)
            self.assertEqual(len(list((logger.directory / "tensors").glob("depth_*.npy"))), 1)
            events = [json.loads(line) for line in (logger.directory / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["source_stamp_ns"], 123)
            self.assertEqual(logger.written, 3)

    def test_acceleration_uses_feedback_sample_intervals(self):
        previous = SimpleNamespace(timestamp=1., qvel=np.array([1., 2.]))
        current = SimpleNamespace(timestamp=1.02, qvel=np.array([1.2, 1.8]))
        np.testing.assert_allclose(estimate_acceleration(previous, current), [10, -10])
        self.assertIsNone(estimate_acceleration(None, current))
        self.assertIsNone(estimate_acceleration(current, current))

    def test_disk_failure_drains_and_shutdown_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = LowCmdLogger({"logging": {"directory": directory}})
            with patch("lowcmd_logger.np.savez", side_effect=OSError("disk full")):
                logger.submit({"_stream": "policy", "policy_step_id": 0, "_tensors": {"input": np.zeros(1)}})
                logger.close()
            self.assertEqual(logger.failed, 1)
            self.assertEqual(logger.written, 0)
            self.assertFalse(logger._thread.is_alive())

    def test_full_queue_counts_drops_without_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = LowCmdLogger({"logging": {"directory": directory}})
            import queue
            with patch.object(logger._queue, "put_nowait", side_effect=queue.Full):
                logger.submit({})
            logger.close()
            self.assertEqual(logger.dropped, 1)


if __name__ == "__main__":
    unittest.main()
