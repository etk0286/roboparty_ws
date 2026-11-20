import tempfile
from pathlib import Path
import json
from unittest import TestCase
from unittest.mock import patch

from dds_health import DDSHealth


class HealthTest(TestCase):
    def test_fresh_feedback_and_successful_write_are_local_evidence(self):
        with patch("dds_health.time.monotonic", return_value=10.):
            health = DDSHealth(.25, .25)
            health.received(10.)
            health.write_started()
            health.write_finished(True)
            self.assertEqual(health.snapshot()["status"], "local_io_ok")
        with patch("dds_health.time.monotonic", return_value=10.3):
            report = health.snapshot()
            self.assertIn("lowstate_missing_or_stale", report["issues"])
            self.assertIn("publishing_stalled", report["issues"])

    def test_blocked_write_visible_without_return_from_sdk(self):
        with patch("dds_health.time.monotonic", return_value=10.):
            health = DDSHealth(.25, .25)
            health.write_started()
        with patch("dds_health.time.monotonic", return_value=10.3):
            health.received(10.3)
            self.assertIn("write_blocked", health.snapshot()["issues"])

    def test_write_failure_and_callback_failure_counters_persist_after_recovery(self):
        health = DDSHealth(.25, .25)
        health.write_started()
        health.write_finished(False, "DDS error")
        health.callback_failed(ValueError("bad state"))
        self.assertIn("write_failed", health.snapshot()["issues"])
        health.write_started()
        health.write_finished(True)
        health.received(health.last_write_end)
        report = health.snapshot()
        self.assertEqual(report["status"], "local_io_ok")
        self.assertEqual(report["write_failures"], 1)
        self.assertEqual(report["callback_errors"], 1)
        self.assertEqual(report["last_callback_error"], "bad state")

    def test_thread_exit_is_saved_independently_of_command_log(self):
        from types import SimpleNamespace
        health = DDSHealth(.25, .25)
        health.control_failed("CRC crashed")
        with tempfile.TemporaryDirectory() as directory:
            # One deterministic monitor iteration, without sleeping.
            def details():
                health._stop.set()
                return {"mode": "damping"}
            health._monitor(SimpleNamespace(is_alive=lambda: False), details, Path(directory))
            report = json.loads((Path(directory) / "dds_health.jsonl").read_text())
            self.assertIn("control_thread_stopped", report["issues"])
            self.assertEqual(report["control_error"], "CRC crashed")
