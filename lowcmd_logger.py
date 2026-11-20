"""Bounded, asynchronous session recording; no ROS or motor SDK dependency."""
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np


def estimate_acceleration(previous, current):
    if previous is None or current.timestamp <= previous.timestamp:
        return None
    return (current.qvel.astype(np.float64) - previous.qvel) / (current.timestamp - previous.timestamp)


def state_record(state):
    if state is None:
        return None
    result = {"state_id": getattr(state, "state_id", None),
              "receive_monotonic_s": state.timestamp}
    for name in ("qpos", "qvel", "ang_vel", "quaternion", "tau_est", "foot_force"):
        value = getattr(state, name, None)
        result[name] = None if value is None else value.tolist()
    return result


class LowCmdLogger:
    def __init__(self, config, runtime_metadata=None):
        settings = config.get("logging", {})
        capacity = int(settings.get("queue_size", 1000))
        if capacity <= 0:
            raise ValueError("logging.queue_size must be positive")
        self.directory = Path(settings.get("directory", "record/lowcmd")).expanduser()
        self.directory /= datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "tensors").mkdir()
        metadata = {"schema_version": 2, "config": config, "runtime": runtime_metadata,
                    "clock_anchor": {"unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()},
                    "clocks": "Host monotonic for ordering; ROS header stamp is a separate, uncalibrated source clock. Receive pairs do not imply clock synchronization.",
                    "units": {"qpos": "rad", "qvel": "rad/s", "ang_vel": "rad/s", "quaternion": "wxyz", "tau_est": "Nm (SDK estimate)", "foot_force": "SDK foot_force raw measured channel; calibration/units unverified"},
                    "depth": "Normalized float32; invalid/nonpositive -> max_m; history oldest to newest, first frame repeated on reset.",
                    "replay": "Stateless ONNX calls; policy NPZ contains exact encoder_input and actor_input including proprio history. Queue drops make sessions incomplete; consult events/summary."}
        (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        self._queue = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._closed = False
        self.submitted = self.dropped = self.written = self.failed = 0
        self._thread = threading.Thread(target=self._write_loop, name="session-log", daemon=True)
        self._thread.start()
        print(f"Session recording: {self.directory.resolve()}")

    @staticmethod
    def snapshot(command, state, acceleration=None, depth=None):
        motors = command.motor_cmd[:12]
        return {"_stream": "control", "state": state_record(state),
                **{key: [getattr(m, field) for m in motors] for key, field in
                   (("q_target", "q"), ("dq_target", "dq"), ("tau_cmd", "tau"),
                    ("kp", "kp"), ("kd", "kd"), ("motor_mode", "mode"))}}

    def event(self, kind, **details):
        return self.submit({"_stream": "events", "event": kind,
                            "monotonic_ns": time.monotonic_ns(), **details})

    def submit(self, record):
        with self._lock:
            if self._closed:
                return False
            record["sequence"] = self.submitted
            self.submitted += 1
            try:
                self._queue.put_nowait(record)
                return True
            except queue.Full:
                self.dropped += 1
                return False

    def _write_loop(self):
        streams = {}
        saved_frames = set()
        reported_drops = 0
        try:
            for name in ("control", "policy", "events"):
                streams[name] = (self.directory / f"{name}.jsonl").open("w", buffering=1)
            while True:
                record = self._queue.get()
                if record is None:
                    break
                if self.dropped != reported_drops:
                    reported_drops = self.dropped
                    streams["events"].write(json.dumps({"event": "log_dropped", "total": reported_drops, "monotonic_ns": time.monotonic_ns()}) + "\n")
                destination = record.pop("_stream", "control")
                tensors = record.pop("_tensors", None)
                if tensors is not None:
                    path = f"tensors/policy_{record['policy_step_id']:08d}.npz"
                    np.savez(self.directory / path, **tensors)
                    record["tensor_file"] = path
                for frame_id, frame, metadata in record.pop("_frames", []):
                    if frame_id not in saved_frames:
                        path = f"tensors/depth_{frame_id:08d}.npy"
                        np.save(self.directory / path, frame, allow_pickle=False)
                        streams["events"].write(json.dumps({"event": "depth_frame", "frame_id": frame_id, "tensor_file": path, **metadata}) + "\n")
                        saved_frames.add(frame_id)
                error = record.pop("write_error", None)
                if error:
                    streams["events"].write(json.dumps({"event": "write_error", "control_sequence": record["sequence"], "monotonic_ns": record.get("publish_end_monotonic_ns"), "error": error}) + "\n")
                streams[destination].write(json.dumps(record) + "\n")
                self.written += 1
        except Exception as exc:
            self.failed += 1
            print(f"ERROR: Session recording failed: {exc}", file=sys.stderr)
            try:
                streams["events"].write(json.dumps({"event": "logging_failure", "error": str(exc), "monotonic_ns": time.monotonic_ns()}) + "\n")
            except Exception:
                pass
            while self._queue.get() is not None:
                self.failed += 1
        finally:
            for stream in streams.values():
                stream.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join()
        summary = dict(submitted=self.submitted, written=self.written, dropped=self.dropped, failed=self.failed)
        try:
            (self.directory / "summary.json").write_text(json.dumps(summary, indent=2))
        except OSError as exc:
            print(f"ERROR: Cannot save log summary: {exc}", file=sys.stderr)
        print(f"Session recording summary: {summary}")
