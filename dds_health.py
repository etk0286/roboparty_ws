"""Local DDS workflow diagnostics; successful writes are not robot ACKs."""

from collections import deque
import json
import sys
import threading
import time


class DDSHealth:
    def __init__(self, stale_timeout, publish_timeout):
        self.stale_timeout = stale_timeout
        self.publish_timeout = publish_timeout
        self._lock = threading.Lock()
        self.started = time.monotonic()
        self.rx_times = deque(maxlen=10000)
        self.tx_times = deque(maxlen=10000)
        self.rx_count = self.tx_count = self.write_failures = 0
        self.callback_errors = 0
        self.last_callback_error = None
        self.last_rx = self.last_write_start = self.last_write_end = None
        self.max_rx_gap_s = self.max_tx_gap_s = 0.0
        self.last_write_result = self.last_write_error = self.control_error = None
        self._stop = threading.Event()
        self._thread = None

    def received(self, timestamp):
        with self._lock:
            if self.last_rx is not None:
                self.max_rx_gap_s = max(self.max_rx_gap_s, timestamp - self.last_rx)
            self.last_rx = timestamp
            self.rx_count += 1
            self.rx_times.append(timestamp)

    def callback_failed(self, error):
        with self._lock:
            self.callback_errors += 1
            self.last_callback_error = str(error)

    def write_started(self):
        with self._lock:
            now = time.monotonic()
            if self.last_write_start is not None:
                self.max_tx_gap_s = max(self.max_tx_gap_s, now - self.last_write_start)
            self.last_write_start = now

    def write_finished(self, result, error=None):
        with self._lock:
            now = time.monotonic()
            self.last_write_end = now
            self.tx_times.append(now)
            self.tx_count += 1
            self.last_write_result = None if result is None else bool(result)
            self.last_write_error = error
            if error is not None or self.last_write_result is False:
                self.write_failures += 1

    def control_failed(self, error):
        with self._lock:
            self.control_error = str(error)

    def snapshot(self, control_alive=True):
        with self._lock:
            now = time.monotonic()
            window = max(min(now - self.started, 1.0), 1e-6)
            rx_age = None if self.last_rx is None else now - self.last_rx
            tx_age = None if self.last_write_end is None else now - self.last_write_end
            pending = self.last_write_start is not None and (
                self.last_write_end is None or self.last_write_start > self.last_write_end
            )
            issues = []
            if not control_alive or self.control_error:
                issues.append("control_thread_stopped")
            if rx_age is None or rx_age > self.stale_timeout:
                issues.append("lowstate_missing_or_stale")
            if (pending and now - self.last_write_start > self.publish_timeout):
                issues.append("write_blocked")
            if (tx_age is None and now - self.started > self.publish_timeout) or (
                tx_age is not None and tx_age > self.publish_timeout
            ):
                issues.append("publishing_stalled")
            if self.last_write_result is False or self.last_write_error:
                issues.append("write_failed")
            return {
                "timestamp_unix_ns": time.time_ns(),
                "monotonic_s": now,
                "status": "fault" if issues else (
                    "local_io_ok" if self.last_write_result is True else "write_unconfirmed"
                ),
                "issues": issues,
                "control_thread_alive": control_alive,
                "lowstate_hz": sum(t > now - 1.0 for t in self.rx_times) / window,
                "write_attempt_hz": sum(t > now - 1.0 for t in self.tx_times) / window,
                "lowstate_age_s": rx_age,
                "last_write_age_s": tx_age,
                "lowstate_count": self.rx_count,
                "write_attempt_count": self.tx_count,
                "write_failures": self.write_failures,
                "max_lowstate_gap_s": self.max_rx_gap_s,
                "max_write_gap_s": self.max_tx_gap_s,
                "last_write_result": self.last_write_result,
                "last_write_error": self.last_write_error,
                "callback_errors": self.callback_errors,
                "last_callback_error": self.last_callback_error,
                "control_error": self.control_error,
            }

    def start(self, control_thread, details, directory=None):
        self._thread = threading.Thread(
            target=self._monitor, args=(control_thread, details, directory),
            name="dds-health", daemon=True,
        )
        self._thread.start()

    def _monitor(self, control_thread, details, directory):
        previous = None
        last_saved = 0.0
        while not self._stop.is_set():
            report = self.snapshot(control_thread.is_alive())
            report.update(details())
            key = (report["status"], tuple(report["issues"]))
            changed = key != previous
            if changed:
                print(f"DDS workflow: {report['status']} {report['issues']}", flush=True)
                previous = key
            now = time.monotonic()
            if directory is not None and (changed or now - last_saved >= 1.0):
                try:
                    with (directory / "dds_health.jsonl").open("a") as stream:
                        stream.write(json.dumps(report) + "\n")
                except OSError as exc:
                    print(f"ERROR: DDS health recording failed: {exc}", file=sys.stderr)
                    directory = None
                last_saved = now
            self._stop.wait(.1)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
