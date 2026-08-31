#!/usr/bin/env python3
"""Deploy the BigRedDog parkour ONNX policy to the Unitree low-level SDK.

The observation and depth-history layout intentionally matches
``sim2sim_bigreddog_parkour.py``.  Use ``--check`` to validate the models and
configuration without initializing ROS, DDS, or sending motor commands.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg

try:
    import tkinter as tk
    from PIL import Image as PILImage
    from PIL import ImageTk
except Exception as exc:
    tk = None
    PILImage = None
    ImageTk = None
    _TK_IMPORT_ERROR: Exception | None = exc
else:
    _TK_IMPORT_ERROR = None

try:
    from pynput import keyboard
except Exception as exc:
    keyboard = None
    _KEYBOARD_IMPORT_ERROR: Exception | None = exc
else:
    _KEYBOARD_IMPORT_ERROR = None

try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
except Exception as exc:  # Allows --check without the hardware SDK installed.
    ChannelFactoryInitialize = None
    ChannelPublisher = None
    ChannelSubscriber = None
    unitree_go_msg_dds__LowCmd_ = None
    LowCmd_ = None
    LowState_ = Any
    CRC = None
    _SDK_IMPORT_ERROR: Exception | None = exc
else:
    _SDK_IMPORT_ERROR = None


NUM_MOTORS = 12
PROPRIO_KEYS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)


def _array(config: dict[str, Any], key: str, size: int) -> np.ndarray:
    value = np.asarray(config[key], dtype=np.float32)
    if value.shape != (size,):
        raise ValueError(f"{key} must have {size} values, got {value.shape}")
    return value


def _resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML root in {config_path}")
    for section in ("policy", "control", "observations", "depth", "safety", "dds"):
        if section not in config:
            raise KeyError(f"Missing YAML section: {section}")
    return config_path, config


def projected_gravity_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Rotate world gravity (0, 0, -1) into the robot body frame."""
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError(f"Invalid IMU quaternion: {quat}")
    w, x, y, z = quat / norm
    gravity = np.array(
        [
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ],
        dtype=np.float32,
    )
    return gravity


def ramp_command(
    current: np.ndarray,
    target: np.ndarray,
    rates_per_s: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Move each command axis toward its target with a linear rate limit."""
    current = np.asarray(current, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    rates = np.asarray(rates_per_s, dtype=np.float32)
    if current.shape != (3,) or target.shape != (3,) or rates.shape != (3,):
        raise ValueError("Command, target, and ramp rates must each have 3 values")
    if np.any(rates < 0.0):
        raise ValueError("Command ramp rates cannot be negative")
    max_step = rates * float(dt)
    delta = np.clip(target - current, -max_step, max_step)
    return np.ascontiguousarray(current + delta, dtype=np.float32)


class DepthPreprocessor:
    """Convert a rectified RealSense image into one normalized 18x32 frame."""

    def __init__(self, config: dict[str, Any]):
        self.input_width = int(config["input_width"])
        self.input_height = int(config["input_height"])
        self.train_width = int(config["train_width"])
        self.train_height = int(config["train_height"])
        self.encoder_width = int(config["encoder_width"])
        self.encoder_height = int(config["encoder_height"])
        self.depth_min = float(config["min_m"])
        self.depth_max = float(config["max_m"])
        self.crop = tuple(int(v) for v in config["crop"])
        if len(self.crop) != 4:
            raise ValueError("depth.crop must be [up, down, left, right]")

        projection = np.asarray(config["projection"], dtype=np.float64)
        if projection.shape != (12,):
            raise ValueError("depth.projection must contain 12 values")
        fx_rs, fy_rs = float(projection[0]), float(projection[5])
        cx_rs, cy_rs = float(projection[2]), float(projection[6])

        hfov = math.radians(float(config["train_hfov_deg"]))
        vfov = math.radians(float(config["train_vfov_deg"]))
        fx_train = (self.train_width / 2.0) / math.tan(hfov / 2.0)
        fy_train = (self.train_height / 2.0) / math.tan(vfov / 2.0)
        cx_train = (self.train_width - 1) / 2.0
        cy_train = (self.train_height - 1) / 2.0

        u_train, v_train = np.meshgrid(
            np.arange(self.train_width, dtype=np.float32),
            np.arange(self.train_height, dtype=np.float32),
        )
        self.map_x = (fx_rs * ((u_train - cx_train) / fx_train) + cx_rs).astype(
            np.float32
        )
        self.map_y = (fy_rs * ((v_train - cy_train) / fy_train) + cy_rs).astype(
            np.float32
        )

        up, down, left, right = self.crop
        cropped_height = self.train_height - up - down
        cropped_width = self.train_width - left - right
        if cropped_height <= 0 or cropped_width <= 0:
            raise ValueError(f"Invalid depth crop: {self.crop}")

    def process(self, depth_m: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.shape != (self.input_height, self.input_width):
            raise ValueError(
                "Unexpected RealSense depth shape: "
                f"{depth.shape}; expected {(self.input_height, self.input_width)}"
            )
        invalid = ~np.isfinite(depth) | (depth <= 0.0)
        if np.any(invalid):
            depth = depth.copy()
            depth[invalid] = self.depth_max

        remapped = cv2.remap(
            depth,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.depth_max,
        )
        up, down, left, right = self.crop
        y_stop = remapped.shape[0] - down if down else remapped.shape[0]
        x_stop = remapped.shape[1] - right if right else remapped.shape[1]
        processed = remapped[up:y_stop, left:x_stop]
        if processed.shape != (self.encoder_height, self.encoder_width):
            processed = cv2.resize(
                processed,
                (self.encoder_width, self.encoder_height),
                interpolation=cv2.INTER_AREA,
            )
        processed = cv2.GaussianBlur(processed, (3, 3), sigmaX=1.0)
        processed = np.clip(processed, self.depth_min, self.depth_max)
        processed = (processed - self.depth_min) / (
            self.depth_max - self.depth_min
        )
        return np.ascontiguousarray(processed, dtype=np.float32)


class RealSenseDepthNode(Node):
    """ROS subscriber that stores the newest preprocessed policy frame."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("bigreddog_parkour_deploy_depth")
        self.topic = str(config["topic"])
        self.preprocessor = DepthPreprocessor(config)
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._first_frame = threading.Event()
        self._latest: np.ndarray | None = None
        self._latest_display: np.ndarray | None = None
        self._latest_time = 0.0
        self._frame_count = 0
        self._image_count = 0
        self._image_time = time.monotonic()
        self.subscription = self.create_subscription(
            ImageMsg,
            self.topic,
            self._callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"Waiting for depth on {self.topic}")

    def _callback(self, msg: ImageMsg) -> None:
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            now = time.monotonic()
            self._image_count += 1
            elapsed = now - self._image_time

            if elapsed > 0.1:
                #print(f"raw image freq ={self._image_count/elapsed} Hz")
                self._image_count = 0
                self._image_time = now

            if msg.encoding in ("16UC1", "mono16"):
                depth_m = raw.astype(np.float32) * 0.001
            elif msg.encoding == "32FC1":
                depth_m = raw.astype(np.float32)
            else:
                raise ValueError(f"Unsupported depth encoding: {msg.encoding}")
            invalid_mask = ~np.isfinite(depth_m) | (depth_m <= 0.0)
            display = np.rint(
                np.clip(depth_m / self.preprocessor.depth_max, 0.0, 1.0)
                * 255.0
            ).astype(np.uint8)
            display[invalid_mask] = 0
            processed = self.preprocessor.process(depth_m)
        except Exception as exc:
            self.get_logger().error(f"Rejected depth frame: {exc}")
            return

        with self._lock:
            self._latest = processed
            self._latest_display = np.ascontiguousarray(display)
            self._latest_time = time.monotonic()
            self._frame_count += 1
            first = self._frame_count == 1
        self._first_frame.set()
        if first:
            self.get_logger().info(
                f"First depth frame: encoding={msg.encoding}, output={processed.shape}"
            )

    def wait_for_first(self, timeout: float) -> bool:
        return self._first_frame.wait(None if timeout <= 0.0 else timeout)

    def latest(self) -> tuple[np.ndarray | None, float, int]:
        with self._lock:
            frame = None if self._latest is None else self._latest.copy()
            age = (
                float("inf")
                if self._latest is None
                else time.monotonic() - self._latest_time
            )
            return frame, age, self._frame_count

    def latest_display(
        self,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        float,
        int,
    ]:
        """Return generated grayscale and policy depth copies for the UI."""
        with self._lock:
            display = (
                None if self._latest_display is None else self._latest_display.copy()
            )
            processed = None if self._latest is None else self._latest.copy()
            age = (
                float("inf")
                if self._latest is None
                else time.monotonic() - self._latest_time
            )
            return (
                display,
                processed,
                age,
                self._frame_count,
            )


class ParkourPolicy:
    """ONNX parkour policy with sim2sim-compatible history semantics."""

    def __init__(self, config_path: Path, config: dict[str, Any]):
        control = config["control"]
        obs = config["observations"]
        depth = config["depth"]
        safety = config["safety"]
        policy_config = config["policy"]

        self.default_angles = _array(control, "default_angles", NUM_MOTORS)
        self.sdk_to_policy = _array(control, "sdk_to_policy", NUM_MOTORS).astype(
            np.int64
        )
        self.policy_to_sdk = _array(
            control, "policy_to_sdk", NUM_MOTORS
        ).astype(np.int64)
        expected = list(range(NUM_MOTORS))
        if sorted(self.sdk_to_policy.tolist()) != expected:
            raise ValueError("control.sdk_to_policy must be a permutation of 0..11")
        if sorted(self.policy_to_sdk.tolist()) != expected:
            raise ValueError("control.policy_to_sdk must be a permutation of 0..11")
        if not np.array_equal(
            self.sdk_to_policy[self.policy_to_sdk], np.arange(NUM_MOTORS)
        ):
            raise ValueError("The SDK/policy joint maps are not inverses")

        self.history_length = int(obs["history_length"])
        self.ang_vel_scale = float(obs["angular_velocity_scale"])
        self.joint_vel_scale = float(obs["joint_velocity_scale"])
        self.command_scale = _array(obs, "command_scale", 3)
        self.command_limits = _array(obs, "command_limits", 3)
        action_clip = safety.get("action_clip")
        self.action_clip = None if action_clip is None else float(action_clip)
        if self.action_clip is not None and (
            not np.isfinite(self.action_clip) or self.action_clip <= 0.0
        ):
            raise ValueError("safety.action_clip must be positive or null")

        self.depth_history_length = int(depth["history_length"])
        self.depth_indices = np.asarray(depth["history_indices"], dtype=np.int64)
        self.encoder_height = int(depth["encoder_height"])
        self.encoder_width = int(depth["encoder_width"])
        if np.any(self.depth_indices < 0) or np.any(
            self.depth_indices >= self.depth_history_length
        ):
            raise ValueError("depth.history_indices are outside the history buffer")

        requested = list(policy_config.get("providers", ["CPUExecutionProvider"]))
        available = ort.get_available_providers()
        providers = [provider for provider in requested if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        encoder_path = _resolve_path(config_path, policy_config["depth_encoder_path"])
        actor_path = _resolve_path(config_path, policy_config["actor_path"])
        if not encoder_path.is_file() or not actor_path.is_file():
            raise FileNotFoundError(
                f"Missing ONNX model: encoder={encoder_path}, actor={actor_path}"
            )
        self.encoder = ort.InferenceSession(
            str(encoder_path), options, providers=providers
        )
        self.actor = ort.InferenceSession(str(actor_path), options, providers=providers)
        self.encoder_input = self.encoder.get_inputs()[0].name
        self.actor_input = self.actor.get_inputs()[0].name
        self._validate_contract()
        self.reset()



    def _validate_contract(self) -> None:
        encoder_shape = self.encoder.get_inputs()[0].shape
        actor_shape = self.actor.get_inputs()[0].shape
        actor_output = self.actor.get_outputs()[0].shape
        expected_encoder = [1, len(self.depth_indices), self.encoder_height, self.encoder_width]
        if encoder_shape != expected_encoder:
            raise ValueError(
                f"Depth encoder expects {encoder_shape}, config produces {expected_encoder}"
            )
        if actor_shape != [1, 488] or actor_output != [1, NUM_MOTORS]:
            raise ValueError(
                f"Unexpected actor contract: input={actor_shape}, output={actor_output}"
            )

    def reset(self) -> None:
        self.histories: dict[str, deque[np.ndarray]] = {
            key: deque(maxlen=self.history_length) for key in PROPRIO_KEYS
        }
        self.depth_history: deque[np.ndarray] = deque(
            maxlen=self.depth_history_length
        )
        self.previous_action = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._first = True

        # Observation/policy frequency measurement
        self._obs_count = 0
        self._obs_rate_start = time.monotonic()

    def _update_proprio(self, values: dict[str, np.ndarray]) -> np.ndarray:
        if self._first:
            for key, value in values.items():
                for _ in range(self.history_length):
                    self.histories[key].append(value.copy())
            self._first = False
        else:
            for key, value in values.items():
                self.histories[key].append(value.copy())
        return np.concatenate(
            [np.concatenate(list(self.histories[key])) for key in PROPRIO_KEYS]
        ).astype(np.float32)

    def _update_depth(self, depth_frame: np.ndarray) -> np.ndarray:
        expected = (self.encoder_height, self.encoder_width)
        if depth_frame.shape != expected:
            raise ValueError(f"Depth frame is {depth_frame.shape}, expected {expected}")
        self.depth_history.append(np.asarray(depth_frame, dtype=np.float32).copy())
        history = list(self.depth_history)
        if len(history) < self.depth_history_length:
            history = [history[0]] * (self.depth_history_length - len(history)) + history
        stacked = np.stack(history[-self.depth_history_length :], axis=0)
        return np.ascontiguousarray(stacked[self.depth_indices][None, ...])

    def infer(
        self,
        qpos_sdk: np.ndarray,
        qvel_sdk: np.ndarray,
        ang_vel: np.ndarray,
        quaternion_wxyz: np.ndarray,
        command: np.ndarray,
        depth_frame: np.ndarray,
    ) -> np.ndarray:
        qpos = np.asarray(qpos_sdk, dtype=np.float32).reshape(NUM_MOTORS)
        qvel = np.asarray(qvel_sdk, dtype=np.float32).reshape(NUM_MOTORS)
        omega = np.asarray(ang_vel, dtype=np.float32).reshape(3)
        command = np.asarray(command, dtype=np.float32).reshape(3)
        command = np.clip(command, -self.command_limits, self.command_limits)

        values = {
            "base_ang_vel": omega * self.ang_vel_scale,
            "projected_gravity": projected_gravity_from_wxyz(quaternion_wxyz),
            "velocity_commands": command * self.command_scale,
            "joint_pos": (qpos - self.default_angles)[self.sdk_to_policy],
            "joint_vel": qvel[self.sdk_to_policy] * self.joint_vel_scale,
            "actions": self.previous_action.copy(),
        }
        proprio = self._update_proprio(values)
        depth_bchw = self._update_depth(depth_frame)
        latent = self.encoder.run(None, {self.encoder_input: depth_bchw})[0]
        actor_input = np.concatenate([proprio, latent.reshape(-1)])[None, :].astype(
            np.float32
        )

        self._obs_count += 1
        now = time.monotonic()
        elapsed = now - self._obs_rate_start

        if elapsed >= 1.0:
            print(f"obs output: {self._obs_count / elapsed:.1f} Hz")
            self._obs_count = 0
            self._obs_rate_start = now

        if actor_input.shape != (1, 488):
            raise ValueError(f"Actor input has unexpected shape {actor_input.shape}")

        actor_start = time.perf_counter()
        action = self.actor.run(None, {self.actor_input: actor_input})[0].reshape(-1)
        if action.shape != (NUM_MOTORS,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Invalid actor output: shape={action.shape}, values={action}")

        actor_elapsed_ms = (time.perf_counter() - actor_start) * 1000.0
        #print(f"Actor inference: {actor_elapsed_ms:.3f} ms")

        # sim2sim applies the actor output directly.  Keep clipping configurable so
        # hardware deployments can opt into it without silently changing policy
        # behaviour when parity with sim2sim is required.
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        action = action.astype(np.float32)
        self.previous_action = action.copy()
        return action


@dataclass
class RobotState:
    qpos: np.ndarray
    qvel: np.ndarray
    ang_vel: np.ndarray
    quaternion: np.ndarray
    timestamp: float


class ParkourDeployController:
    """Unitree DDS controller with explicit mode and safety transitions."""

    def __init__(
        self,
        config: dict[str, Any],
        policy: ParkourPolicy,
        depth_node: RealSenseDepthNode,
    ):
        if _SDK_IMPORT_ERROR is not None:
            raise RuntimeError(
                "unitree_sdk2py is unavailable; install it and cyclonedds 0.10.2"
            ) from _SDK_IMPORT_ERROR

        control = config["control"]
        safety = config["safety"]
        dds = config["dds"]
        self.policy = policy
        self._policy_lock = threading.Lock()
        self.depth_node = depth_node
        self.dt = float(control["dt"])
        self.decimation = int(control["decimation"])
        self.stand_duration = float(control["stand_duration"])
        self.sit_duration = float(control["sit_duration"])
        self.kps = _array(control, "kps", NUM_MOTORS)
        self.kds = _array(control, "kds", NUM_MOTORS)
        self.action_scale = _array(control, "action_scale", NUM_MOTORS)
        self.default_angles = _array(control, "default_angles", NUM_MOTORS)
        self.sit_angles = _array(control, "sit_angles", NUM_MOTORS)
        self.policy_to_sdk = _array(
            control, "policy_to_sdk", NUM_MOTORS
        ).astype(np.int64)
        self.damping_kd = float(control["damping_kd"])

        self.low_state_wait_timeout = float(safety["low_state_wait_timeout"])
        self.low_state_stale_timeout = float(safety["low_state_stale_timeout"])
        self.depth_stale_timeout = float(config["depth"]["stale_timeout"])
        self.max_tilt_deg = float(safety["max_tilt_deg"])
        joint_lower = _array(safety, "joint_lower", NUM_MOTORS)
        joint_upper = _array(safety, "joint_upper", NUM_MOTORS)
        if np.any(joint_lower >= joint_upper):
            raise ValueError("Every safety.joint_lower must be below joint_upper")
        max_target_delta = safety.get("max_target_delta")
        if max_target_delta is None:
            # Match sim2sim's target calculation while retaining physical joint
            # limits as the final hardware safety boundary.
            self.target_lower = joint_lower
            self.target_upper = joint_upper
        else:
            max_delta = np.asarray(max_target_delta, dtype=np.float32)
            if max_delta.ndim == 0:
                max_delta = np.full(NUM_MOTORS, float(max_delta), dtype=np.float32)
            if max_delta.shape != (NUM_MOTORS,):
                raise ValueError(
                    "safety.max_target_delta must be a scalar, 12 values, or null"
                )
            if not np.all(np.isfinite(max_delta)) or np.any(max_delta <= 0.0):
                raise ValueError("safety.max_target_delta values must be positive")
            self.target_lower = np.maximum(
                joint_lower, self.default_angles - max_delta
            )
            self.target_upper = np.minimum(
                joint_upper, self.default_angles + max_delta
            )

        self.command_limits = _array(config["observations"], "command_limits", 3)
        self.command_ramp_per_s = _array(
            config["observations"], "command_ramp_per_s", 3
        )
        self.policy_dt = self.dt * self.decimation
        self.command = np.zeros(3, dtype=np.float32)
        self.target_command = np.zeros(3, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.last_action_policy = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.last_requested_target = self.default_angles.copy()
        self.last_joint_limit_clips = np.zeros(NUM_MOTORS, dtype=bool)
        self._last_limit_warning_time = 0.0
        self.mode = "damping"
        self.fault_reason = ""
        self._mode_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state_event = threading.Event()
        self._state: RobotState | None = None
        self._transition_start = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._transition_time = 0.0
        self._transition_done = False
        self._counter = 0
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self._init_low_cmd()
        self.publisher = ChannelPublisher(str(dds["lowcmd_topic"]), LowCmd_)
        self.subscriber = ChannelSubscriber(str(dds["lowstate_topic"]), LowState_)


    def _init_low_cmd(self) -> None:
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i, motor in enumerate(self.low_cmd.motor_cmd):
            motor.mode = 0x01
            motor.q = float(self.sit_angles[i]) if i < NUM_MOTORS else 0.0
            motor.kp = 0.0
            motor.dq = 0.0
            motor.kd = self.damping_kd if i < NUM_MOTORS else 0.0
            motor.tau = 0.0

    def initialize_dds(self) -> None:
        self.publisher.Init()
        self.subscriber.Init(self._low_state_callback, 10)
        if not self._state_event.wait(self.low_state_wait_timeout):
            raise RuntimeError(
                f"No rt/lowstate received within {self.low_state_wait_timeout:.1f}s"
            )
        print("LowState received; controller starts in damping mode.")

    def _low_state_callback(self, msg: LowState_) -> None:
        state = RobotState(
            qpos=np.array(
                [msg.motor_state[i].q for i in range(NUM_MOTORS)],
                dtype=np.float32,
            ),
            qvel=np.array(
                [msg.motor_state[i].dq for i in range(NUM_MOTORS)],
                dtype=np.float32,
            ),
            ang_vel=np.asarray(msg.imu_state.gyroscope[:3], dtype=np.float32),
            quaternion=np.asarray(msg.imu_state.quaternion[:4], dtype=np.float32),
            timestamp=time.monotonic(),
        )
        with self._state_lock:
            self._state = state
        self._state_event.set()

    def _state_snapshot(self) -> RobotState:
        with self._state_lock:
            if self._state is None:
                raise RuntimeError("LowState is not available")
            return RobotState(
                self._state.qpos.copy(),
                self._state.qvel.copy(),
                self._state.ang_vel.copy(),
                self._state.quaternion.copy(),
                self._state.timestamp,
            )

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(
            target=self._control_loop,
            name="parkour-lowcmd",
            daemon=False,
        )
        self._thread.start()

    def shutdown(self) -> None:
        self.set_damping("shutdown")
        time.sleep(max(0.1, 3.0 * self.dt))
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.subscriber.Close()
        except Exception:
            pass
        try:
            self.publisher.Close()
        except Exception:
            pass

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        command = np.array([vx, vy, wz], dtype=np.float32)
        with self._mode_lock:
            self.target_command = np.clip(
                command,
                -self.command_limits,
                self.command_limits,
            )
            target = self.target_command.copy()
        #print(f"target_command={target.tolist()}")

    def _begin_transition(self, mode: str) -> None:
        state = self._state_snapshot()
        with self._mode_lock:
            self.mode = mode
            self.fault_reason = ""
            self._transition_start = state.qpos.copy()
            self._transition_time = time.monotonic()
            self._transition_done = False
            self.command[:] = 0.0
            self.target_command[:] = 0.0
        with self._policy_lock:
            self.policy.reset()
        print(f"mode={mode}")

    def stand_up(self) -> None:
        self._begin_transition("stand")

    def sit_down(self) -> None:
        self._begin_transition("sit")

    def move(self) -> None:
        frame, age, _ = self.depth_node.latest()
        if frame is None or age > self.depth_stale_timeout:
            raise RuntimeError(f"Cannot enter move: depth age={age:.3f}s")
        with self._mode_lock:
            if self.mode != "stand" or not self._transition_done:
                raise RuntimeError(
                    "Enter 'stand' and wait for the transition to finish before 'move'"
                )
            with self._policy_lock:
                self.policy.reset()
            self.mode = "move"
            self.fault_reason = ""
            self.command[:] = 0.0
            self.target_command[:] = 0.0
        self._counter = 0
        print("mode=move; command reset to zero")

    def set_damping(self, reason: str = "operator request") -> None:
        with self._mode_lock:
            changed = self.mode != "damping" or self.fault_reason != reason
            self.mode = "damping"
            self.fault_reason = reason
            self.command[:] = 0.0
            self.target_command[:] = 0.0
        if changed:
            print(f"mode=damping: {reason}")

    def status(self) -> str:
        state = self._state_snapshot()
        _, depth_age, frames = self.depth_node.latest()
        gravity = projected_gravity_from_wxyz(state.quaternion)
        tilt = math.degrees(
            math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        )
        with self._mode_lock:
            mode = self.mode
            command = self.command.copy()
            target_command = self.target_command.copy()
            fault = self.fault_reason
        return (
            f"mode={mode}, command={command.tolist()}, "
            f"target_command={target_command.tolist()}, "
            f"max_abs_action={np.max(np.abs(self.last_action_policy)):.3f}, "
            f"joint_limit_clips={np.flatnonzero(self.last_joint_limit_clips).tolist()}, "
            f"max_tracking_error={np.max(np.abs(self.target_dof_pos - state.qpos)):.3f}rad, "
            f"imu_quaternion={np.round(state.quaternion, 4).tolist()}, "
            f"projected_gravity={np.round(gravity, 4).tolist()}, "
            f"tilt={tilt:.1f}deg, "
            f"lowstate_age={time.monotonic() - state.timestamp:.3f}s, "
            f"depth_age={depth_age:.3f}s, depth_frames={frames}, fault={fault!r}"
        )

    def _apply_position(self, target: np.ndarray) -> None:
        for i in range(NUM_MOTORS):
            motor = self.low_cmd.motor_cmd[i]
            motor.q = float(target[i])
            motor.kp = float(self.kps[i])
            motor.dq = 0.0
            motor.kd = float(self.kds[i])
            motor.tau = 0.0

    def _apply_damping(self, qpos: np.ndarray) -> None:
        for i in range(NUM_MOTORS):
            motor = self.low_cmd.motor_cmd[i]
            motor.q = float(qpos[i])
            motor.kp = 0.0
            motor.dq = 0.0
            motor.kd = self.damping_kd
            motor.tau = 0.0

    def _transition_target(
        self, destination: np.ndarray, duration: float
    ) -> tuple[np.ndarray, bool]:
        elapsed = time.monotonic() - self._transition_time
        alpha = min(max(elapsed / max(duration, 1e-6), 0.0), 1.0)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        target = (1.0 - smooth) * self._transition_start + smooth * destination
        return target.astype(np.float32), alpha >= 1.0

    def _validate_state_safety(
        self, state: RobotState, *, check_tilt: bool = True
    ) -> None:
        age = time.monotonic() - state.timestamp
        if age > self.low_state_stale_timeout:
            raise RuntimeError(f"LowState stale for {age:.3f}s")
        if not all(
            np.all(np.isfinite(value))
            for value in (state.qpos, state.qvel, state.ang_vel, state.quaternion)
        ):
            raise RuntimeError("LowState contains NaN or infinity")
        if check_tilt:
            gravity = projected_gravity_from_wxyz(state.quaternion)
            tilt = math.degrees(
                math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
            )
            if tilt > self.max_tilt_deg:
                raise RuntimeError(
                    f"Robot tilt {tilt:.1f}deg exceeds {self.max_tilt_deg:.1f}deg"
                )

    def _control_tick(self) -> None:
        state = self._state_snapshot()
        with self._mode_lock:
            mode = self.mode

        if mode == "damping":
            self._apply_damping(state.qpos)
            return

        # Sitting is the operator-requested recovery path after a tilt fault. It
        # still requires fresh, finite feedback, but deliberately bypasses only
        # the tilt threshold so the controller cannot immediately fall back to
        # damping before completing the sit transition.
        self._validate_state_safety(state, check_tilt=mode != "sit")
        if mode == "stand":
            target, done = self._transition_target(
                self.default_angles, self.stand_duration
            )
            self._apply_position(target)
            if done:
                self.target_dof_pos = self.default_angles.copy()
                with self._mode_lock:
                    self._transition_done = True
            return
        if mode == "sit":
            target, _ = self._transition_target(self.sit_angles, self.sit_duration)
            self._apply_position(target)
            return
        if mode != "move":
            raise RuntimeError(f"Unknown control mode: {mode}")

        if self._counter % self.decimation == 0:
            with self._mode_lock:
                self.command = ramp_command(
                    self.command,
                    self.target_command,
                    self.command_ramp_per_s,
                    self.policy_dt,
                )
                command = self.command.copy()
            depth_frame, depth_age, _ = self.depth_node.latest()
            if depth_frame is None or depth_age > self.depth_stale_timeout:
                raise RuntimeError(f"Depth stale for {depth_age:.3f}s")
            with self._policy_lock:
                action_policy = self.policy.infer(
                    state.qpos,
                    state.qvel,
                    state.ang_vel,
                    state.quaternion,
                    command,
                    depth_frame,
                )
            action_sdk = action_policy[self.policy_to_sdk]
            target = self.default_angles + action_sdk * self.action_scale
            self.last_action_policy = action_policy.copy()
            self.last_requested_target = target.copy()
            self.last_joint_limit_clips = (target < self.target_lower) | (
                target > self.target_upper
            )
            self.target_dof_pos = np.clip(
                target, self.target_lower, self.target_upper
            ).astype(np.float32)
            now = time.monotonic()
            if (
                np.any(self.last_joint_limit_clips)
                and now - self._last_limit_warning_time >= 1.0
            ):
                clipped = np.flatnonzero(self.last_joint_limit_clips).tolist()
                print(f"[WARN] Hardware joint limits clipped SDK joints {clipped}")
                self._last_limit_warning_time = now
        self._apply_position(self.target_dof_pos)
        self._counter += 1

    def _control_loop(self) -> None:
        self._cmd_count = 0
        self._cmd_rate_start = time.monotonic()
        next_tick = time.perf_counter()

        while self._running.is_set():
            try:
                self._control_tick()
            except Exception as exc:
                self.set_damping(str(exc))
                try:
                    self._apply_damping(self._state_snapshot().qpos)
                except Exception:
                    pass

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.publisher.Write(self.low_cmd)

            self._cmd_count += 1
            now = time.monotonic()
            elapsed = now - self._cmd_rate_start

            if elapsed >= 1.0:
                #print(f"LowCmd output: {self._cmd_count / elapsed:.1f} Hz")
                self._cmd_count = 0
                self._cmd_rate_start = now

            next_tick += self.dt
            remaining = next_tick - time.perf_counter()

            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -self.dt:
                # Avoid attempting a large burst of catch-up iterations.
                next_tick = time.perf_counter()


def run_check(config_path: Path, config: dict[str, Any]) -> None:
    """Validate the complete observation/model path without ROS or DDS."""
    preprocessor = DepthPreprocessor(config["depth"])
    policy = ParkourPolicy(config_path, config)
    raw_depth = np.full(
        (preprocessor.input_height, preprocessor.input_width),
        1.0,
        dtype=np.float32,
    )
    depth = preprocessor.process(raw_depth)
    action = policy.infer(
        policy.default_angles,
        np.zeros(NUM_MOTORS, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        depth,
    )
    print(f"config: {config_path}")
    print(f"depth frame: shape={depth.shape}, range=({depth.min():.3f}, {depth.max():.3f})")
    print(f"action: shape={action.shape}, range=({action.min():.3f}, {action.max():.3f})")
    print("CHECK PASSED: no ROS or DDS motor commands were used")


def interactive_console(controller: ParkourDeployController) -> None:
    print(
        "Commands: stand | move | sit | damp | cmd VX VY WZ | status | exit"
    )
    while True:
        try:
            words = input("parkour> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not words:
            continue
        try:
            if words[0] == "stand" and len(words) == 1:
                controller.stand_up()
            elif words[0] == "move" and len(words) == 1:
                controller.move()
            elif words[0] == "sit" and len(words) == 1:
                controller.sit_down()
            elif words[0] in ("damp", "stop") and len(words) == 1:
                controller.set_damping()
            elif words[0] == "cmd" and len(words) == 4:
                controller.set_command(*(float(value) for value in words[1:]))
            elif words[0] == "status" and len(words) == 1:
                print(controller.status())
            elif words[0] in ("exit", "quit") and len(words) == 1:
                return
            else:
                print("Invalid command")
        except Exception as exc:
            print(f"Command rejected: {exc}")


class KeyboardCommandController:
    """Hold 7/8/9 to command yaw-left/forward/yaw-right."""

    MOVE_KEYS = frozenset(("7", "8", "9"))

    def __init__(self, controller: ParkourDeployController):
        self.controller = controller
        self.pressed: set[str] = set()
        self._lock = threading.Lock()
        self.listener = None
        if keyboard is not None:
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )

    def _publish_command(self) -> None:
        with self._lock:
            pressed = self.pressed.copy()
        vx = float(self.controller.command_limits[0]) if "8" in pressed else 0.0
        wz = 0.0
        if "7" in pressed:
            wz += float(self.controller.command_limits[2])
        if "9" in pressed:
            wz -= float(self.controller.command_limits[2])
        self.controller.set_command(vx, 0.0, wz)

    def press(self, value: str) -> None:
        if value not in self.MOVE_KEYS:
            return
        with self._lock:
            changed = value not in self.pressed
            self.pressed.add(value)
        if changed:
            self._publish_command()

    def release(self, value: str) -> None:
        with self._lock:
            changed = value in self.pressed
            self.pressed.discard(value)
        if changed:
            self._publish_command()

    def stop_motion(self) -> None:
        with self._lock:
            self.pressed.clear()
        self.controller.set_command(0.0, 0.0, 0.0)

    def pressed_snapshot(self) -> set[str]:
        with self._lock:
            return self.pressed.copy()

    def _on_press(self, key: Any) -> None:
        value = getattr(key, "char", None)
        if value is not None:
            self.press(value)

    def _on_release(self, key: Any) -> None:
        value = getattr(key, "char", None)
        if value is not None:
            self.release(value)

    def start(self) -> None:
        if self.listener is not None:
            self.listener.start()
            print("Keyboard: hold 8=forward, 7=turn left, 9=turn right; release=stop")
        else:
            print(
                "[WARN] pynput is unavailable; use the control-window buttons "
                f"instead ({_KEYBOARD_IMPORT_ERROR})"
            )

    def stop(self) -> None:
        self.stop_motion()
        if self.listener is not None:
            self.listener.stop()


class ParkourControlWindow:
    """OpenCV operator controls with live raw and processed depth previews."""

    WIDTH = 940
    HEIGHT = 560
    WINDOW_NAME = "BigRedDog Parkour Control + Depth"
    BG = (27, 25, 22)
    PANEL = (42, 39, 35)
    BUTTON = (77, 70, 64)
    TEXT = (238, 238, 238)
    MUTED = (175, 173, 170)
    GREEN = (75, 155, 55)
    BRIGHT_GREEN = (90, 220, 75)
    RED = (55, 55, 185)

    def __init__(
        self,
        controller: ParkourDeployController,
        keyboard_controller: KeyboardCommandController,
        depth_node: RealSenseDepthNode,
        depth_config: dict[str, Any],
    ) -> None:
        self.controller = controller
        self.keyboard_controller = keyboard_controller
        self.depth_node = depth_node
        self.topic = str(depth_config["topic"])
        self.alive = True
        self.message = "Ready"
        self.message_color = self.MUTED
        self._mouse_key: str | None = None
        self.move_rects = {
            "7": (25, 145, 115, 215),
            "8": (125, 145, 215, 215),
            "9": (225, 145, 315, 215),
        }
        self.stop_rect = (125, 225, 215, 290)
        self.mode_rects = {
            "stand": (25, 330, 115, 375),
            "move": (125, 330, 215, 375),
            "sit": (225, 330, 315, 375),
            "damp": (25, 390, 315, 440),
        }
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)
        self.draw()

    @staticmethod
    def _contains(rect: tuple[int, int, int, int], x: int, y: int) -> bool:
        return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    @staticmethod
    def _text(
        image: np.ndarray,
        value: str,
        position: tuple[int, int],
        scale: float = 0.5,
        color: tuple[int, int, int] = TEXT,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            image,
            value,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _run_action(self, label: str, action: Any) -> None:
        try:
            action()
            self.message = label
            self.message_color = self.BRIGHT_GREEN
        except Exception as exc:
            self.message = f"Rejected: {exc}"
            self.message_color = (90, 120, 255)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _data: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, rect in self.move_rects.items():
                if self._contains(rect, x, y):
                    self.keyboard_controller.press(key)
                    self._mouse_key = key
                    return
            if self._contains(self.stop_rect, x, y):
                self.keyboard_controller.stop_motion()
                self.message = "Motion command stopped"
                self.message_color = self.BRIGHT_GREEN
                return
            actions = {
                "stand": ("Standing up", self.controller.stand_up),
                "move": ("Move mode enabled", self.controller.move),
                "sit": ("Sitting down", self.controller.sit_down),
                "damp": (
                    "Damping mode enabled",
                    lambda: self.controller.set_damping("control window"),
                ),
            }
            for name, rect in self.mode_rects.items():
                if self._contains(rect, x, y):
                    label, action = actions[name]
                    self._run_action(label, action)
                    return
        elif event == cv2.EVENT_LBUTTONUP and self._mouse_key is not None:
            self.keyboard_controller.release(self._mouse_key)
            self._mouse_key = None

    def _draw_button(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        title: str,
        subtitle: str = "",
        *,
        active: bool = False,
        danger: bool = False,
    ) -> None:
        color = self.RED if danger else self.BRIGHT_GREEN if active else self.BUTTON
        cv2.rectangle(canvas, rect[:2], rect[2:], color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, rect[:2], rect[2:], (150, 150, 150), 1, cv2.LINE_AA)
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0]
        cx = (rect[0] + rect[2]) // 2
        title_y = rect[1] + (29 if subtitle else 29)
        self._text(canvas, title, (cx - title_size[0] // 2, title_y), 0.58, self.TEXT, 2)
        if subtitle:
            sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            self._text(
                canvas,
                subtitle,
                (cx - sub_size[0] // 2, rect[1] + 54),
                0.35,
                self.TEXT,
            )

    def _controller_snapshot(self) -> tuple[str, np.ndarray, np.ndarray, str]:
        with self.controller._mode_lock:
            return (
                self.controller.mode,
                self.controller.command.copy(),
                self.controller.target_command.copy(),
                self.controller.fault_reason,
            )

    def draw(self) -> None:
        if not self.alive:
            return
        try:
            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.alive = False
                return
        except cv2.error:
            self.alive = False
            return

        canvas = np.full((self.HEIGHT, self.WIDTH, 3), self.BG, dtype=np.uint8)
        cv2.rectangle(canvas, (10, 10), (330, 545), self.PANEL, -1)
        mode, command, target, fault = self._controller_snapshot()
        pressed = self.keyboard_controller.pressed_snapshot()

        self._text(canvas, "BigRedDog Control", (25, 42), 0.72, self.TEXT, 2)
        mode_color = self.BRIGHT_GREEN if mode == "move" else (80, 190, 255)
        self._text(canvas, f"MODE: {mode.upper()}", (25, 78), 0.58, mode_color, 2)
        self._text(
            canvas,
            f"cmd  {command[0]:+.2f}  {command[1]:+.2f}  {command[2]:+.2f}",
            (25, 106),
            0.48,
            self.TEXT,
        )
        self._text(
            canvas,
            f"target {target[0]:+.2f}  {target[1]:+.2f}  {target[2]:+.2f}",
            (25, 129),
            0.43,
            self.MUTED,
        )

        move_labels = {"7": "TURN LEFT", "8": "FORWARD", "9": "TURN RIGHT"}
        for key, rect in self.move_rects.items():
            self._draw_button(
                canvas, rect, key, move_labels[key], active=key in pressed
            )
        self._draw_button(canvas, self.stop_rect, "STOP", "MOTION", danger=True)

        self._text(canvas, "Robot mode", (25, 318), 0.46, self.MUTED)
        self._draw_button(canvas, self.mode_rects["stand"], "STAND", active=mode == "stand")
        self._draw_button(canvas, self.mode_rects["move"], "MOVE", active=mode == "move")
        self._draw_button(canvas, self.mode_rects["sit"], "SIT", active=mode == "sit")
        self._draw_button(canvas, self.mode_rects["damp"], "DAMPING", danger=True)

        message = fault if fault else self.message
        color = (90, 120, 255) if fault else self.message_color
        for index in range(3):
            part = message[index * 44 : (index + 1) * 44]
            if part:
                self._text(canvas, part, (25, 474 + index * 20), 0.38, color)

        display, processed, age, frames = (
            self.depth_node.latest_display()
        )
        self._text(canvas, "Live depth", (355, 35), 0.68, self.TEXT, 2)
        self._text(canvas, self.topic, (355, 58), 0.38, self.MUTED)
        depth_color = self.BRIGHT_GREEN if age <= self.controller.depth_stale_timeout else (80, 120, 255)
        self._text(
            canvas,
            f"frame {frames} age {age:.3f}s",
            (740, 58),
            0.38,
            depth_color,
        )

        if display is None:
            cv2.rectangle(canvas, (355, 75), (915, 415), (0, 0, 0), -1)
            self._text(canvas, "Waiting for depth image...", (520, 250), 0.55, self.MUTED)
        else:
            if display.ndim == 2:
                display_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            else:
                display_bgr = display
            display_bgr = cv2.resize(
                display_bgr, (453, 340), interpolation=cv2.INTER_NEAREST
            )
            canvas[75:415, 408:861] = display_bgr

        self._text(canvas, "Policy depth (18 x 32)", (355, 445), 0.46, self.MUTED)
        if processed is not None:
            policy_gray = (
                (1.0 - np.clip(processed, 0.0, 1.0)) * 255.0
            ).astype(np.uint8)
            policy_bgr = cv2.cvtColor(policy_gray, cv2.COLOR_GRAY2BGR)
            policy_bgr = cv2.resize(
                policy_bgr, (384, 90), interpolation=cv2.INTER_NEAREST
            )
            canvas[458:548, 355:739] = policy_bgr
        self._text(
            canvas,
            "policy: near = white",
            (750, 505),
            0.38,
            self.MUTED,
        )

        cv2.imshow(self.WINDOW_NAME, canvas)
        cv2.waitKey(1)

    def run(self) -> None:
        while self.alive:
            self.draw()
            time.sleep(1.0 / 30.0)

    def close(self) -> None:
        if self._mouse_key is not None:
            self.keyboard_controller.release(self._mouse_key)
            self._mouse_key = None
        if self.alive:
            cv2.destroyWindow(self.WINDOW_NAME)
        self.alive = False


class TkParkourControlWindow:
    """Tk control/depth window for OpenCV installations built without HighGUI."""

    BG = "#1b1916"
    PANEL = "#2a2723"
    BUTTON = "#4d4640"
    TEXT = "#eeeeee"
    MUTED = "#afadaa"
    GREEN = "#5adc4b"
    RED = "#b93737"
    ORANGE = "#ffbe50"

    def __init__(
        self,
        controller: ParkourDeployController,
        keyboard_controller: KeyboardCommandController,
        depth_node: RealSenseDepthNode,
        depth_config: dict[str, Any],
    ) -> None:
        if tk is None or PILImage is None or ImageTk is None:
            raise RuntimeError(
                "The control window requires tkinter and Pillow"
            ) from _TK_IMPORT_ERROR
        self.controller = controller
        self.keyboard_controller = keyboard_controller
        self.depth_node = depth_node
        self.topic = str(depth_config["topic"])
        self.alive = True
        self._mouse_key: str | None = None
        self._raw_photo: Any = None
        self._policy_photo: Any = None

        self.root = tk.Tk()
        self.root.title("BigRedDog Parkour Control + Depth")
        self.root.geometry("940x590")
        self.root.minsize(850, 540)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        left = tk.Frame(self.root, bg=self.PANEL, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left.pack_propagate(False)
        right = tk.Frame(self.root, bg=self.BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 12), pady=10)

        tk.Label(
            left,
            text="BigRedDog Control",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", padx=15, pady=(14, 4))
        self.mode_var = tk.StringVar(value="MODE: DAMPING")
        tk.Label(
            left,
            textvariable=self.mode_var,
            bg=self.PANEL,
            fg=self.ORANGE,
            font=("Arial", 14, "bold"),
        ).pack(anchor="w", padx=15)
        self.command_var = tk.StringVar(value="cmd    +0.00  +0.00  +0.00")
        self.target_var = tk.StringVar(value="target +0.00  +0.00  +0.00")
        for variable, color in (
            (self.command_var, self.TEXT),
            (self.target_var, self.MUTED),
        ):
            tk.Label(
                left,
                textvariable=variable,
                bg=self.PANEL,
                fg=color,
                font=("Courier", 11),
            ).pack(anchor="w", padx=15, pady=2)

        move_frame = tk.Frame(left, bg=self.PANEL)
        move_frame.pack(padx=10, pady=(16, 5))
        labels = {"7": "7\nTURN LEFT", "8": "8\nFORWARD", "9": "9\nTURN RIGHT"}
        self.move_buttons: dict[str, Any] = {}
        for column, key in enumerate(("7", "8", "9")):
            button = tk.Button(
                move_frame,
                text=labels[key],
                width=9,
                height=3,
                bg=self.BUTTON,
                fg=self.TEXT,
                activebackground=self.GREEN,
                activeforeground=self.TEXT,
                relief=tk.FLAT,
                font=("Arial", 9, "bold"),
            )
            button.grid(row=0, column=column, padx=3)
            button.bind(
                "<ButtonPress-1>",
                lambda _event, value=key: self._press_motion(value),
            )
            self.move_buttons[key] = button
        self.root.bind_all("<ButtonRelease-1>", self._release_mouse_motion)

        tk.Button(
            left,
            text="STOP MOTION",
            command=self._stop_motion,
            bg=self.RED,
            fg=self.TEXT,
            activebackground="#e14b4b",
            activeforeground=self.TEXT,
            relief=tk.FLAT,
            font=("Arial", 11, "bold"),
        ).pack(fill=tk.X, padx=72, pady=(5, 18), ipady=7)

        tk.Label(
            left,
            text="Robot mode",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Arial", 11),
        ).pack(anchor="w", padx=15)
        modes = tk.Frame(left, bg=self.PANEL)
        modes.pack(padx=10, pady=6)
        for column, (name, label, action) in enumerate(
            (
                ("stand", "STAND", self.controller.stand_up),
                ("move", "MOVE", self.controller.move),
                ("sit", "SIT", self.controller.sit_down),
            )
        ):
            tk.Button(
                modes,
                text=label,
                width=8,
                command=lambda n=name, a=action: self._run_action(n, a),
                bg=self.BUTTON,
                fg=self.TEXT,
                activebackground=self.GREEN,
                activeforeground=self.TEXT,
                relief=tk.FLAT,
                font=("Arial", 10, "bold"),
            ).grid(row=0, column=column, padx=3, ipady=7)
        tk.Button(
            left,
            text="DAMPING",
            command=lambda: self._run_action(
                "damping",
                lambda: self.controller.set_damping("control window"),
            ),
            bg=self.RED,
            fg=self.TEXT,
            activebackground="#e14b4b",
            activeforeground=self.TEXT,
            relief=tk.FLAT,
            font=("Arial", 11, "bold"),
        ).pack(fill=tk.X, padx=15, pady=6, ipady=8)
        self.message_var = tk.StringVar(value="Ready")
        self.message_label = tk.Label(
            left,
            textvariable=self.message_var,
            wraplength=285,
            justify=tk.LEFT,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Arial", 10),
        )
        self.message_label.pack(anchor="w", padx=15, pady=10)

        tk.Label(
            right,
            text="Live depth",
            bg=self.BG,
            fg=self.TEXT,
            font=("Arial", 17, "bold"),
        ).pack(anchor="w")
        tk.Label(
            right,
            text=self.topic,
            bg=self.BG,
            fg=self.MUTED,
            font=("Arial", 10),
        ).pack(anchor="w")
        self.depth_status_var = tk.StringVar(value="Waiting for depth image...")
        tk.Label(
            right,
            textvariable=self.depth_status_var,
            bg=self.BG,
            fg=self.GREEN,
            font=("Arial", 10),
        ).pack(anchor="w", pady=(2, 5))
        self.raw_image_label = tk.Label(
            right,
            text="Waiting for depth image...",
            bg="#000000",
            fg=self.MUTED,
            width=60,
            height=20,
        )
        self.raw_image_label.pack(anchor="center", fill=tk.BOTH, expand=True)
        tk.Label(
            right,
            text="Policy depth (18 x 32, near = white)",
            bg=self.BG,
            fg=self.MUTED,
            font=("Arial", 10),
        ).pack(anchor="w", pady=(8, 3))
        self.policy_image_label = tk.Label(right, bg="#000000")
        self.policy_image_label.pack(anchor="w")

        for key in KeyboardCommandController.MOVE_KEYS:
            self.root.bind(
                f"<KeyPress-{key}>",
                lambda _event, value=key: self.keyboard_controller.press(value),
            )
            self.root.bind(
                f"<KeyRelease-{key}>",
                lambda _event, value=key: self.keyboard_controller.release(value),
            )
        self._refresh()

    def _run_action(self, label: str, action: Any) -> None:
        try:
            action()
            self.message_var.set(f"Mode requested: {label}")
            self.message_label.configure(fg=self.GREEN)
        except Exception as exc:
            self.message_var.set(f"Rejected: {exc}")
            self.message_label.configure(fg="#ff785a")

    def _press_motion(self, key: str) -> None:
        self.keyboard_controller.press(key)
        self._mouse_key = key

    def _release_mouse_motion(self, _event: Any = None) -> None:
        if self._mouse_key is not None:
            self.keyboard_controller.release(self._mouse_key)
            self._mouse_key = None

    def _stop_motion(self) -> None:
        self.keyboard_controller.stop_motion()
        self.message_var.set("Motion command stopped")
        self.message_label.configure(fg=self.GREEN)

    @staticmethod
    def _photo(image: np.ndarray, size: tuple[int, int]) -> Any:
        pil_image = PILImage.fromarray(np.ascontiguousarray(image))
        resampling = getattr(PILImage, "Resampling", PILImage)
        pil_image = pil_image.resize(size, resampling.NEAREST)
        return ImageTk.PhotoImage(pil_image)

    def _refresh(self) -> None:
        if not self.alive:
            return
        with self.controller._mode_lock:
            mode = self.controller.mode
            command = self.controller.command.copy()
            target = self.controller.target_command.copy()
            fault = self.controller.fault_reason
        self.mode_var.set(f"MODE: {mode.upper()}")
        self.command_var.set(
            f"cmd    {command[0]:+0.2f}  {command[1]:+0.2f}  {command[2]:+0.2f}"
        )
        self.target_var.set(
            f"target {target[0]:+0.2f}  {target[1]:+0.2f}  {target[2]:+0.2f}"
        )
        if fault:
            self.message_var.set(fault)
            self.message_label.configure(fg="#ff785a")

        pressed = self.keyboard_controller.pressed_snapshot()
        for key, button in self.move_buttons.items():
            button.configure(bg=self.GREEN if key in pressed else self.BUTTON)

        display, processed, age, frames = (
            self.depth_node.latest_display()
        )
        self.depth_status_var.set(
            f"frame {frames}  age {age:.3f}s"
        )
        if display is not None:
            if display.ndim == 2:
                preview = display
            else:
                preview = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            self._raw_photo = self._photo(preview, (520, 340))
            self.raw_image_label.configure(image=self._raw_photo, text="")
        if processed is not None:
            policy = (
                (1.0 - np.clip(processed, 0.0, 1.0)) * 255.0
            ).astype(np.uint8)
            self._policy_photo = self._photo(policy, (384, 90))
            self.policy_image_label.configure(image=self._policy_photo)
        self.root.after(33, self._refresh)

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        if not self.alive:
            return
        self._release_mouse_motion()
        self.alive = False
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="isaaclab_parkour_deploy.yaml",
        help="Parkour deployment YAML",
    )
    parser.add_argument(
        "interface",
        nargs="?",
        help=(
            "Optional Unitree DDS interface, matching isaaclab_deploy.py. "
            "When omitted, use dds.interface from the YAML."
        ),
    )
    parser.add_argument("--domain", type=int, help="Override Unitree DDS domain")
    parser.add_argument(
        "--depth-topic",
        help="Override the metric policy-depth topic from the YAML",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Use the text console instead of the control/depth window",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate config/models only; never initialize DDS or motors",
    )
    args = parser.parse_args()

    config_path, config = load_config(args.config)
    if args.depth_topic:
        config["depth"]["topic"] = args.depth_topic
    if args.check:
        run_check(config_path, config)
        return 0

    if _SDK_IMPORT_ERROR is not None:
        raise RuntimeError(
            "unitree_sdk2py/cyclonedds is not available in this environment"
        ) from _SDK_IMPORT_ERROR

    policy = ParkourPolicy(config_path, config)
    dds_config = config["dds"]
    interface = args.interface or str(dds_config["interface"])
    domain = int(args.domain if args.domain is not None else dds_config["domain"])

    print("DANGER: this program sends low-level commands to 12 motors.")
    print(f"DDS interface={interface!r}, domain={domain}")
    print(f"Depth topic={config['depth']['topic']}")

    ChannelFactoryInitialize(domain, interface)
    rclpy.init(args=[])
    depth_node = RealSenseDepthNode(config["depth"])
    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(depth_node,),
        name="parkour-depth-ros",
        daemon=True,
    )
    ros_thread.start()
    controller: ParkourDeployController | None = None
    keyboard_controller: KeyboardCommandController | None = None
    control_window: ParkourControlWindow | TkParkourControlWindow | None = None
    try:
        timeout = float(config["depth"]["wait_timeout"])
        if not depth_node.wait_for_first(timeout):
            raise RuntimeError(
                f"No valid depth frame on {depth_node.topic} within {timeout:.1f}s"
            )
        controller = ParkourDeployController(config, policy, depth_node)
        controller.initialize_dds()
        controller.start()
        keyboard_controller = KeyboardCommandController(controller)
        keyboard_controller.start()
        if args.console:
            interactive_console(controller)
        else:
            highgui_available = not any(
                "GUI:" in line and "NONE" in line
                for line in cv2.getBuildInformation().splitlines()
            )
            if highgui_available:
                try:
                    control_window = ParkourControlWindow(
                        controller,
                        keyboard_controller,
                        depth_node,
                        config["depth"],
                    )
                except cv2.error:
                    control_window = None
            if control_window is None:
                print(
                    "[INFO] OpenCV has no GUI backend; using the Tkinter "
                    "control window."
                )
                try:
                    control_window = TkParkourControlWindow(
                        controller,
                        keyboard_controller,
                        depth_node,
                        config["depth"],
                    )
                except Exception as exc:
                    is_display_error = (
                        tk is None
                        or (hasattr(tk, "TclError") and isinstance(exc, tk.TclError))
                    )
                    if not is_display_error:
                        raise
                    print(
                        "[WARN] Tkinter could not connect to the graphical "
                        f"display ({exc}); using the text console."
                    )
                    interactive_console(controller)
            if control_window is not None:
                control_window.run()
    finally:
        if control_window is not None:
            control_window.close()
        if keyboard_controller is not None:
            keyboard_controller.stop()
        if controller is not None:
            controller.shutdown()
        depth_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
