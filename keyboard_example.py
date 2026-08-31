#!/usr/bin/env python3
"""Run an MGDP Stage 1 policy on a Unitree Go2 in MuJoCo.

This runner intentionally has no Isaac Gym dependency.  It reconstructs the
small inference-only pieces of the Stage 1 actor and world model, then feeds
them observations with the same ordering and scaling used during training.
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import time
from typing import Dict, Optional, Sequence, Tuple


# Select a headless backend before importing mujoco.  A caller can override it
# explicitly, e.g. MUJOCO_GL=osmesa or MUJOCO_GL=glfw.
if "MUJOCO_GL" not in os.environ and "--headless" in sys.argv:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
LEGGED_GYM_ROOT = SCRIPT_DIR.parent
DEFAULT_RUN = LEGGED_GYM_ROOT / "models/MGDP/stage1/baseline/stage1_nn"

JOINT_NAMES: Tuple[str, ...] = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

DEFAULT_DOF_POS = np.asarray(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
     0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=np.float64,
)


def _group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class DoubleConv(nn.Module):
    """Checkpoint-compatible copy of the training-side convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.block(images)


class ImageEncoder(nn.Module):
    """Inference-compatible copy of random_dog/utils/net_work.py:ImageEncoder."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = DoubleConv(2, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv(64, 96)
        self.bottleneck = DoubleConv(96, 96)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 192),
            nn.LayerNorm(192),
            nn.SiLU(inplace=True),
            nn.Linear(192, 64),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(images))
        x = self.pool2(self.conv2(x))
        x = self.bottleneck(self.conv3(x))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)


class Stage1Policy(nn.Module):
    """Actor, four-frame proprio encoder, and recurrent depth encoder."""

    def __init__(self, policy_path: Path, world_model_path: Path, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.proprio_encoder = nn.Sequential(
            nn.Linear(180, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 7),
        )
        self.image_encoder = ImageEncoder()
        self.depth_memory = nn.GRU(64, 64, num_layers=1, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(116, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12),
        )

        policy_checkpoint = _torch_load(policy_path, device)
        actor_state = policy_checkpoint.get("actor_state_dict", policy_checkpoint)
        self.actor.load_state_dict(_strip_prefix(actor_state, "actor."), strict=True)
        self.proprio_encoder.load_state_dict(
            _strip_prefix(actor_state, "mlp_encoder.encoder."), strict=True
        )

        world_checkpoint = _torch_load(world_model_path, device)
        if "image_encoder" not in world_checkpoint or "memory_encoder" not in world_checkpoint:
            raise KeyError(
                "World-model checkpoint must contain image_encoder and memory_encoder"
            )
        self.image_encoder.load_state_dict(world_checkpoint["image_encoder"], strict=True)
        self.depth_memory.load_state_dict(
            _strip_prefix(world_checkpoint["memory_encoder"], "rnn."), strict=True
        )

        self.to(device).eval()
        self.depth_hidden: Optional[torch.Tensor] = None

    def reset_memory(self) -> None:
        self.depth_hidden = None

    @torch.inference_mode()
    def encode_depth(self, depth_history: torch.Tensor) -> torch.Tensor:
        image_token = self.image_encoder(depth_history)
        output, self.depth_hidden = self.depth_memory(
            image_token.unsqueeze(1), self.depth_hidden
        )
        return output[:, 0]

    @torch.inference_mode()
    def act(
        self,
        observation: np.ndarray,
        observation_history: np.ndarray,
        depth_token: torch.Tensor,
    ) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        history = torch.as_tensor(
            observation_history, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        proprio_token = self.proprio_encoder(history)
        actor_input = torch.cat((obs, proprio_token, depth_token), dim=-1)
        return self.actor(actor_input)[0].cpu().numpy()


def _torch_load(path: Path, device: torch.device) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        return torch.load(str(path), map_location=device)


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    result = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not result:
        raise KeyError("No checkpoint parameters found with prefix {!r}".format(prefix))
    return result


class _PygameKeyboardWindow:
    """Pygame window. This object lives entirely in the UI subprocess."""

    WIDTH = 760
    HEIGHT = 430

    def __init__(
        self,
        simulation: "Go2Simulation",
        vx_scale: float = 1.0,
        vy_scale: float = 1.0,
        yaw_scale: float = 1.0,
        smooth: float = 0.2,
    ) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        # Only initialize the subsystems used here; initializing the mixer can
        # emit ALSA errors on simulation workstations without an audio device.
        pygame.display.init()
        pygame.font.init()
        self.pygame = pygame
        self.simulation = simulation
        self.vx_scale = vx_scale
        self.vy_scale = vy_scale
        self.yaw_scale = yaw_scale
        self.smooth = smooth
        self.vx, self.vy, self.yaw = simulation.commands.astype(float)
        self.alive = True

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Go2 Keyboard Control + Depth")
        try:
            from pygame._sdl2 import Window
            self.window = Window.from_display_module()
            self.window.position = (20, 40)
            self.window.focus()
        except (ImportError, RuntimeError):
            self.window = None
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.font = pygame.font.SysFont("Arial", 18)
        self.small_font = pygame.font.SysFont("Arial", 15)

        self.key_rects = {
            "Q": pygame.Rect(30, 150, 75, 55),
            "W": pygame.Rect(115, 150, 75, 55),
            "E": pygame.Rect(200, 150, 75, 55),
            "A": pygame.Rect(30, 215, 75, 55),
            "X": pygame.Rect(115, 215, 75, 55),
            "D": pygame.Rect(200, 215, 75, 55),
            "S": pygame.Rect(115, 280, 75, 55),
        }
        self.reset_rect = pygame.Rect(30, 360, 120, 38)
        self.camera_rect = pygame.Rect(165, 360, 150, 38)
        self.last_pressed = {name: False for name in self.key_rects}
        # Paint the window immediately. Waiting for the first MuJoCo viewer
        # synchronization can leave a blank SDL window on some desktops.
        self.draw(self.last_pressed)

    def smooth_update(self, old: float, new: float) -> float:
        return old * (1.0 - self.smooth) + new * self.smooth

    def _draw_text(self, text: str, position: Tuple[int, int], color=(235, 235, 235),
                   font=None) -> None:
        surface = (font or self.font).render(text, True, color)
        self.screen.blit(surface, position)

    def _draw_button(self, name: str, subtitle: str, active: bool) -> None:
        rect = self.key_rects[name]
        if name == "X":
            color = (180, 55, 55) if not active else (235, 75, 75)
        else:
            color = (38, 145, 82) if active else (58, 64, 72)
        self.pygame.draw.rect(self.screen, color, rect, border_radius=7)
        self.pygame.draw.rect(self.screen, (150, 155, 162), rect, 1, border_radius=7)
        label = self.title_font.render(name, True, (255, 255, 255))
        self.screen.blit(label, label.get_rect(center=(rect.centerx, rect.centery - 8)))
        sub = self.small_font.render(subtitle, True, (225, 225, 225))
        self.screen.blit(sub, sub.get_rect(center=(rect.centerx, rect.centery + 15)))

    def draw(self, pressed: Dict[str, bool]) -> None:
        self.screen.fill((20, 22, 26))
        self._draw_text("Go2 Keyboard Control", (28, 20), font=self.title_font)
        self._draw_text("Hold a key or hold a button with the mouse", (28, 53),
                        color=(170, 175, 182), font=self.small_font)
        self._draw_text("vx {:+.2f}   vy {:+.2f}   yaw {:+.2f}".format(
            self.vx, self.vy, self.yaw), (28, 85), color=(120, 220, 160))
        self._draw_text("Commands return smoothly to zero when released", (28, 113),
                        color=(170, 175, 182), font=self.small_font)

        subtitles = {
            "Q": "yaw +", "W": "forward", "E": "yaw -",
            "A": "left", "X": "STOP", "D": "right", "S": "back",
        }
        for name, subtitle in subtitles.items():
            self._draw_button(name, subtitle, pressed[name])

        self.pygame.draw.rect(self.screen, (58, 64, 72), self.reset_rect, border_radius=6)
        self.pygame.draw.rect(self.screen, (58, 64, 72), self.camera_rect, border_radius=6)
        self._draw_text("R  Reset", (52, 369), font=self.small_font)
        self._draw_text("C  Camera {}".format(
            "ON" if self.simulation.follow_camera else "OFF"
        ), (180, 369), font=self.small_font)

        self._draw_text("MGDP depth (near = white, far = black)", (365, 20),
                        font=self.small_font)
        if self.simulation.latest_depth_rgb is None:
            self.pygame.draw.rect(self.screen, (0, 0, 0), (390, 65, 320, 320))
            if self.simulation.depth_display_error:
                self._draw_text("DEPTH CAPTURE ERROR", (455, 175),
                                color=(255, 105, 105), font=self.font)
                error_text = self.simulation.depth_display_error
                for line_number in range(3):
                    start = line_number * 38
                    line = error_text[start:start + 38]
                    if line:
                        self._draw_text(line, (405, 215 + line_number * 24),
                                        color=(235, 180, 180), font=self.small_font)
            else:
                self._draw_text("Waiting for depth image...", (445, 210),
                                font=self.small_font)
        else:
            depth_surface = self.pygame.surfarray.make_surface(
                self.simulation.latest_depth_rgb.transpose(1, 0, 2)
            )
            depth_surface = self.pygame.transform.scale(depth_surface, (320, 320))
            self.screen.blit(depth_surface, (390, 65))
            self._draw_text("LIVE  frame {} | t={:.2f}s | {:.2f}-{:.2f} m".format(
                self.simulation.depth_frame_id,
                self.simulation.data.time,
                self.simulation.latest_depth_min, self.simulation.latest_depth_max
            ), (420, 397), color=(120, 220, 160), font=self.small_font)
        self.pygame.display.flip()

    def _poll_input(self) -> np.ndarray:
        if not self.alive:
            return np.asarray([self.vx, self.vy, self.yaw], dtype=np.float32)

        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.alive = False
                return np.zeros(3, dtype=np.float32)
            if event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_r:
                    self.simulation.request_reset()
                elif event.key == self.pygame.K_c:
                    self.simulation.toggle_camera_follow()
                elif event.key == self.pygame.K_x:
                    self.vx = self.vy = self.yaw = 0.0
            if event.type == self.pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.reset_rect.collidepoint(event.pos):
                    self.simulation.request_reset()
                elif self.camera_rect.collidepoint(event.pos):
                    self.simulation.toggle_camera_follow()

        keys = self.pygame.key.get_pressed()
        mouse_down = self.pygame.mouse.get_pressed()[0]
        mouse_position = self.pygame.mouse.get_pos()
        pressed = {
            name: bool(keys[getattr(self.pygame, "K_{}".format(name.lower()))])
            or bool(mouse_down and rect.collidepoint(mouse_position))
            for name, rect in self.key_rects.items()
        }
        self.last_pressed = pressed

        target_vx = self.vx_scale if pressed["W"] else -self.vx_scale if pressed["S"] else 0.0
        target_vy = self.vy_scale if pressed["A"] else -self.vy_scale if pressed["D"] else 0.0
        target_yaw = (
            self.yaw_scale if pressed["Q"] else -self.yaw_scale if pressed["E"] else 0.0
        )
        if pressed["X"]:
            self.vx = self.vy = self.yaw = 0.0
            target_vx = target_vy = target_yaw = 0.0
        else:
            self.vx = self.smooth_update(self.vx, target_vx)
            self.vy = self.smooth_update(self.vy, target_vy)
            self.yaw = self.smooth_update(self.yaw, target_yaw)

        self.draw(pressed)
        return np.asarray([self.vx, self.vy, self.yaw], dtype=np.float32)

    def read(self) -> np.ndarray:
        """Poll input and repaint from the main thread, as required by SDL."""
        return self._poll_input()

    def refresh(self) -> None:
        """Repaint immediately after a new depth frame is captured."""
        if self.alive:
            self.draw(self.last_pressed)

    def close(self) -> None:
        self.alive = False
        self.pygame.display.quit()


def _replace_queue_item(output_queue, item) -> None:
    """Publish newest data without allowing a slow consumer to block a producer."""
    try:
        output_queue.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        output_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        output_queue.put_nowait(item)
    except queue.Full:
        pass


class _UISimulationProxy:
    """Small simulation-shaped object used by the isolated Pygame process."""

    def __init__(self, event_queue, initial_command: Sequence[float]) -> None:
        self.event_queue = event_queue
        self.commands = np.asarray(initial_command, dtype=np.float32)
        self.latest_depth_rgb = None
        self.depth_display_error = None
        self.latest_depth_min = 0.0
        self.latest_depth_max = 0.0
        self.depth_frame_id = 0
        self.follow_camera = True
        self.data_time = 0.0

        class _Data:
            time = 0.0

        self.data = _Data()

    def apply_state(self, state) -> None:
        (
            self.latest_depth_rgb,
            self.depth_display_error,
            self.latest_depth_min,
            self.latest_depth_max,
            self.depth_frame_id,
            self.data.time,
            self.follow_camera,
        ) = state

    def request_reset(self) -> None:
        self.event_queue.put_nowait("reset")

    def toggle_camera_follow(self) -> None:
        self.follow_camera = not self.follow_camera
        self.event_queue.put_nowait("camera")


def _control_ui_process(
    state_queue,
    command_queue,
    event_queue,
    stop_event,
    initial_command,
    vx_scale: float,
    vy_scale: float,
    yaw_scale: float,
    smooth: float,
) -> None:
    """Own SDL/Pygame in a process separate from MuJoCo's GLFW viewer."""
    # Let the main simulation process handle Ctrl+C and request an orderly UI
    # shutdown instead of printing a second traceback from this child.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    proxy = _UISimulationProxy(event_queue, initial_command)
    window = _PygameKeyboardWindow(
        proxy,
        vx_scale=vx_scale,
        vy_scale=vy_scale,
        yaw_scale=yaw_scale,
        smooth=smooth,
    )
    clock = window.pygame.time.Clock()
    try:
        while window.alive and not stop_event.is_set():
            try:
                while True:
                    proxy.apply_state(state_queue.get_nowait())
            except queue.Empty:
                pass
            command = window.read()
            _replace_queue_item(command_queue, tuple(float(value) for value in command))
            clock.tick(60)
    finally:
        window.close()


class KeyboardController:
    """Main-process bridge to an isolated Pygame control/depth window."""

    def __init__(
        self,
        simulation: "Go2Simulation",
        vx_scale: float = 1.0,
        vy_scale: float = 1.0,
        yaw_scale: float = 1.0,
        smooth: float = 0.2,
    ) -> None:
        self.simulation = simulation
        self.output_command = simulation.commands.astype(np.float32).copy()
        context = mp.get_context("spawn")
        self.state_queue = context.Queue(maxsize=2)
        self.command_queue = context.Queue(maxsize=2)
        self.event_queue = context.Queue(maxsize=16)
        self.stop_event = context.Event()
        self.process = context.Process(
            target=_control_ui_process,
            args=(
                self.state_queue,
                self.command_queue,
                self.event_queue,
                self.stop_event,
                tuple(float(value) for value in self.output_command),
                vx_scale,
                vy_scale,
                yaw_scale,
                smooth,
            ),
            name="go2-control-depth-ui",
            daemon=True,
        )
        self.process.start()
        self.refresh()

    @property
    def alive(self) -> bool:
        return self.process.is_alive()

    def read(self) -> np.ndarray:
        try:
            while True:
                self.output_command[:] = self.command_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            while True:
                event = self.event_queue.get_nowait()
                if event == "reset":
                    self.simulation.request_reset()
                elif event == "camera":
                    self.simulation.toggle_camera_follow()
        except queue.Empty:
            pass
        return self.output_command.copy()

    def refresh(self) -> None:
        depth = self.simulation.latest_depth_rgb
        state = (
            None if depth is None else depth.copy(),
            self.simulation.depth_display_error,
            self.simulation.latest_depth_min,
            self.simulation.latest_depth_max,
            self.simulation.depth_frame_id,
            float(self.simulation.data.time),
            self.simulation.follow_camera,
        )
        _replace_queue_item(self.state_queue, state)

    def close(self) -> None:
        self.stop_event.set()
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)


class Go2Simulation:
    PHYSICS_DT = 0.005
    CONTROL_DECIMATION = 4
    CAMERA_DECIMATION = 5  # Camera updates every five 50 Hz policy steps, as in Stage 1.
    DEPTH_DISPLAY_PERIOD = 0.01  # Keep the UI camera live at 10 Hz, including while settling.
    KP = 30.0
    KD = 0.8
    ACTION_SCALE = 0.25

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        model_path = Path(args.model).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = self.PHYSICS_DT
        self.data = mujoco.MjData(self.model)

        self.joint_qpos_ids = np.asarray([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )]
            for name in JOINT_NAMES
        ])
        self.joint_dof_ids = np.asarray([
            self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )]
            for name in JOINT_NAMES
        ])
        self.base_id = self._first_named_id(
            mujoco.mjtObj.mjOBJ_BODY, ("base_link", "base")
        )
        self.gyro_id = self._first_named_id(
            mujoco.mjtObj.mjOBJ_SENSOR, ("imu_gyro", "base_gyro")
        )
        self.gyro_adr = self.model.sensor_adr[self.gyro_id]

        self.policy = Stage1Policy(
            Path(args.checkpoint).expanduser().resolve(),
            Path(args.world_model).expanduser().resolve(),
            torch.device(args.device),
        )

        # The offscreen renderer is used only by the robot depth camera. Robot
        # geoms use group 2; showing group 0 alone matches Warp's
        # terrain-only rays.
        self.depth_renderer = mujoco.Renderer(self.model, height=120, width=120)
        self.depth_renderer.enable_depth_rendering()
        self.depth_scene_option = mujoco.MjvOption()
        self.depth_scene_option.geomgroup[:] = 0
        self.depth_scene_option.geomgroup[0] = 1

        self.viewer = None
        self.control_panel: Optional[KeyboardController] = None
        self.closed = False
        self.follow_camera = True
        self.reset_requested = False
        self.commands = np.asarray([args.vx, args.vy, args.yaw], dtype=np.float64)
        self.previous_action = np.zeros(12, dtype=np.float64)
        self.target_q = DEFAULT_DOF_POS.copy()
        self.observation_history: Optional[np.ndarray] = None
        self.depth_history: Optional[torch.Tensor] = None
        self.depth_token: Optional[torch.Tensor] = None
        self.latest_depth_rgb: Optional[np.ndarray] = None
        self.depth_display_error: Optional[str] = None
        self.latest_depth_min = 0.0
        self.latest_depth_max = 0.0
        self.depth_frame_id = 0
        self.depth_last_capture_time = -np.inf
        self.policy_tick = 0
        self.control_tick = 0

        self.reset()

    def _first_named_id(self, object_type: mujoco.mjtObj, names: Sequence[str]) -> int:
        for name in names:
            object_id = mujoco.mj_name2id(self.model, object_type, name)
            if object_id >= 0:
                return object_id
        raise ValueError("MuJoCo model has none of these required names: {}".format(
            ", ".join(names)
        ))

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = np.asarray([0.0, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.joint_qpos_ids] = DEFAULT_DOF_POS
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.previous_action[:] = 0.0
        self.target_q[:] = DEFAULT_DOF_POS
        self.observation_history = None
        self.depth_history = None
        self.depth_token = None
        self.policy.reset_memory()
        self.policy_tick = 0
        self.control_tick = 0
        self.depth_last_capture_time = -np.inf
        mujoco.mj_forward(self.model, self.data)

    def _joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.data.qpos[self.joint_qpos_ids].copy(),
            self.data.qvel[self.joint_dof_ids].copy(),
        )

    def _observation(self) -> np.ndarray:
        q, dq = self._joint_state()
        omega = self.data.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        rotation_world_from_body = self.data.xmat[self.base_id].reshape(3, 3)
        projected_gravity = rotation_world_from_body.T @ np.asarray([0.0, 0.0, -1.0])

        obs = np.concatenate((
            omega * 0.25,
            projected_gravity,
            self.commands * np.asarray([2.0, 2.0, 0.25]),
            q - DEFAULT_DOF_POS,
            dq * 0.05,
            self.previous_action,
        )).astype(np.float32)
        if obs.shape != (45,):
            raise RuntimeError("Expected a 45-value observation, got {}".format(obs.shape))
        return np.clip(obs, -100.0, 100.0)

    def _capture_depth(self) -> np.ndarray:
        """Render one depth frame and publish its grayscale UI representation."""
        self.depth_renderer.update_scene(
            self.data, camera="depth_camera", scene_option=self.depth_scene_option
        )
        depth = np.asarray(self.depth_renderer.render(), dtype=np.float32).copy()
        display_depth = np.nan_to_num(depth, nan=2.5, posinf=2.5, neginf=0.1)
        display_depth = np.clip(display_depth, 0.1, 2.5)
        proximity = 1.0 - (display_depth - 0.1) / 2.4
        grayscale = (np.clip(proximity, 0.0, 1.0) * 255.0).astype(np.uint8)
        self.latest_depth_rgb = np.repeat(grayscale[..., None], 3, axis=-1)
        self.depth_display_error = None
        self.latest_depth_min = float(display_depth.min())
        self.latest_depth_max = float(display_depth.max())
        self.depth_frame_id += 1
        self.depth_last_capture_time = float(self.data.time)
        if self.control_panel is not None:
            self.control_panel.refresh()
        if self.depth_frame_id == 1:
            print("Depth frame ready: {:.2f} m to {:.2f} m".format(
                self.latest_depth_min, self.latest_depth_max
            ))
        return depth

    def _update_depth(self) -> None:
        depth = self._capture_depth()
        depth_tensor = torch.as_tensor(depth, device=self.policy.device).view(1, 1, 120, 120)

        # Warp Stage 1: clamp to [0.1, 2.5], bilinear resize to 16x16,
        # optionally add sensor noise/dropout, and finally normalize by 5 m.
        depth_tensor.clamp_(0.1, 2.5)
        depth_tensor = F.interpolate(
            depth_tensor, size=(16, 16), mode="bilinear", align_corners=False
        )[0, 0]
        if self.args.depth_noise:
            depth_tensor.add_(torch.randn_like(depth_tensor) * 0.03)
            depth_tensor.mul_((torch.rand_like(depth_tensor) > 0.1).to(depth_tensor.dtype))
        depth_tensor = depth_tensor.clamp(0.0, 5.0) / 5.0

        if self.depth_history is None:
            self.depth_history = depth_tensor.repeat(2, 1, 1)
        else:
            self.depth_history = torch.cat(
                (self.depth_history[1:], depth_tensor.unsqueeze(0)), dim=0
            )
        self.depth_token = self.policy.encode_depth(self.depth_history.unsqueeze(0))

    def _policy_step(self) -> None:
        obs = self._observation()
        if self.observation_history is None:
            self.observation_history = np.tile(obs, 4)
        else:
            self.observation_history = np.concatenate((
                self.observation_history[45:], obs
            ))

        if self.policy_tick % self.CAMERA_DECIMATION == 0 or self.depth_token is None:
            self._update_depth()
        assert self.depth_token is not None

        action = self.policy.act(obs, self.observation_history, self.depth_token)
        action = np.clip(action, -self.args.action_clip, self.args.action_clip)
        self.previous_action[:] = action
        self.target_q[:] = DEFAULT_DOF_POS + self.ACTION_SCALE * action
        self.policy_tick += 1

    def _physics_step(self) -> None:
        q, dq = self._joint_state()
        torque = self.KP * (self.target_q - q) - self.KD * dq
        limits = self.model.actuator_ctrlrange
        self.data.ctrl[:] = np.clip(torque, limits[:, 0], limits[:, 1])
        mujoco.mj_step(self.model, self.data)

    def adjust_command(self, axis: int, amount: float) -> None:
        self.commands[axis] = np.clip(self.commands[axis] + amount, -1.0, 1.0)
        self._print_commands()

    def stop_commands(self) -> None:
        self.commands[:] = 0.0
        self._print_commands()

    def request_reset(self) -> None:
        self.reset_requested = True

    def toggle_camera_follow(self) -> None:
        self.follow_camera = not self.follow_camera
        print("camera follow: {}".format("on" if self.follow_camera else "off"))

    def _print_commands(self) -> None:
        print("command: vx={:+.2f}, vy={:+.2f}, yaw={:+.2f}".format(*self.commands))

    def _key_callback(self, keycode: int) -> None:
        if keycode == ord("W"):
            self.adjust_command(0, 0.1)
        elif keycode == ord("S"):
            self.adjust_command(0, -0.1)
        elif keycode == ord("A"):
            self.adjust_command(1, 0.1)
        elif keycode == ord("D"):
            self.adjust_command(1, -0.1)
        elif keycode == ord("Q"):
            self.adjust_command(2, 0.1)
        elif keycode == ord("E"):
            self.adjust_command(2, -0.1)
        elif keycode == ord("X"):
            self.stop_commands()
        elif keycode == ord("R"):
            self.request_reset()
        elif keycode == ord("C"):
            self.toggle_camera_follow()
        else:
            return

    def _viewer_alive(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def _configure_viewer_camera(self) -> None:
        """Set the initial view once, leaving orbit and zoom under mouse control."""
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.lookat[:] = self.data.xpos[self.base_id]
                self.viewer.cam.distance = 1.5
                self.viewer.cam.azimuth = 135.0
                self.viewer.cam.elevation = -20.0

    def _sync_viewer(self) -> None:
        # Capture independently from policy inference so the displayed sensor
        # remains live during stand settling and when the robot is stationary.
        if (
            self.control_panel is not None
            and self.control_panel.alive
            and self.data.time - self.depth_last_capture_time
            >= self.DEPTH_DISPLAY_PERIOD - 1e-9
        ):
            try:
                self._capture_depth()
            except Exception as error:
                self.depth_display_error = "{}: {}".format(type(error).__name__, error)

        # Poll and paint Pygame first; viewer.sync() can occasionally wait for
        # the window compositor on some Linux desktop configurations.
        if self.control_panel is not None:
            if self.control_panel.alive:
                self.commands[:] = self.control_panel.read()
            else:
                self.commands[:] = 0.0
                self.control_panel = None
        if self.viewer is not None:
            if self.follow_camera:
                with self.viewer.lock():
                    self.viewer.cam.lookat[:] = self.data.xpos[self.base_id]
            self.viewer.sync()

    def run(self) -> None:
        if not self.args.headless:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            )
            self._configure_viewer_camera()
            print("Keys: W/S forward, A/D lateral, Q/E yaw, X stop, R reset, C camera follow")
            if not self.args.no_control_ui:
                # Prime the camera before opening Pygame.  The very first
                # compositor-visible frame therefore already contains depth,
                # instead of a stale "Waiting" placeholder.
                try:
                    self._capture_depth()
                except Exception as error:
                    self.depth_display_error = "{}: {}".format(
                        type(error).__name__, error
                    )
                    print("Depth capture failed: {}".format(self.depth_display_error))
                try:
                    self.control_panel = KeyboardController(
                        self,
                        vx_scale=self.args.vx_scale,
                        vy_scale=self.args.vy_scale,
                        yaw_scale=self.args.yaw_scale,
                        smooth=self.args.command_smoothing,
                    )
                    print(
                        "Depth display opened in the separate "
                        "'Go2 Keyboard Control + Depth' window"
                    )
                except Exception as error:
                    print("Warning: could not open Go2 control panel: {}".format(error))

        # First let the PD controller settle the robot at its default pose.
        settle_steps = max(0, int(self.args.stand_time / self.PHYSICS_DT))
        for step in range(settle_steps):
            if not self._viewer_alive():
                return
            self._physics_step()
            if step % self.CONTROL_DECIMATION == 0:
                self._sync_viewer()
            if self.args.realtime:
                time.sleep(self.PHYSICS_DT)

        start_wall = time.monotonic()
        sim_start = self.data.time
        last_report = sim_start
        while self._viewer_alive():
            elapsed = self.data.time - sim_start
            if self.args.duration > 0 and elapsed >= self.args.duration:
                break
            if self.reset_requested:
                self.reset_requested = False
                self.reset()
                sim_start = self.data.time
                start_wall = time.monotonic()

            if self.control_tick % self.CONTROL_DECIMATION == 0:
                self._policy_step()
                self._sync_viewer()

            self._physics_step()
            self.control_tick += 1

            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("MuJoCo state became non-finite")

            if self.args.headless and self.data.time - last_report >= 1.0:
                base = self.data.xpos[self.base_id]
                print("t={:6.2f}s  base=({:+.2f}, {:+.2f}, {:.2f})".format(
                    elapsed, base[0], base[1], base[2]
                ))
                last_report = self.data.time

            if self.args.realtime:
                target_wall = start_wall + elapsed
                remaining = target_wall - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

        base = self.data.xpos[self.base_id]
        print("Finished at t={:.2f}s; base position=({:.3f}, {:.3f}, {:.3f})".format(
            self.data.time - sim_start, base[0], base[1], base[2]
        ))
        self.close()

    def close(self) -> None:
        """Close UI and OpenGL resources once, including after Ctrl+C."""
        if self.closed:
            return
        self.closed = True
        if self.control_panel is not None:
            self.control_panel.close()
        if self.viewer is not None:
            # Handle.close() only requests that MuJoCo's viewer thread exit;
            # it does not join that thread.  Wait for the internal Simulate
            # object to be released before another GL context is destroyed.
            simulate_ref = getattr(self.viewer, "_sim", None)
            self.viewer.close()
            close_deadline = time.monotonic() + 2.0
            while (
                simulate_ref is not None
                and simulate_ref() is not None
                and time.monotonic() < close_deadline
            ):
                time.sleep(0.01)
        # Close the off-screen renderer last.  With MuJoCo's GLFW backend,
        # destroying its GL context before the passive viewer can segfault
        # during viewer shutdown.
        self.depth_renderer.close()


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    stage_label: str = "Stage 1",
    default_run: Path = DEFAULT_RUN,
    default_model: Optional[Path] = None,
    training_model: Optional[Path] = None,
) -> argparse.Namespace:
    default_model = default_model or (SCRIPT_DIR / "go2/scene.xml")
    training_model = training_model or (SCRIPT_DIR / "go2/scene_training.xml")
    parser = argparse.ArgumentParser(
        description="Play an MGDP {} policy on a Unitree Go2 in MuJoCo".format(
            stage_label
        )
    )
    parser.add_argument("--checkpoint", default=str(default_run / "model_best.pt"),
                        help="{} model_*.pt or model_best.pt".format(stage_label))
    parser.add_argument("--world-model", default=str(default_run / "wm_best.pt"),
                        help="Matching {} wm_*.pt or wm_best.pt".format(stage_label))
    parser.add_argument("--model", default=str(default_model),
                        help="MuJoCo scene XML")
    parser.add_argument("--terrain", action="store_true",
                        help="Use the hurdle/stairs scene instead of the flat scene")
    parser.add_argument("--training-terrain", action="store_true",
                        help="Use the {} training-style terrain course".format(stage_label))
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    parser.add_argument("--vx", type=float, default=0.0, help="Initial forward command [m/s]")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral command [m/s]")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw-rate command [rad/s]")
    parser.add_argument("--vx-scale", type=float, default=1.0,
                        help="Forward/backward speed while W/S is held [m/s]")
    parser.add_argument("--vy-scale", type=float, default=1.0,
                        help="Lateral speed while A/D is held [m/s]")
    parser.add_argument("--yaw-scale", type=float, default=1.0,
                        help="Yaw rate while Q/E is held [rad/s]")
    parser.add_argument("--command-smoothing", type=float, default=0.2,
                        help="Keyboard command smoothing in (0, 1]; larger is faster")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Run time after standing; default 0 runs until the viewer closes")
    parser.add_argument("--stand-time", type=float, default=1.0,
                        help="Seconds to hold the default pose before policy control")
    parser.add_argument("--action-clip", type=float, default=100.0,
                        help="Policy action clip; 100 reproduces the training config")
    parser.add_argument("--depth-noise", action="store_true",
                        help="Enable the training depth Gaussian noise and dropout")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    parser.add_argument("--no-control-ui", action="store_true",
                        help="Do not open the separate Pygame control/depth window")
    parser.add_argument("--realtime", action="store_true",
                        help="Rate-limit physics to wall-clock time")
    args = parser.parse_args(argv)
    if not 0.0 < args.command_smoothing <= 1.0:
        parser.error("--command-smoothing must be in (0, 1]")
    if args.terrain and args.training_terrain:
        parser.error("choose only one of --terrain and --training-terrain")
    if args.terrain and args.model == str(default_model):
        args.model = str(SCRIPT_DIR / "go2/scene_terrain.xml")
    if args.training_terrain and args.model == str(default_model):
        args.model = str(training_model)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("{} requested, but CUDA is not available to PyTorch".format(args.device))
    return args


def main() -> None:
    args = parse_args()
    print("MuJoCo model: {}".format(Path(args.model).resolve()))
    print("Policy:       {}".format(Path(args.checkpoint).resolve()))
    print("World model:  {}".format(Path(args.world_model).resolve()))
    simulation = Go2Simulation(args)
    try:
        simulation.run()
    except KeyboardInterrupt:
        print("\nStopping Go2 simulation...")
    finally:
        simulation.close()


if __name__ == "__main__":
    main()
