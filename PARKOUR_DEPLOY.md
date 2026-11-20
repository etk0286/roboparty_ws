# BigRedDog parkour deployment

The deployment entry point is `isaaclab_parkour_deploy.py`; its parameters are
in `isaaclab_parkour_deploy.yaml`. It uses the same policy observation layout as
`sim2sim_bigreddog_parkour.py`:

1. Eight frames of angular velocity, projected gravity, command, joint
   position error, joint velocity, and previous action are grouped by term and
   flattened to 360 values.
2. RealSense depth is remapped to 48x64, cropped to 18x32, normalized, stored
   in a 37-frame buffer, and sampled at indices 1, 6, 11, 16, 21, 26, 31, 36.
   The buffer advances only when inference receives a new camera frame ID.
   Reusing the latest image does not insert duplicate history entries or
   compensate for missed frames. Startup padding still repeats the oldest
   frame to provide the encoder's fixed input shape. With slower camera input,
   this history covers more elapsed time than policy-rate sampling.
3. The depth encoder produces 128 values. These are concatenated with the 360
   proprioceptive values for the actor's 488-value input.
4. The actor's 12 policy-order actions are mapped back to SDK motor order and
   converted to position targets.

## Environment

```bash
conda activate roboparty_full
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash
cd ~/roboparty_ws
```

The Unitree SDK in this workspace uses CycloneDDS 0.10.2. If the environment
needs to be rebuilt:

```bash
export CYCLONEDDS_HOME=/home/spot/cyclonedds/install
python -m pip install cyclonedds==0.10.2 rich-click
python -m pip install --no-deps -e /home/spot/unitree_sdk2_python
```

## Safe validation (no DDS and no motor commands)

```bash
python -s isaaclab_parkour_deploy.py --check
```

This must end with `CHECK PASSED`.

Verify the depth stream separately:

```bash
ros2 topic info /camera0/realsense_splitter_node/output/depth
```

## Real robot

Before real deployment, physically verify these YAML values against the robot:

- `control.default_angles`
- `control.sdk_to_policy` and `control.policy_to_sdk`
- `safety.joint_lower` and `safety.joint_upper`
- the DDS network interface and domain

Keep the robot supported above the ground for the first test and keep an
independent hardware emergency stop within reach.

```bash
python -s isaaclab_parkour_deploy.py \
  --config isaaclab_parkour_deploy.yaml \
  --domain 1
```

This matches `isaaclab_deploy.py`: when no network argument is supplied, the
program uses `dds.interface: lo` from the YAML. To override it, place the
interface name at the end, for example:

```bash
python -s isaaclab_parkour_deploy.py --domain 1 eth0
```

The program does not initialize the DDS publisher unless `DEPLOY` is typed
exactly. It starts in damping mode. Console commands are:

```text
stand
status
move
cmd 0.2 0.0 0.0
cmd 0.0 0.0 0.0
damp
sit
exit
```

`move` is rejected until the stand transition is complete and a fresh depth
frame is available. A stale LowState, stale depth frame, excessive tilt,
non-finite state, or inference failure switches the controller to damping.

Keyboard and `cmd` inputs set a target command. The command supplied to the
policy ramps toward it at the YAML rates in `command_ramp_per_s`; releasing a
key therefore ramps smoothly back to zero instead of changing instantly.

## Session recordings

`logging.enabled` enables a UTC-named session directory under `logging.directory`.
Relative paths resolve against the launch directory; the supplied YAML uses an
absolute path. Recording covers every control mode, including the interval before
`move` and the damping interval during shutdown. Hardware LowCmd packets are unchanged.

Schema version 2 replaces `commands.jsonl` / `config.json` with:

- `metadata.json`: full effective config, joint maps, units, preprocessing,
  model SHA-256 hashes, deployment source hash, runtime versions/providers and
  host clock anchor.
- `control.jsonl`: each DDS write attempt, its start/end host timestamps,
  `control_sequence`, mode, actual applied `policy_step_id`, 12 motor targets and
  effective gains/modes/feedforward values, and latest feedback with receive time
  and state ID. A null step means no policy action supplied that command.
  `write_result` reports the SDK result, not motor acknowledgement.
- `policy.jsonl`: inference start/end, input state snapshot/ID, clipped user
  velocity command, raw-history and selected depth IDs (oldest to newest), processed
  action, requested/final joint targets and `accepted` status. An in-flight result
  invalidated by a mode change is saved with `accepted: false`.
- `events.jsonl`: mode changes/fault reasons, policy resets/errors/discards,
  depth frame metadata, DDS write errors, queue losses and shutdown.
- `tensors/policy_XXXXXXXX.npz`: exact `encoder_input`, `actor_input`,
  `proprio_history`, `latent`, and unclipped `raw_action` arrays for each step.
- `tensors/depth_XXXXXXXX.npy`: normalized float32 frame, stored once per used
  camera frame ID. Frame events include ROS source stamp/frame name, encoding,
  host receive monotonic time and Unix time. Raw camera depth and PNG previews
  are not recorded.
- `summary.json`: submitted/written/dropped/failed totals across streams.

`sequence` is a session-wide queue sequence; `control_sequence` counts write
attempts independently. Join control and policy on `policy_step_id`; repeated IDs
mean the control loop reused the same policy target. Always check queue-loss and
failure totals before assuming every referenced policy record is present.
Disk I/O runs in a bounded background queue; a disk failure is reported to stderr
and, when possible, events. A full or unwritable disk cannot guarantee a saved event.

Replay the encoder using `encoder_input` and the actor using `actor_input` from
the NPZ. The actor input already includes the exact proprio history and encoder
latent. Compare the actor output with `raw_action`; `action` and `q_target` include
subsequent deployment clipping/scaling. The model contract is stateless; policy
reset events identify history initialization. At the first inference, histories
are padded using the first input, exactly as in the live policy; there is no
pre-move inference warm-up. The complete padded input is saved.

ROS header time is a separate source clock with no calibrated conversion to host
monotonic time. The recorded receive pairs permit latency analysis only when the
source clock relationship is independently known. `tau_est` is SDK-estimated
motor torque; `foot_force` is the SDK's raw measured channel (units/calibration
unverified). Missing optional feedback is null. Base velocity estimates and
inferred contact states are not available from this deployment path.

```python
import json
from pathlib import Path
import numpy as np

run = Path("YOUR_SESSION_DIRECTORY")
step = json.loads((run / "policy.jsonl").read_text().splitlines()[0])
with np.load(run / step["tensor_file"], allow_pickle=False) as tensors:
    encoder_input = tensors["encoder_input"]
    actor_input = tensors["actor_input"]
    expected_action = tensors["raw_action"]
```

## Checking the DDS workflow

An independent monitor checks the sending thread and received LowState activity
every 100 ms. With recording enabled it writes `dds_health.jsonl` in the run
directory once per second and whenever the diagnostic status changes. Status
changes also appear in the terminal, and the console `status` command includes
DDS status, measured LowState frequency, and cumulative write failures.

The monitor records received feedback frequency and age, completed write-attempt
frequency, maximum observed receive/send gaps, SDK failures, callback errors,
control-thread errors, inference-thread activity, depth age, and logger losses.
Receive timestamps are local callback times, so these observations do not measure
network latency or exact packet loss. A camera issue is separate from a DDS issue.

| Diagnostic | Meaning |
| --- | --- |
| `local_io_ok` | Fresh LowState and the most recent local SDK write returned true; this is not a robot acknowledgement. |
| `write_unconfirmed` | No explicit successful SDK write status is available. |
| `lowstate_missing_or_stale` | No feedback, or feedback older than `safety.low_state_stale_timeout` (currently 0.25 s). |
| `write_failed` | SDK returned false or raised an exception. |
| `write_blocked` | A write has not returned for `max(0.25, 5 * control.dt)` seconds. |
| `publishing_stalled` | No write has completed within that same interval. |
| `control_thread_stopped` | Sender thread exited; inspect `control_error` and terminal traceback. |

A false or exceptional SDK write switches the controller to damping and continues
attempting damping commands at the configured rate. It does not automatically
resume motion after recovery. If the transport is broken, damping delivery cannot
be guaranteed. An unexpected error outside the write is recorded and reported;
the monitor remains active even if the sender exits. It cannot unblock a hung SDK
call or recover a native crash of the whole process.

To inspect a run without publishing commands:

```bash
tail -f record/lowcmd/YOUR_RUN_DIRECTORY/dds_health.jsonl
```

For the next intended robot operation, capture terminal errors and enable Python
fatal-error tracebacks as well (this command starts the normal motor controller):

```bash
mkdir -p record
set -o pipefail
python -u -X faulthandler isaaclab_parkour_deploy.py --console 2>&1 | tee record/operation_console.log
```

This terminal file is overwritten on each invocation. It can contain SDK errors
or native fatal-error tracebacks that cannot reach the background log after a
process crash. An abrupt end of a health log alone does not identify a DDS crash.
Correlate its last timestamps with the terminal output and OS/network diagnostics.
Robot-side instrumentation is required to establish command reception/execution;
continued LowState reception alone is not a per-command acknowledgement.
