# BigRedDog parkour deployment

The deployment entry point is `isaaclab_parkour_deploy.py`; its parameters are
in `isaaclab_parkour_deploy.yaml`. It uses the same policy observation layout as
`sim2sim_bigreddog_parkour.py`:

1. Eight frames of angular velocity, projected gravity, command, joint
   position error, joint velocity, and previous action are grouped by term and
   flattened to 360 values.
2. RealSense depth is remapped to 48x64, cropped to 18x32, normalized, stored
   in a 37-frame buffer, and sampled at indices 1, 6, 11, 16, 21, 26, 31, 36.
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
