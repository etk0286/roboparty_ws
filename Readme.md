### Open docker and run realsense

run realsense
```bash
cd /mnt/nova_ssd/workspaces/isaac_ros-dev

cd $ISAAC_ROS_WS && ./src/isaac_ros_common/scripts/run_dev.sh

cd /workspaces/isaac_ros-dev

colcon build --symlink-install --packages-up-to realsense_splitter
source install/setup.bash

ros2 launch nvblox_examples_bringup realsense_example.launch.py
```

create subscriber and test the image into depth encoder
```bash
cd /home/spot/roboparty_ws

python /home/spot/roboparty_ws/src/bigreddog_controller/bigreddog_controller/controller.py --ros-args -p depth_encoder_path:=/home/spot/roboparty_ws/0-depth_encoder.onnx
```

display the subscribed depth image
```bash
cd /home/spot/roboparty_ws
colcon build --symlink-install --packages-select bigreddog_controller
source install/setup.bash
ros2 run bigreddog_controller depth_viewer
```

To display the camera's raw rectified depth topic instead:
```bash
ros2 run bigreddog_controller depth_viewer --ros-args \
  -p depth_topic:=/camera0/depth/image_rect_raw \
  -p min_depth_m:=0.3 \
  -p max_depth_m:=3.0
```

view the exact preprocessed frame sent to the encoder
```bash
ros2 run bigreddog_controller depth_encoder_tester --ros-args \
  -p depth_encoder_path:=/home/spot/roboparty_ws/0-depth_encoder.onnx
```

In RViz, add an `Image` display and select
`/bigreddog/depth_processed`. The image is `32FC1`, has shape 18x32,
and is normalized to the range 0.0 to 1.0. Set `Normalize Range` to
false, `Min Value` to 0.0, and `Max Value` to 1.0.

If RViz does not render the `32FC1` image, select the ready-to-display
color topic `/bigreddog/depth_processed_visualization` instead. It is
published as `bgr8` at 480x270 by default, so RViz normalization settings
are not needed. Set the Image display's reliability to `Best Effort`.

To display the preprocessed image with the standalone depth viewer:
```bash
ros2 run bigreddog_controller depth_viewer --ros-args \
  -p depth_topic:=/bigreddog/depth_processed_meters \
  -p min_depth_m:=0.1 \
  -p max_depth_m:=2.5 \
  -p display_scale:=15
```

view without rviz
```bash
ros2 run rqt_image_view rqt_image_view
```

### Record image data
```bash
ros2 bag record \
  --compression-mode file \
  --compression-format zstd \
  /camera0/realsense_splitter_node/output/depth \
  /bigreddog/depth_original_grayscale \
  /bigreddog/depth_processed
```

```bash
ros2 bag play camera0_depth_bag --loop
```



deploy


realsense

```bash

cd /mnt/nova_ssd/workspaces/isaac_ros-dev

cd $ISAAC_ROS_WS && ./src/isaac_ros_common/scripts/run_dev.sh

cd /workspaces/isaac_ros-dev

# colcon build --symlink-install --packages-up-to realsense_splitter
source install/setup.bash

ros2 launch nvblox_examples_bringup realsense_example.launch.py

```




offlime docker
```bash

# 主機端：使用本機映像，跳過遠端檢查與建置
export ISAAC_ROS_WS=/mnt/nova_ssd/workspaces/isaac_ros-dev

cd "$ISAAC_ROS_WS"
./src/isaac_ros_common/scripts/run_dev.sh -b -d "$ISAAC_ROS_WS"


cd /workspaces/isaac_ros-dev
source install/setup.bash
ros2 launch nvblox_examples_bringup realsense_example.launch.py




```
```bash

conda activate roboparty_full

cd roboparty_ws
python isaaclab_parkour_deploy.py 

```