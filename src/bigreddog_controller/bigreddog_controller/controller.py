from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class DepthEncoderTester(Node):

    TRAIN_WIDTH = 64
    TRAIN_HEIGHT = 48

    ENCODER_HEIGHT = 18
    ENCODER_WIDTH = 32

    TRAIN_HFOV_DEG = 89.51
    TRAIN_VFOV_DEG = 58.29

    DEPTH_MIN = 0.0
    DEPTH_MAX = 2.5

    HISTORY_LENGTH = 37
    HISTORY_INDICES = (1, 6, 11, 16, 21, 26, 31, 36)

    def __init__(self):
        super().__init__("bigreddog_depth_encoder_tester")

        self.declare_parameter("depth_encoder_path", "")
        self.declare_parameter(
            "processed_depth_topic",
            "/bigreddog/depth_processed",
        )
        self.declare_parameter(
            "processed_depth_meters_topic",
            "/bigreddog/depth_processed_meters",
        )
        self.declare_parameter(
            "processed_depth_visualization_topic",
            "/bigreddog/depth_processed_visualization",
        )
        self.declare_parameter(
            "depth_comparison_visualization_topic",
            "/bigreddog/depth_comparison_visualization",
        )
        self.declare_parameter(
            "original_depth_grayscale_topic",
            "/bigreddog/depth_original_grayscale",
        )
        self.declare_parameter("visualization_scale", 15)

        depth_encoder_path = (
            self.get_parameter("depth_encoder_path")
            .get_parameter_value()
            .string_value
        )

        if not depth_encoder_path:
            raise RuntimeError(
                "The depth_encoder_path parameter is empty"
            )

        depth_encoder_path = Path(depth_encoder_path)

        if not depth_encoder_path.is_file():
            raise FileNotFoundError(
                f"Depth encoder not found: {depth_encoder_path}"
            )

        self.bridge = CvBridge()

        self.depth_remap = None
        self.camera_info_signature = None

        self.depth_history = deque(
            maxlen=self.HISTORY_LENGTH
        )

        self.previous_latent = None
        self.inference_count = 0

        self._load_depth_encoder(depth_encoder_path)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera0/depth/camera_info",
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/camera0/realsense_splitter_node/output/depth",
            self.depth_callback,
            qos_profile_sensor_data,
        )

        processed_depth_topic = self.get_parameter(
            "processed_depth_topic"
        ).value
        self.processed_depth_pub = self.create_publisher(
            Image,
            processed_depth_topic,
            qos_profile_sensor_data,
        )
        processed_depth_meters_topic = self.get_parameter(
            "processed_depth_meters_topic"
        ).value
        self.processed_depth_meters_pub = self.create_publisher(
            Image,
            processed_depth_meters_topic,
            qos_profile_sensor_data,
        )
        visualization_topic = self.get_parameter(
            "processed_depth_visualization_topic"
        ).value
        self.visualization_scale = int(
            self.get_parameter("visualization_scale").value
        )
        if self.visualization_scale < 1:
            raise ValueError("visualization_scale must be at least 1")
        self.processed_depth_visualization_pub = self.create_publisher(
            Image,
            visualization_topic,
            qos_profile_sensor_data,
        )
        comparison_topic = self.get_parameter(
            "depth_comparison_visualization_topic"
        ).value
        self.depth_comparison_visualization_pub = self.create_publisher(
            Image,
            comparison_topic,
            qos_profile_sensor_data,
        )
        original_grayscale_topic = self.get_parameter(
            "original_depth_grayscale_topic"
        ).value
        self.original_depth_grayscale_pub = self.create_publisher(
            Image,
            original_grayscale_topic,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "Depth encoder tester started; publishing normalized "
            f"18x32 depth to {processed_depth_topic} and metres to "
            f"{processed_depth_meters_topic}; color visualization to "
            f"{visualization_topic}; comparison to {comparison_topic}; "
            f"original grayscale to {original_grayscale_topic}"
        )

    def _load_depth_encoder(
        self,
        depth_encoder_path: Path,
    ):
        """Load the depth-encoder ONNX model."""

        available_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in available_providers:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = [
                "CPUExecutionProvider",
            ]

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3

        self.depth_encoder = ort.InferenceSession(
            str(depth_encoder_path),
            sess_options=session_options,
            providers=providers,
        )

        encoder_input = self.depth_encoder.get_inputs()[0]
        encoder_output = self.depth_encoder.get_outputs()[0]

        self.encoder_input_name = encoder_input.name
        self.encoder_output_name = encoder_output.name

        self.get_logger().info(
            "Depth encoder loaded: "
            f"input={encoder_input.name} "
            f"{encoder_input.shape} "
            f"{encoder_input.type}, "
            f"output={encoder_output.name} "
            f"{encoder_output.shape}, "
            f"providers={self.depth_encoder.get_providers()}"
        )

    def camera_info_callback(
        self,
        msg: CameraInfo,
    ):
        """Create the RealSense-to-training-camera mapping."""

        if msg.width <= 0 or msg.height <= 0:
            self.get_logger().error(
                "CameraInfo has an invalid resolution"
            )
            return

        if msg.p[0] <= 0.0 or msg.p[5] <= 0.0:
            self.get_logger().error(
                "CameraInfo.P has invalid fx/fy values"
            )
            return

        signature = (
            msg.width,
            msg.height,
            msg.header.frame_id,
            tuple(msg.p),
        )

        if signature == self.camera_info_signature:
            return

        map_x, map_y = self.create_train_depth_remap(msg)

        # Update all remap information together.
        self.depth_remap = (
            map_x,
            map_y,
            msg.width,
            msg.height,
            msg.header.frame_id,
        )

        self.camera_info_signature = signature

        self.get_logger().info(
            "CameraInfo received: "
            f"resolution={msg.width}x{msg.height}, "
            f"frame={msg.header.frame_id}, "
            f"fx={msg.p[0]:.2f}, "
            f"fy={msg.p[5]:.2f}"
        )

    def depth_callback(
        self,
        msg: Image,
    ):
        """Process one depth frame and run the depth encoder."""

        remap = self.depth_remap

        # Ignore depth images until CameraInfo is available.
        if remap is None:
            return

        (
            map_x,
            map_y,
            expected_width,
            expected_height,
            expected_frame,
        ) = remap

        if (
            msg.width != expected_width
            or msg.height != expected_height
        ):
            self.get_logger().error(
                "Depth and CameraInfo resolution mismatch: "
                f"depth={msg.width}x{msg.height}, "
                f"camera_info="
                f"{expected_width}x{expected_height}"
            )
            return

        if msg.header.frame_id != expected_frame:
            self.get_logger().error(
                "Depth and CameraInfo frame mismatch: "
                f"depth={msg.header.frame_id}, "
                f"camera_info={expected_frame}"
            )
            return

        depth_raw = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="passthrough",
        )

        raw_zero_count = int(
            np.count_nonzero(depth_raw == 0)
        )

        if msg.encoding in ("16UC1", "mono16"):
            # RealSense Z16 depth is normally in millimeters.
            depth_meter = (
                depth_raw.astype(np.float32) * 0.001
            )

        elif msg.encoding == "32FC1":
            # A 32FC1 depth image is normally already in meters.
            depth_meter = depth_raw.astype(np.float32)

        else:
            self.get_logger().error(
                f"Unsupported depth encoding: {msg.encoding}"
            )
            return

        # Treat invalid measurements as the maximum policy depth.
        invalid_mask = (
            ~np.isfinite(depth_meter)
            | (depth_meter <= 0.0)
        )

        invalid_count = int(
            np.count_nonzero(invalid_mask)
        )

        depth_meter[invalid_mask] = self.DEPTH_MAX

        # Remap RealSense rays to the 48x64 training camera.
        depth_48x64 = cv2.remap(
            depth_meter,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.DEPTH_MAX,
        )

        if depth_48x64.shape != (
            self.TRAIN_HEIGHT,
            self.TRAIN_WIDTH,
        ):
            self.get_logger().error(
                "Unexpected remapped depth shape: "
                f"{depth_48x64.shape}"
            )
            return

        # Keep the bottom 18 rows and center 32 columns.
        depth_18x32 = depth_48x64[
            30:48,
            16:48,
        ]

        # Match the Gaussian blur used during training.
        depth_18x32 = cv2.GaussianBlur(
            depth_18x32,
            (3, 3),
            sigmaX=1.0,
        )

        # Clip to [0, 2.5] meters and normalize to [0, 1].
        depth_processed_meters = np.clip(
            depth_18x32,
            self.DEPTH_MIN,
            self.DEPTH_MAX,
        ).astype(np.float32)

        depth_processed = (
            depth_processed_meters / self.DEPTH_MAX
        ).astype(np.float32)

        meters_msg = self.bridge.cv2_to_imgmsg(
            depth_processed_meters,
            encoding="32FC1",
        )
        meters_msg.header = msg.header
        self.processed_depth_meters_pub.publish(meters_msg)

        # Publish the exact current frame supplied to the history buffer.
        # Encoding is 32FC1 and values are normalized to [0, 1].
        processed_msg = self.bridge.cv2_to_imgmsg(
            depth_processed,
            encoding="32FC1",
        )
        processed_msg.header = msg.header
        self.processed_depth_pub.publish(processed_msg)

        # Publish an enlarged BGR image for RViz/rqt_image_view. This is
        # visualization only; the encoder continues to receive 32FC1 data.
        visualization_u8 = np.rint(
            np.clip(depth_processed, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        visualization_bgr = cv2.applyColorMap(
            visualization_u8,
            cv2.COLORMAP_TURBO,
        )
        if self.visualization_scale > 1:
            visualization_bgr = cv2.resize(
                visualization_bgr,
                None,
                fx=self.visualization_scale,
                fy=self.visualization_scale,
                interpolation=cv2.INTER_NEAREST,
            )
        visualization_msg = self.bridge.cv2_to_imgmsg(
            visualization_bgr,
            encoding="bgr8",
        )
        visualization_msg.header = msg.header
        self.processed_depth_visualization_pub.publish(
            visualization_msg
        )

        # Create one side-by-side view. The green polygon on the input is
        # the source-image footprint of training rows 30:48, columns 16:48.
        input_u8 = np.rint(
            np.clip(depth_meter / self.DEPTH_MAX, 0.0, 1.0)
            * 255.0
        ).astype(np.uint8)
        input_u8[invalid_mask] = 0

        original_grayscale_msg = self.bridge.cv2_to_imgmsg(
            input_u8,
            encoding="mono8",
        )
        original_grayscale_msg.header = msg.header
        self.original_depth_grayscale_pub.publish(
            original_grayscale_msg
        )

        input_bgr = cv2.applyColorMap(input_u8, cv2.COLORMAP_TURBO)

        crop_points = np.array(
            [
                [map_x[30, 16], map_y[30, 16]],
                [map_x[30, 47], map_y[30, 47]],
                [map_x[47, 47], map_y[47, 47]],
                [map_x[47, 16], map_y[47, 16]],
            ],
            dtype=np.float32,
        )
        crop_points[:, 0] = np.clip(
            crop_points[:, 0], 0, input_bgr.shape[1] - 1
        )
        crop_points[:, 1] = np.clip(
            crop_points[:, 1], 0, input_bgr.shape[0] - 1
        )
        crop_points = np.rint(crop_points).astype(np.int32)
        cv2.polylines(
            input_bgr,
            [crop_points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )

        # Direct-resize reference: for a 640x480 input this is exactly
        # columns 160:480 and rows 300:480.
        direct_x0 = int(round(input_bgr.shape[1] * 16 / 64))
        direct_x1 = int(round(input_bgr.shape[1] * 48 / 64)) - 1
        direct_y0 = int(round(input_bgr.shape[0] * 30 / 48))
        direct_y1 = input_bgr.shape[0] - 1
        cv2.rectangle(
            input_bgr,
            (direct_x0, direct_y0),
            (direct_x1, direct_y1),
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        input_height, input_width = input_bgr.shape[:2]
        processed_panel = np.zeros_like(input_bgr)
        processed_height = min(
            input_height,
            int(input_width * self.ENCODER_HEIGHT / self.ENCODER_WIDTH),
        )
        processed_width = int(
            processed_height * self.ENCODER_WIDTH / self.ENCODER_HEIGHT
        )
        processed_for_panel = cv2.resize(
            cv2.applyColorMap(
                visualization_u8,
                cv2.COLORMAP_TURBO,
            ),
            (processed_width, processed_height),
            interpolation=cv2.INTER_NEAREST,
        )
        panel_x = (input_width - processed_width) // 2
        panel_y = (input_height - processed_height) // 2
        processed_panel[
            panel_y:panel_y + processed_height,
            panel_x:panel_x + processed_width,
        ] = processed_for_panel

        comparison_bgr = np.hstack((input_bgr, processed_panel))
        cv2.putText(
            comparison_bgr,
            "INPUT: GREEN=ACTUAL, YELLOW=DIRECT CROP",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            comparison_bgr,
            "PROCESSED 18x32 NORMALIZED",
            (input_width + 12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        comparison_msg = self.bridge.cv2_to_imgmsg(
            comparison_bgr,
            encoding="bgr8",
        )
        comparison_msg.header = msg.header
        self.depth_comparison_visualization_pub.publish(comparison_msg)

        depth_bchw = self.update_depth_history(
            depth_processed
        )

        depth_latent = self.run_depth_encoder(
            depth_bchw
        )

        self.inference_count += 1

        if self.previous_latent is None:
            latent_delta_l2 = 0.0
        else:
            latent_delta_l2 = float(
                np.linalg.norm(
                    depth_latent
                    - self.previous_latent
                )
            )

        self.previous_latent = depth_latent.copy()

        # Print one summary approximately every 30 depth frames.
        if self.inference_count % 30 == 0:
            self.get_logger().info(
                "Depth encoder: "
                f"encoding={msg.encoding}, "
                f"input_shape={depth_bchw.shape}, "
                f"input_range="
                f"[{depth_bchw.min():.4f}, "
                f"{depth_bchw.max():.4f}], "
                f"latent_shape={depth_latent.shape}, "
                f"latent_range="
                f"[{depth_latent.min():.4f}, "
                f"{depth_latent.max():.4f}], "
                f"latent_mean={depth_latent.mean():.4f}, "
                f"latent_std={depth_latent.std():.4f}, "
                f"latent_delta_l2={latent_delta_l2:.4f}, "
                f"raw_zero={raw_zero_count}, "
                f"invalid={invalid_count}"
            )

    def update_depth_history(
        self,
        depth_frame: np.ndarray,
    ) -> np.ndarray:
        """Build a (1, 8, 18, 32) depth-encoder input."""

        if depth_frame.shape != (
            self.ENCODER_HEIGHT,
            self.ENCODER_WIDTH,
        ):
            raise ValueError(
                "Unexpected processed-depth shape: "
                f"{depth_frame.shape}"
            )

        self.depth_history.append(
            depth_frame.astype(np.float32).copy()
        )

        history = list(self.depth_history)

        # Match sim2sim startup behavior by padding with
        # the oldest available depth frame.
        if len(history) < self.HISTORY_LENGTH:
            history = (
                [history[0]]
                * (self.HISTORY_LENGTH - len(history))
                + history
            )

        history = history[-self.HISTORY_LENGTH:]

        depth_37hw = np.stack(
            history,
            axis=0,
        )

        depth_8chw = depth_37hw[
            list(self.HISTORY_INDICES)
        ]

        depth_bchw = np.ascontiguousarray(
            depth_8chw[None, ...],
            dtype=np.float32,
        )

        if depth_bchw.shape != (
            1,
            8,
            self.ENCODER_HEIGHT,
            self.ENCODER_WIDTH,
        ):
            raise ValueError(
                "Unexpected depth-encoder input shape: "
                f"{depth_bchw.shape}"
            )

        return depth_bchw

    def run_depth_encoder(
        self,
        depth_bchw: np.ndarray,
    ) -> np.ndarray:
        """Run one depth-encoder ONNX inference."""

        encoder_outputs = self.depth_encoder.run(
            [self.encoder_output_name],
            {
                self.encoder_input_name: depth_bchw,
            },
        )

        depth_latent = np.asarray(
            encoder_outputs[0],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(depth_latent)):
            raise ValueError(
                "Depth encoder produced NaN or infinity"
            )

        return depth_latent

    @classmethod
    def create_train_depth_remap(
        cls,
        camera_info: CameraInfo,
    ):
        """Create a rectified RealSense-to-training-camera map."""

        fx_rs = float(camera_info.p[0])
        fy_rs = float(camera_info.p[5])
        cx_rs = float(camera_info.p[2])
        cy_rs = float(camera_info.p[6])

        fx_train = (
            cls.TRAIN_WIDTH / 2.0
        ) / np.tan(
            np.deg2rad(cls.TRAIN_HFOV_DEG / 2.0)
        )

        fy_train = (
            cls.TRAIN_HEIGHT / 2.0
        ) / np.tan(
            np.deg2rad(cls.TRAIN_VFOV_DEG / 2.0)
        )

        cx_train = (cls.TRAIN_WIDTH - 1) / 2.0
        cy_train = (cls.TRAIN_HEIGHT - 1) / 2.0

        u_train, v_train = np.meshgrid(
            np.arange(
                cls.TRAIN_WIDTH,
                dtype=np.float32,
            ),
            np.arange(
                cls.TRAIN_HEIGHT,
                dtype=np.float32,
            ),
        )

        # Compute one normalized optical ray per training pixel.
        ray_x = (
            u_train - cx_train
        ) / fx_train

        ray_y = (
            v_train - cy_train
        ) / fy_train

        # Project the training rays into the RealSense image.
        map_x = fx_rs * ray_x + cx_rs
        map_y = fy_rs * ray_y + cy_rs

        return (
            map_x.astype(np.float32),
            map_y.astype(np.float32),
        )


def main(args=None):
    rclpy.init(args=args)

    node = DepthEncoderTester()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
