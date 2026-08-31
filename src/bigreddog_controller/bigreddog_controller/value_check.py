import tkinter as tk
from PIL import Image as PILImage
from PIL import ImageTk

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge


DEFAULT_DEPTH_TOPIC = "/camera0/realsense_splitter_node/output/depth"
DEFAULT_MIN_DEPTH_M = 0.1
DEFAULT_MAX_DEPTH_M = 2.5


class DepthViewer(Node):

    def __init__(self):
        super().__init__("depth_viewer")

        self.declare_parameter("depth_topic", DEFAULT_DEPTH_TOPIC)
        self.declare_parameter("min_depth_m", DEFAULT_MIN_DEPTH_M)
        self.declare_parameter("max_depth_m", DEFAULT_MAX_DEPTH_M)
        self.declare_parameter("display_scale", 1)

        depth_topic = self.get_parameter("depth_topic").value
        self.min_depth_m = float(
            self.get_parameter("min_depth_m").value
        )
        self.max_depth_m = float(
            self.get_parameter("max_depth_m").value
        )
        self.display_scale = int(
            self.get_parameter("display_scale").value
        )
        if self.min_depth_m < 0.0:
            raise ValueError("min_depth_m cannot be negative")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError(
                "max_depth_m must be greater than min_depth_m"
            )
        if self.display_scale < 1:
            raise ValueError("display_scale must be at least 1")

        self.bridge = CvBridge()
        self.depth_m = None

        self.selected_x = None
        self.selected_y = None

        self.depth_subscription = self.create_subscription(
            ImageMsg,
            depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Displaying {depth_topic} from "
            f"{self.min_depth_m:.2f} to {self.max_depth_m:.2f} m"
        )

        # ---------------- Tkinter ----------------

        self.root = tk.Tk()
        self.root.title("Depth Viewer")

        self.canvas = tk.Canvas(
            self.root,
            width=640,
            height=480,
        )
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_click)

        self.info = tk.Label(
            self.root,
            text="Click on image",
            font=("Consolas", 13),
            justify="left",
        )
        self.info.pack()

        self.photo = None

        self.root.after(30, self.update_gui)

    def depth_callback(self, msg):
        depth_raw = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="passthrough",
        )

        if msg.encoding in ("16UC1", "mono16"):
            self.depth_m = depth_raw.astype(np.float32) * 0.001
        elif msg.encoding == "32FC1":
            self.depth_m = depth_raw.astype(np.float32)
        else:
            self.get_logger().error(
                f"Unsupported depth encoding: {msg.encoding}"
            )
            return

        height, width = self.depth_m.shape
        self.canvas.config(
            width=width * self.display_scale,
            height=height * self.display_scale,
        )

    def on_click(self, event):
        self.selected_x = event.x // self.display_scale
        self.selected_y = event.y // self.display_scale

    def update_gui(self):

        if self.depth_m is not None:

            # ----- Display image -----

            valid = (
                np.isfinite(self.depth_m)
                & (self.depth_m >= self.min_depth_m)
                & (self.depth_m <= self.max_depth_m)
            )
            display = np.zeros(self.depth_m.shape, dtype=np.uint8)
            display[valid] = (
                (self.depth_m[valid] - self.min_depth_m)
                / (self.max_depth_m - self.min_depth_m)
                * 255.0
            ).astype(np.uint8)

            image = PILImage.fromarray(display)
            if self.display_scale > 1:
                image = image.resize(
                    (
                        image.width * self.display_scale,
                        image.height * self.display_scale,
                    ),
                    resample=PILImage.NEAREST,
                )

            self.photo = ImageTk.PhotoImage(image=image)

            self.canvas.delete("all")
            self.canvas.create_image(
                0,
                0,
                anchor="nw",
                image=self.photo,
            )

            # ----- Crosshair -----

            if self.selected_x is not None:

                x = self.selected_x
                y = self.selected_y
                canvas_x = int((x + 0.5) * self.display_scale)
                canvas_y = int((y + 0.5) * self.display_scale)

                self.canvas.create_line(
                    canvas_x - 10, canvas_y,
                    canvas_x + 10, canvas_y,
                    fill="red",
                    width=2,
                )

                self.canvas.create_line(
                    canvas_x, canvas_y - 10,
                    canvas_x, canvas_y + 10,
                    fill="red",
                    width=2,
                )

                if (
                    0 <= x < self.depth_m.shape[1]
                    and
                    0 <= y < self.depth_m.shape[0]
                ):

                    depth = float(self.depth_m[y, x])

                    if (
                        np.isfinite(depth)
                        and self.min_depth_m <= depth <= self.max_depth_m
                    ):
                        depth_text = f"{depth:.3f} m"
                    else:
                        depth_text = "out of range"

                    self.info.config(
                        text=(
                            f"Pixel : ({x}, {y})\n"
                            f"Depth : {depth_text}"
                        )
                    )

        self.root.after(30, self.update_gui)


def main(args=None):

    rclpy.init(args=args)

    node = DepthViewer()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.root.update()
    except tk.TclError:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
