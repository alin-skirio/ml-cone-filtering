"""Classify live ROS 2 PointCloud2 clusters and publish Foxglove bounding boxes."""

from __future__ import annotations

from pathlib import Path

import torch
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from pointnext_classifier import build_pointnext_s_classifier
from predict import DEFAULT_CLASS_NAMES, DEFAULT_SEED, prepare_input
from predict_rosbag import pointcloud2_to_xyz


class LiveConeClassifier(Node):
    def __init__(self) -> None:
        super().__init__("live_cone_classifier")
        project_directory = Path(__file__).resolve().parent
        self.declare_parameter("cluster_topic", "/cone_clusters")
        self.declare_parameter("marker_topic", "/cone_classifier/markers")
        self.declare_parameter(
            "checkpoint", str(project_directory / "pointnext_s_10epoch.pth")
        )
        self.declare_parameter("device", "cpu")
        self.declare_parameter("seed", DEFAULT_SEED)
        self.declare_parameter("marker_lifetime_seconds", 2.0)

        cluster_topic = str(self.get_parameter("cluster_topic").value)
        marker_topic = str(self.get_parameter("marker_topic").value)
        checkpoint_path = Path(str(self.get_parameter("checkpoint").value)).expanduser()
        device_name = str(self.get_parameter("device").value)
        self.seed = int(self.get_parameter("seed").value)
        marker_lifetime = float(
            self.get_parameter("marker_lifetime_seconds").value
        )
        if device_name not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch has no CUDA device")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        if marker_lifetime <= 0:
            raise ValueError("marker_lifetime_seconds must be positive")

        self.device = torch.device(device_name)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must contain a state dictionary or checkpoint object")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a model state dictionary")
        class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
        if (
            not isinstance(class_names, (list, tuple))
            or len(class_names) != 2
            or not all(isinstance(name, str) for name in class_names)
        ):
            raise ValueError("Checkpoint class_names must contain exactly two strings")

        self.class_names = (class_names[0], class_names[1])
        self.marker_lifetime = Duration(
            sec=int(marker_lifetime),
            nanosec=int((marker_lifetime % 1) * 1_000_000_000),
        )
        self.marker_id = 0
        self.model = build_pointnext_s_classifier(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.marker_publisher = self.create_publisher(
            MarkerArray, marker_topic, qos_profile_sensor_data
        )
        self.subscription = self.create_subscription(
            PointCloud2,
            cluster_topic,
            self.classify_cluster,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Loaded {checkpoint_path}; listening on {cluster_topic}, "
            f"publishing Foxglove markers on {marker_topic}"
        )

    def classify_cluster(self, message: PointCloud2) -> None:
        try:
            points = pointcloud2_to_xyz(message)
        except ValueError as error:
            self.get_logger().warning(f"Skipping invalid cluster: {error}")
            return

        model_input = prepare_input(
            points, self.seed + self.marker_id, self.device
        )
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(model_input), dim=1)[0]
            predicted_class = probabilities.argmax().item()

        confidence = probabilities[predicted_class].item()
        is_cone = predicted_class == 1
        color = (0.1, 0.9, 0.2) if is_cone else (0.95, 0.15, 0.1)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = (minimum + maximum) / 2
        extent = maximum - minimum
        extent = extent.clip(min=0.01)

        box = Marker()
        box.header = message.header
        box.ns = "cone_classifier_boxes"
        box.id = self.marker_id
        box.type = Marker.CUBE
        box.action = Marker.ADD
        box.pose.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
        box.pose.orientation.w = 1.0
        box.scale.x = float(extent[0])
        box.scale.y = float(extent[1])
        box.scale.z = float(extent[2])
        box.color.r, box.color.g, box.color.b = color
        box.color.a = 0.35
        box.lifetime = self.marker_lifetime

        label = Marker()
        label.header = message.header
        label.ns = "cone_classifier_labels"
        label.id = self.marker_id
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = Point(
            x=float(center[0]),
            y=float(center[1]),
            z=float(maximum[2] + max(float(extent[2]) * 0.1, 0.05)),
        )
        label.pose.orientation.w = 1.0
        label.scale.z = max(float(extent[2]) * 0.15, 0.08)
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0
        label.text = f"{self.class_names[predicted_class]} {confidence:.0%}"
        label.lifetime = self.marker_lifetime

        self.marker_publisher.publish(MarkerArray(markers=[box, label]))
        self.get_logger().info(f"{label.text} (cluster {self.marker_id})")
        self.marker_id = (self.marker_id + 1) % 1_000_000


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LiveConeClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
