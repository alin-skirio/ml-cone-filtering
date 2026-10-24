"""Print PointNeXt-S predictions for clustered PointCloud2 messages in a ROS bag."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rosbags.highlevel import AnyReader

from pointnext_classifier import build_pointnext_s_classifier
from predict import DEFAULT_CLASS_NAMES, DEFAULT_SEED, prepare_input


POINT_FIELD_DTYPES = {
    1: "i1",
    2: "u1",
    3: "i2",
    4: "u2",
    5: "i4",
    6: "u4",
    7: "f4",
    8: "f8",
}
POINTCLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"
XYZ_FIELDS = ("x", "y", "z")


def pointcloud2_to_xyz(message: Any) -> np.ndarray:
    if message.width <= 0 or message.height <= 0:
        raise ValueError("PointCloud2 message is empty")
    if message.point_step <= 0 or message.row_step < message.width * message.point_step:
        raise ValueError("PointCloud2 message has invalid point or row stride")
    if len(message.data) < message.row_step * message.height:
        raise ValueError("PointCloud2 message data is shorter than its declared dimensions")

    fields_by_name = {field.name: field for field in message.fields}
    missing = [name for name in XYZ_FIELDS if name not in fields_by_name]
    if missing:
        raise ValueError(f"PointCloud2 message is missing required fields: {', '.join(missing)}")

    endian = ">" if message.is_bigendian else "<"
    formats = []
    offsets = []
    for name in XYZ_FIELDS:
        field = fields_by_name[name]
        type_code = POINT_FIELD_DTYPES.get(field.datatype)
        if type_code is None:
            raise ValueError(f"PointCloud2 field {name!r} has unsupported datatype")
        if field.count != 1:
            raise ValueError(f"PointCloud2 field {name!r} must have count 1")
        field_dtype = np.dtype(endian + type_code)
        if field.offset < 0 or field.offset + field_dtype.itemsize > message.point_step:
            raise ValueError(f"PointCloud2 field {name!r} has an invalid offset")
        formats.append(field_dtype)
        offsets.append(field.offset)

    dtype = np.dtype(
        {
            "names": list(XYZ_FIELDS),
            "formats": formats,
            "offsets": offsets,
            "itemsize": message.point_step,
        }
    )
    organized = np.ndarray(
        shape=(message.height, message.width),
        dtype=dtype,
        buffer=message.data,
        strides=(message.row_step, message.point_step),
    )
    xyzi = np.column_stack(
        [organized[name].reshape(-1) for name in XYZ_FIELDS]
    ).astype(np.float32, copy=False)
    xyzi = xyzi[np.isfinite(xyzi).all(axis=1)]
    if not len(xyzi):
        raise ValueError("PointCloud2 message has no finite XYZ points")
    return np.ascontiguousarray(xyzi)


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, tuple[str, str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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

    model = build_pointnext_s_classifier(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, (class_names[0], class_names[1])


def main() -> None:
    data_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bag",
        type=Path,
        help="ROS 2 bag directory or bag file",
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Topic containing one clustered sensor_msgs/PointCloud2 per cone cluster",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=data_directory / "pointnext_s_10epoch.pth",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.bag.exists():
        parser.error(f"Bag does not exist: {args.bag}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint file does not exist: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but this PyTorch installation has no CUDA device")

    device = torch.device(args.device)
    model, class_names = load_model(args.checkpoint, device)
    count = 0

    with AnyReader([args.bag]) as reader:
        topic_connections = [
            connection for connection in reader.connections if connection.topic == args.topic
        ]
        if not topic_connections:
            available = sorted({connection.topic for connection in reader.connections})
            raise ValueError(
                f"Topic {args.topic!r} not found in bag. Available topics: "
                + (", ".join(available) if available else "(none)")
            )
        message_types = {connection.msgtype for connection in topic_connections}
        if message_types != {POINTCLOUD2_TYPE}:
            raise ValueError(
                f"Topic {args.topic!r} must have message type {POINTCLOUD2_TYPE}; "
                f"found {', '.join(sorted(message_types))}"
            )

        for connection, timestamp, rawdata in reader.messages(
            connections=topic_connections
        ):
            message = reader.deserialize(rawdata, connection.msgtype)
            try:
                points = pointcloud2_to_xyz(message)
            except ValueError as error:
                message_context = f"Invalid cluster message {count} on {args.topic}: {error}"
                raise ValueError(message_context) from error

            model_input = prepare_input(points, args.seed + count, device)
            with torch.inference_mode():
                probabilities = torch.softmax(model(model_input), dim=1)[0]
                predicted_class = probabilities.argmax().item()

            print(
                f"cluster {count} at {timestamp / 1_000_000_000:.3f}s: "
                f"{class_names[predicted_class]} "
                f"({probabilities[predicted_class].item():.1%})"
            )
            count += 1

    if count == 0:
        raise ValueError(f"No cluster messages found on topic {args.topic!r}")
    print(f"Processed {count} clusters from {args.topic}")


if __name__ == "__main__":
    main()
