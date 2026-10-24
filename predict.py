"""Classify a point cluster using a trained PointNeXt-S checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pointnext_classifier import build_pointnext_s_classifier


POINTS_PER_CLUSTER = 64
DEFAULT_SEED = 42
DEFAULT_CLASS_NAMES = ("non-cone", "cone")


def read_cluster(input_path: Path, cluster_index: int) -> np.ndarray:
    with input_path.open("r", encoding="utf-8") as input_file:
        data: Any = json.load(input_file)

    if isinstance(data, list):
        if cluster_index != 0:
            raise ValueError("A raw points array only contains cluster index 0")
        points = data
    elif isinstance(data, dict) and isinstance(data.get("clusters"), list):
        clusters = data["clusters"]
        if not 0 <= cluster_index < len(clusters):
            raise ValueError(
                f"Cluster index {cluster_index} is out of range "
                f"for {len(clusters)} clusters"
            )
        cluster = clusters[cluster_index]
        if not isinstance(cluster, dict):
            raise ValueError(f"Cluster {cluster_index} must be a JSON object")
        points = cluster.get("points_xyz", cluster.get("points_xyzi"))
    elif isinstance(data, dict):
        points = data.get("points_xyz", data.get("points_xyzi"))
    else:
        points = None

    if not isinstance(points, list) or not points:
        raise ValueError(
            "Input must contain non-empty points_xyz or points_xyzi data, "
            "either directly or inside a clusters array"
        )

    point_array = np.asarray(points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] not in (3, 4):
        raise ValueError(
            f"Expected points_xyz/points_xyzi with shape (N, 3) or (N, 4), "
            f"got {point_array.shape}"
        )
    if not np.isfinite(point_array).all():
        raise ValueError("Input points contain non-finite values")
    return point_array


def prepare_input(
    points: np.ndarray, seed: int, device: torch.device
) -> dict[str, torch.Tensor]:
    point_tensor = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32))
    generator = torch.Generator().manual_seed(seed)

    if point_tensor.shape[0] > POINTS_PER_CLUSTER:
        indices = torch.randperm(point_tensor.shape[0], generator=generator)[
            :POINTS_PER_CLUSTER
        ]
        point_tensor = point_tensor[indices]
    elif point_tensor.shape[0] < POINTS_PER_CLUSTER:
        duplicate_indices = torch.randint(
            low=0,
            high=point_tensor.shape[0],
            size=(POINTS_PER_CLUSTER - point_tensor.shape[0],),
            generator=generator,
        )
        point_tensor = torch.cat((point_tensor, point_tensor[duplicate_indices]), dim=0)

    return {
        "pos": point_tensor[:, :3].unsqueeze(0).to(device),
        "x": point_tensor[:, :3].transpose(0, 1).unsqueeze(0).to(device),
    }


def main() -> None:
    data_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="JSON file containing points_xyz/points_xyzi or a clusters array",
    )
    parser.add_argument(
        "--cluster-index",
        type=int,
        default=0,
        help="Cluster to classify when input contains a clusters array",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=data_directory / "pointnext_s_10epoch.pth",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.cluster_index < 0:
        parser.error("--cluster-index must be non-negative")
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint file does not exist: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but this PyTorch installation has no CUDA device")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a state dictionary or checkpoint object")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")

    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    if not isinstance(class_names, (list, tuple)) or len(class_names) != 2:
        raise ValueError("Checkpoint class_names must contain exactly two class names")

    model = build_pointnext_s_classifier(device)
    model.load_state_dict(state_dict)
    model.eval()

    points = read_cluster(args.input, args.cluster_index)
    model_input = prepare_input(points, args.seed, device)
    with torch.no_grad():
        probabilities = torch.softmax(model(model_input), dim=1)[0]
        predicted_class = probabilities.argmax().item()

    print(f"Prediction: {class_names[predicted_class]}")
    print(f"Confidence: {probabilities[predicted_class].item():.1%}")
    print(
        "Probabilities: "
        + ", ".join(
            f"{name}={probability.item():.1%}"
            for name, probability in zip(class_names, probabilities)
        )
    )


if __name__ == "__main__":
    main()
