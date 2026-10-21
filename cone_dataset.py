"""PyTorch Dataset and DataLoader setup for cone-cluster classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


POINTS_PER_CLUSTER = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42


class ConeDataset(Dataset[dict[str, torch.Tensor]]):
    """Load labeled point clusters and sample each one to a fixed size."""

    def __init__(
        self,
        json_path: str | Path,
        mode: str,
        points_per_cluster: int = POINTS_PER_CLUSTER,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if mode not in {"train", "validation", "test"}:
            raise ValueError("mode must be 'train', 'validation', or 'test'")
        if points_per_cluster <= 0:
            raise ValueError("points_per_cluster must be positive")

        self.json_path = Path(json_path)
        self.mode = mode
        self.points_per_cluster = points_per_cluster
        self.seed = seed

        with self.json_path.open("r", encoding="utf-8") as dataset_file:
            dataset = json.load(dataset_file)

        clusters = dataset.get("clusters")
        if not isinstance(clusters, list):
            raise ValueError(f"{self.json_path} must contain a top-level 'clusters' array")

        self.points: list[np.ndarray] = []
        self.labels: list[int] = []
        self.original_point_counts: list[int] = []
        for index, cluster in enumerate(clusters):
            if not isinstance(cluster, dict):
                raise ValueError(f"Cluster {index} in {self.json_path} is not an object")

            raw_points = cluster.get("points_xyzi")
            if not isinstance(raw_points, list):
                raise ValueError(f"Cluster {index} in {self.json_path} has no points_xyzi array")
            if not raw_points:
                raise ValueError(f"Cluster {index} in {self.json_path} has no points")

            points = np.asarray(raw_points, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 4:
                raise ValueError(
                    f"Cluster {index} in {self.json_path} must have points_xyzi "
                    f"with shape (N, 4); got {points.shape}"
                )
            if not np.isfinite(points).all():
                raise ValueError(f"Cluster {index} in {self.json_path} has non-finite points")

            label = cluster.get("label")
            if isinstance(label, bool) or label not in (0, 1):
                raise ValueError(f"Cluster {index} in {self.json_path} has invalid label {label!r}")

            self.points.append(np.ascontiguousarray(points))
            self.labels.append(label)
            self.original_point_counts.append(points.shape[0])

    def __len__(self) -> int:
        return len(self.points)

    def _sampling_generator(self, index: int) -> torch.Generator | None:
        if self.mode == "train":
            return None

        generator = torch.Generator()
        generator.manual_seed(self.seed + index)
        return generator

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self.points)
        if index < 0 or index >= len(self.points):
            raise IndexError(f"Cluster index {index} is out of range")

        points = self.points[index]
        point_count = points.shape[0]
        target_count = self.points_per_cluster
        generator = self._sampling_generator(index)

        if point_count > target_count:
            selected = torch.randperm(point_count, generator=generator)[:target_count].numpy()
            points = points[selected]
        elif point_count < target_count:
            duplicate_count = target_count - point_count
            duplicates = torch.randint(
                low=0,
                high=point_count,
                size=(duplicate_count,),
                generator=generator,
            ).numpy()
            points = np.concatenate((points, points[duplicates]), axis=0)

        pos = torch.from_numpy(np.ascontiguousarray(points[:, :3], dtype=np.float32))
        intensity = torch.from_numpy(
            np.ascontiguousarray(points[:, 3], dtype=np.float32)
        ).unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return {
            "pos": pos,
            "x": intensity,
            "y": label,
            "original_num_points": torch.tensor(point_count, dtype=torch.long),
        }


def create_dataloaders(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    points_per_cluster: int = POINTS_PER_CLUSTER,
    seed: int = DEFAULT_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test loaders with the required settings."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    train_dataset = ConeDataset(train_path, "train", points_per_cluster, seed)
    validation_dataset = ConeDataset(validation_path, "validation", points_per_cluster, seed)
    test_dataset = ConeDataset(test_path, "test", points_per_cluster, seed)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader


def validate_loaders(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Check a full batch, finite values, valid labels, and deterministic eval sampling."""
    for name, loader in (
        ("training", train_loader),
        ("validation", validation_loader),
        ("testing", test_loader),
    ):
        if len(loader.dataset) < batch_size:
            raise ValueError(f"The {name} dataset needs at least {batch_size} examples")
        batch = next(iter(loader))
        point_count = loader.dataset.points_per_cluster
        expected_shapes = {
            "pos": (batch_size, point_count, 3),
            "x": (batch_size, 1, point_count),
            "y": (batch_size,),
        }
        for key, expected_shape in expected_shapes.items():
            if tuple(batch[key].shape) != expected_shape:
                raise AssertionError(
                    f"{name} batch {key} shape is {tuple(batch[key].shape)}, "
                    f"expected {expected_shape}"
                )
            if not torch.isfinite(batch[key].to(torch.float32)).all():
                raise AssertionError(f"{name} batch {key} contains non-finite values")
        if not torch.all((batch["y"] == 0) | (batch["y"] == 1)):
            raise AssertionError(f"{name} batch has labels outside {{0, 1}}")
        if batch["original_num_points"].shape != (batch_size,):
            raise AssertionError(f"{name} batch is missing original point-count metadata")

    for name, loader in (("validation", validation_loader), ("testing", test_loader)):
        dataset = loader.dataset
        first = dataset[0]
        second = dataset[0]
        for key in ("pos", "x", "y", "original_num_points"):
            if not torch.equal(first[key], second[key]):
                raise AssertionError(f"{name} sample 0 is not reproducible for {key}")

    train_dataset = train_loader.dataset
    candidate = next(
        (
            index
            for index, count in enumerate(train_dataset.original_point_counts)
            if count > 1 and count != train_dataset.points_per_cluster
        ),
        None,
    )
    if candidate is None:
        raise ValueError("Training data has no cluster suitable for testing random sampling")
    first = train_dataset[candidate]
    changes_across_accesses = any(
        not torch.equal(first["pos"], train_dataset[candidate]["pos"])
        or not torch.equal(first["x"], train_dataset[candidate]["x"])
        for _ in range(4)
    )
    if not changes_across_accesses:
        raise AssertionError("Training point sampling did not change across accesses")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_directory = Path(__file__).resolve().parent
    parser.add_argument(
        "--train",
        type=Path,
        default=data_directory / "normalized_centered_train.json",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=data_directory / "normalized_centered_validation.json",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=data_directory / "normalized_centered_test.json",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    train_loader, validation_loader, test_loader = create_dataloaders(
        args.train,
        args.validation,
        args.test,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    validate_loaders(
        train_loader,
        validation_loader,
        test_loader,
        batch_size=args.batch_size,
    )

    batch = next(iter(train_loader))
    print("Dataset and DataLoader validation passed.")
    print(f"  train clusters: {len(train_loader.dataset):,}")
    print(f"  validation clusters: {len(validation_loader.dataset):,}")
    print(f"  test clusters: {len(test_loader.dataset):,}")
    print(
        "  batch shapes: "
        f"pos={tuple(batch['pos'].shape)}, "
        f"x={tuple(batch['x'].shape)}, "
        f"y={tuple(batch['y'].shape)}"
    )
    print("  values finite; labels binary; validation/test sampling reproducible.")


if __name__ == "__main__":
    main()
