"""Build and smoke-test an OpenPoints PointNeXt-S cone classifier."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from easydict import EasyDict

from cone_dataset import create_dataloaders, validate_loaders


def build_pointnext_s_classifier(device: torch.device) -> torch.nn.Module:
    """Build the OpenPoints PointNeXt-S encoder with a two-class head.

    The CPU path uses OpenPoints' native PyTorch grouping operation, KNN
    neighborhoods, and random downsampling because the PyPI package does not
    include the CUDA extensions needed for FPS and ball query.
    """
    from openpoints.models import build_model_from_cfg
    from openpoints.models.layers import group as group_ops

    if device.type == "cuda":
        if importlib.util.find_spec("pointnet2_batch_cuda") is None:
            raise RuntimeError(
                "CUDA was requested, but OpenPoints' pointnet2_batch_cuda extension "
                "is not installed. Build the OpenPoints CUDA ops for this PyTorch/CUDA "
                "environment, or run the CPU smoke test with --device cpu."
            )
        sampler = "fps"
        grouper = "ballquery"
    else:
        group_ops.grouping_operation = group_ops.torch_grouping_operation
        sampler = "random"
        grouper = "knn"

    model_cfg = EasyDict(
        NAME="BaseCls",
        encoder_args=EasyDict(
            NAME="PointNextEncoder",
            in_channels=3,
            width=32,
            blocks=[1, 1, 1, 1, 1, 1],
            strides=[1, 2, 2, 2, 2, 1],
            sampler=sampler,
            radius=0.15,
            radius_scaling=1.5,
            nsample=[32, 32, 16, 8, 4, 32],
            sa_layers=2,
            sa_use_res=True,
            expansion=4,
            aggr_args=EasyDict(feature_type="dp_fj", reduction="max"),
            group_args=EasyDict(NAME=grouper, normalize_dp=False),
            conv_args=EasyDict(order="conv-norm-act"),
            act_args=EasyDict(act="relu"),
            norm_args=EasyDict(norm="bn"),
        ),
        cls_args=EasyDict(
            NAME="ClsHead",
            num_classes=2,
            mlps=[512, 256],
            norm_args=EasyDict(norm="bn1d"),
            dropout=0.5,
        ),
    )
    return build_model_from_cfg(model_cfg).to(device)


def smoke_test(device: torch.device, batch_size: int = 32) -> None:
    data_directory = Path(__file__).resolve().parent
    train_loader, validation_loader, test_loader = create_dataloaders(
        data_directory / "train.json",
        data_directory / "validation.json",
        data_directory / "test.json",
        batch_size=batch_size,
    )
    validate_loaders(
        train_loader,
        validation_loader,
        test_loader,
        batch_size=batch_size,
    )

    batch = next(iter(train_loader))
    model = build_pointnext_s_classifier(device)
    model.train()

    model_inputs = {
        "pos": batch["pos"].to(device),
        "x": batch["x"].to(device),
    }
    labels = batch["y"].to(device)

    model.zero_grad(set_to_none=True)
    logits = model(model_inputs)
    if tuple(logits.shape) != (batch_size, 2):
        raise AssertionError(
            f"Expected classifier output {(batch_size, 2)}, got {tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise AssertionError("Classifier output contains non-finite values")

    loss = torch.nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise AssertionError("Classification loss is non-finite")
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise AssertionError("Backward pass produced no parameter gradients")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("Backward pass produced non-finite gradients")

    print("OpenPoints PointNeXt-S smoke test passed.")
    print(f"  device: {device}")
    print(f"  classifier output: {tuple(logits.shape)}")
    print(f"  cross-entropy loss: {loss.item():.6f}")
    print(f"  parameters with gradients: {len(gradients)}")
    print(f"  first-stage input points: 64")
    if device.type == "cpu":
        print("  CPU operators: random downsampling + KNN + PyTorch grouping")
    else:
        print("  CUDA operators: FPS + ball query")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CPU-compatible OpenPoints ops for smoke testing or compiled CUDA ops.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.batch_size <= 1:
        parser.error("--batch-size must be greater than 1 because the model uses BatchNorm")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but this PyTorch installation has no CUDA device")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    smoke_test(device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
