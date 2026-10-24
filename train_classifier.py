"""Train the PointNeXt-S cone classifier for a bounded number of steps."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cone_dataset import ConeDataset, DEFAULT_BATCH_SIZE, DEFAULT_SEED
from pointnext_classifier import build_pointnext_s_classifier


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, float, int]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs = {"pos": batch["pos"].to(device), "x": batch["x"].to(device)}
            labels = batch["y"].to(device)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise FloatingPointError("Validation produced non-finite loss or logits")

            count = labels.size(0)
            loss_sum += loss.item() * count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            examples += count

    if examples == 0:
        raise ValueError("Validation processed zero examples")
    return loss_sum / examples, correct / examples, examples


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, int, float, float]:
    model.train()
    loss_sum = 0.0
    examples = 0
    first_loss: float | None = None
    last_loss: float | None = None

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        inputs = {"pos": batch["pos"].to(device), "x": batch["x"].to(device)}
        labels = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise FloatingPointError("Training produced non-finite loss or logits")

        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Training produced missing or non-finite gradients")
        optimizer.step()

        count = labels.size(0)
        value = loss.item()
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        loss_sum += value * count
        examples += count

    if examples == 0 or first_loss is None or last_loss is None:
        raise ValueError("Training processed zero examples")
    return loss_sum / examples, examples, first_loss, last_loss


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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=2,
        help="Cap batches per epoch for a quick run; use 0 for the full training split.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=1,
        help="Cap validation batches; use 0 to evaluate the full validation split.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=data_directory / "pointnext_s_xyz.pth",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 1:
        parser.error("--epochs must be positive and --batch-size must be greater than 1")
    if args.max_train_batches < 0 or args.max_validation_batches < 0:
        parser.error("batch limits must be 0 (unlimited) or positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but this PyTorch installation has no CUDA device")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_dataset = ConeDataset(args.train, mode="train", seed=args.seed)
    validation_dataset = ConeDataset(args.validation, mode="validation", seed=args.seed)
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_pointnext_s_classifier(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    max_train_batches = args.max_train_batches or None
    max_validation_batches = args.max_validation_batches or None
    best_validation_loss = float("inf")

    print(f"Device: {device}")
    print(
        f"Training examples: {len(train_dataset):,}; "
        f"validation examples: {len(validation_dataset):,}"
    )
    print("The test split is not loaded or evaluated by this script.")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_examples, first_loss, last_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_train_batches,
        )
        validation_loss, validation_accuracy, validation_examples = evaluate(
            model,
            validation_loader,
            device,
            max_validation_batches,
        )
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.6f} over {train_examples} examples "
            f"(first batch={first_loss:.6f}, last batch={last_loss:.6f}); "
            f"validation_loss={validation_loss:.6f}, "
            f"validation_accuracy={validation_accuracy:.3f} "
            f"over {validation_examples} examples"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "seed": args.seed,
                    "points_per_cluster": train_dataset.points_per_cluster,
                    "class_names": ["non-cone", "cone"],
                },
                args.checkpoint,
            )
            print(f"  saved best validation checkpoint: {args.checkpoint}")

    if max_train_batches is not None or max_validation_batches is not None:
        print(
            "Note: this was a bounded smoke run, not a complete training or "
            "validation pass. Its metrics are not final model-performance results."
        )


if __name__ == "__main__":
    main()
