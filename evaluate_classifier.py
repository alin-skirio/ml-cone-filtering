"""Evaluate a trained PointNeXt-S checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cone_dataset import DEFAULT_BATCH_SIZE, DEFAULT_SEED, ConeDataset
from pointnext_classifier import build_pointnext_s_classifier


def main() -> None:
    data_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        type=Path,
        default=data_directory / "test.json",
        help="Labeled held-out test data in the training JSON format",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=data_directory / "pointnext_s_10epoch.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_directory / "test_predictions.csv",
        help="CSV file for per-example actual labels and predictions",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.test.is_file():
        parser.error(f"Test data file does not exist: {args.test}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint file does not exist: {args.checkpoint}")
    if args.output.resolve() in (args.test.resolve(), args.checkpoint.resolve()):
        parser.error("--output must not overwrite the test data or checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but this PyTorch installation has no CUDA device")

    device = torch.device(args.device)
    dataset = ConeDataset(args.test, mode="test", seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a state dictionary or checkpoint object")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    class_names = checkpoint.get("class_names", ("non-cone", "cone"))
    if (
        not isinstance(class_names, (list, tuple))
        or len(class_names) != 2
        or not all(isinstance(name, str) for name in class_names)
    ):
        raise ValueError("Checkpoint class_names must contain exactly two strings")

    model = build_pointnext_s_classifier(device)
    model.load_state_dict(state_dict)
    model.eval()

    confusion = torch.zeros((2, 2), dtype=torch.int64)
    loss_sum = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "example_index",
        "actual_label",
        "actual_class",
        "predicted_label",
        "predicted_class",
        "confidence",
        "probability_non_cone",
        "probability_cone",
        "correct",
    )
    example_index = 0
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        with torch.inference_mode():
            for batch in loader:
                inputs = {"pos": batch["pos"].to(device), "x": batch["x"].to(device)}
                labels = batch["y"].to(device)
                logits = model(inputs)
                loss_sum += torch.nn.functional.cross_entropy(
                    logits, labels, reduction="sum"
                ).item()
                probabilities = torch.softmax(logits, dim=1)
                predictions = probabilities.argmax(dim=1)
                encoded = labels * 2 + predictions
                confusion += torch.bincount(encoded.cpu(), minlength=4).reshape(2, 2)

                for actual, predicted, class_probabilities in zip(
                    labels.cpu().tolist(),
                    predictions.cpu().tolist(),
                    probabilities.cpu().tolist(),
                ):
                    writer.writerow(
                        {
                            "example_index": example_index,
                            "actual_label": actual,
                            "actual_class": class_names[actual],
                            "predicted_label": predicted,
                            "predicted_class": class_names[predicted],
                            "confidence": f"{class_probabilities[predicted]:.6f}",
                            "probability_non_cone": f"{class_probabilities[0]:.6f}",
                            "probability_cone": f"{class_probabilities[1]:.6f}",
                            "correct": actual == predicted,
                        }
                    )
                    example_index += 1

    total = int(confusion.sum().item())
    if total == 0:
        raise ValueError("Test data contains no examples")

    true_negative, false_positive = confusion[0].tolist()
    false_negative, true_positive = confusion[1].tolist()
    accuracy = (true_negative + true_positive) / total
    cone_precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    cone_recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    cone_f1 = (
        2 * cone_precision * cone_recall / (cone_precision + cone_recall)
        if cone_precision + cone_recall
        else 0.0
    )
    non_cone_recall = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test set: {args.test}")
    print(f"Per-example results: {args.output}")
    print(f"Examples: {total}")
    print(f"Test loss: {loss_sum / total:.4f}")
    print(f"Accuracy: {accuracy:.1%} ({true_negative + true_positive}/{total})")
    print(f"Non-cone recall (specificity): {non_cone_recall:.1%}")
    print(f"Cone precision: {cone_precision:.1%}")
    print(f"Cone recall: {cone_recall:.1%}")
    print(f"Cone F1: {cone_f1:.1%}")
    print("Confusion matrix (actual rows, predicted columns):")
    print(f"  {class_names[0]}: {true_negative:6d} {false_positive:6d}")
    print(f"  {class_names[1]}: {false_negative:6d} {true_positive:6d}")
    print("Sampling: deterministic 64-point sampling per cluster (seed "
          f"{args.seed})")


if __name__ == "__main__":
    main()
