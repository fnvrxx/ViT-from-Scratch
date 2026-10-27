import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .cnn import SmallCNN
from .data import CLASS_NAMES, load_datasets
from .model import TinyViT, ViTConfig


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(labels)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            count += len(labels)
            confusion += torch.bincount(
                labels * 10 + predictions, minlength=100
            ).reshape(10, 10)
    return {
        "loss": total_loss / count,
        "accuracy": correct / count,
        "confusion_matrix": confusion.tolist(),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer) -> dict:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += len(labels)
    return {"loss": total_loss / count, "accuracy": correct / count}


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_data, validation_data, test_data = load_datasets(args.data_dir, args.seed)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 0}
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kwargs,
    )
    validation_loader = DataLoader(validation_data, **loader_kwargs)
    test_loader = DataLoader(test_data, **loader_kwargs)
    model = TinyViT() if args.model == "vit" else SmallCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.model}.pt"
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer)
        validation_metrics = evaluate(model, validation_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )
        print(
            f"{args.model} epoch {epoch}/{args.epochs} "
            f"train={train_metrics['accuracy']:.3f} "
            f"validation={validation_metrics['accuracy']:.3f}",
            flush=True,
        )
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": args.model,
                    "config": asdict(model.config) if args.model == "vit" else None,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Stopped after {epoch} epochs (validation loss).", flush=True)
                break

    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate(model, test_loader)
    results = {
        "model": args.model,
        "seed": args.seed,
        "split": {"train": 54_000, "validation": 6_000, "test": 10_000},
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "best_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "classes": list(CLASS_NAMES),
        "history": history,
        "test": test_metrics,
    }
    result_path = output_dir / f"{args.model}_metrics.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {checkpoint_path} and {result_path}", flush=True)
    print(f"Test accuracy: {test_metrics['accuracy']:.3f}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train a ViT or CNN on Fashion-MNIST")
    parser.add_argument("--model", choices=("vit", "cnn"), required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience, and batch-size must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    run(args)


if __name__ == "__main__":
    main()
