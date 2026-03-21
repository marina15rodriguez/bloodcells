"""
Training script for the WBC CNN classifier.

Usage:
    python src/train.py --data-dir data/ --epochs 20 --output-dir results/

Two-phase training strategy:
  Phase 1 (warm-up): freeze backbone, train head only at full lr.
  Phase 2 (fine-tune): unfreeze all, backbone at lr*0.1, head at lr,
                        with CosineAnnealingLR.
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as: python src/train.py from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_dataloaders
from model import (
    count_parameters,
    create_model,
    freeze_backbone,
    unfreeze_backbone,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(
    model: nn.Module, lr: float, weight_decay: float, phase: str
) -> torch.optim.Optimizer:
    """
    phase='warmup': single param group (head only).
    phase='finetune': two param groups — backbone at lr*0.1, head at lr.
    """
    if phase == "warmup":
        head_params = [p for p in model.classifier.parameters()]
        return torch.optim.AdamW(head_params, lr=lr, weight_decay=weight_decay)
    else:
        backbone_params = [
            p for name, p in model.named_parameters() if "classifier" not in name
        ]
        head_params = list(model.classifier.parameters())
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr * 0.1},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple:
    """
    Run one training epoch with AMP.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Run evaluation on the validation set.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val  ", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def save_checkpoint(state: dict, path: Path) -> None:
    """Atomic checkpoint save: write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.rename(path)


def plot_training_curves(history: dict, output_dir: Path) -> None:
    """
    Save a two-panel training curves figure.

    history keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.tight_layout()
    out_path = output_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WBC CNN Classifier")
    parser.add_argument("--data-dir", required=True, help="Root of the dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_loader, val_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = create_model(num_classes=4, pretrained=not args.no_pretrained)
    model = model.to(device)
    params = count_parameters(model)
    print(f"Parameters — total: {params['total']:,}, trainable: {params['trainable']:,}")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # Optionally resume
    start_epoch = 0
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']} (best val acc: {best_val_acc:.4f})")

    # Derive phase boundaries relative to start_epoch
    warmup_end = args.warmup_epochs
    total_epochs = args.epochs

    # --- Phase 1: Warm-up ---
    if start_epoch < warmup_end:
        print(f"\n=== Warm-up phase (epochs 1–{warmup_end}) ===")
        freeze_backbone(model)
        warmup_params = count_parameters(model)
        print(f"Trainable params (head only): {warmup_params['trainable']:,}")
        optimizer = get_optimizer(model, args.lr, args.weight_decay, phase="warmup")

        for epoch in range(start_epoch, warmup_end):
            print(f"Epoch {epoch + 1}/{total_epochs}")
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(
                f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": None,
                        "best_val_acc": best_val_acc,
                        "class_to_idx": train_loader.dataset.subset.dataset.class_to_idx,
                        "args": vars(args),
                    },
                    output_dir / "best_model.pth",
                )
                print(f"  --> New best val acc: {best_val_acc:.4f} (checkpoint saved)")

    # --- Phase 2: Full fine-tune ---
    finetune_start = max(start_epoch, warmup_end)
    if finetune_start < total_epochs:
        print(f"\n=== Fine-tune phase (epochs {warmup_end + 1}–{total_epochs}) ===")
        unfreeze_backbone(model)
        ft_params = count_parameters(model)
        print(f"Trainable params (all): {ft_params['trainable']:,}")
        optimizer = get_optimizer(model, args.lr, args.weight_decay, phase="finetune")
        remaining = total_epochs - finetune_start
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining
        )

        for epoch in range(finetune_start, total_epochs):
            print(f"Epoch {epoch + 1}/{total_epochs}")
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(
                f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_acc": best_val_acc,
                        "class_to_idx": train_loader.dataset.subset.dataset.class_to_idx,
                        "args": vars(args),
                    },
                    output_dir / "best_model.pth",
                )
                print(f"  --> New best val acc: {best_val_acc:.4f} (checkpoint saved)")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    plot_training_curves(history, output_dir)


if __name__ == "__main__":
    main()
