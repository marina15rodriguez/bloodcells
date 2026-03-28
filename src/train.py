import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def save_checkpoint(state, path):
    torch.save(state, path)


def plot_training_curves(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset import build_dataloaders
    from model import create_model, freeze_backbone, unfreeze_backbone

    parser = argparse.ArgumentParser(description="Train WBC classifier")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    set_seed(42)

    print(f"Device: {device}")

    train_loader, val_loader, _ = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=0
    )
    model = create_model(num_classes=4, pretrained=not args.no_pretrained).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Phase 1: Warm-up
    print("\n=== Warm-up phase ===")
    freeze_backbone(model)
    optimizer = get_optimizer(model, args.lr, args.weight_decay, phase="warmup")

    for epoch in range(args.warmup_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1}/{args.epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_to_idx": train_loader.dataset.subset.dataset.class_to_idx,
                    "args": vars(args),
                },
                output_dir / "best_model.pth",
            )
            print(f"  --> Saved (val_acc={best_val_acc:.4f})")

    # Phase 2: Fine-tune
    print("\n=== Fine-tune phase ===")
    unfreeze_backbone(model)
    optimizer = get_optimizer(model, args.lr, args.weight_decay, phase="finetune")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )

    for epoch in range(args.warmup_epochs, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{args.epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  lr={lr_now:.2e}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_to_idx": train_loader.dataset.subset.dataset.class_to_idx,
                    "args": vars(args),
                },
                output_dir / "best_model.pth",
            )
            print(f"  --> Saved (val_acc={best_val_acc:.4f})")

    print(f"\nBest val acc: {best_val_acc:.4f}")
    plot_training_curves(history, output_dir / "training_curves.png")
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()


def get_optimizer(model, lr, weight_decay, phase="warmup"):
    if phase == "warmup":
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    classifier_ids = {id(p) for p in model.classifier.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in classifier_ids]
    head_params = list(model.classifier.parameters())
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total
