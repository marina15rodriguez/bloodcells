import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def load_model_from_checkpoint(checkpoint_path, device):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import create_model

    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = len(ckpt["class_to_idx"])
    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def run_inference(model, loader, device):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(labels, preds, class_names, output_path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate WBC classifier")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset import build_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.checkpoint).parent

    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    _, _, test_loader = build_dataloaders(args.data_dir, num_workers=0)

    print("Running inference on test set...")
    preds, labels = run_inference(model, test_loader, device)

    print("\n=== Classification Report ===")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    plot_confusion_matrix(labels, preds, class_names, output_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
