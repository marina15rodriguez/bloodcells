"""
Evaluation script for the WBC CNN classifier.

Loads a saved checkpoint, runs inference on the test set, prints a
classification report, saves a confusion matrix PNG, and generates
Grad-CAM visualizations per class.

Usage:
    python src/evaluate.py --checkpoint results/best_model.pth --data-dir data/
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
from torch.utils.data import DataLoader
from torchvision import datasets

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD, build_dataloaders, get_transforms
from model import create_model, get_gradcam_target_layer


def load_model_from_checkpoint(
    checkpoint_path,
    device: torch.device,
):
    """
    Load a model from a saved checkpoint.

    Returns:
        (model, checkpoint_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = create_model(num_classes=4, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """
    Run inference over a DataLoader.

    Returns:
        (all_preds, all_labels, all_probs) as numpy arrays.
        all_probs shape: (N, num_classes)
    """
    all_preds = []
    all_labels = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list,
    output_path,
) -> None:
    """
    Save a normalized confusion matrix heatmap.

    Args:
        output_path: If None, calls plt.show() instead of saving.
    """
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()


def generate_gradcam(
    model: torch.nn.Module,
    test_dataset: datasets.ImageFolder,
    class_to_idx: dict,
    output_dir: Path,
    n_samples_per_class: int = 4,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Generate Grad-CAM visualizations for n_samples_per_class images per class.

    Saves results/gradcam_{classname}.png for each class.

    Args:
        test_dataset: ImageFolder for the test set (with test transforms).
        class_to_idx: Class name → index mapping from the checkpoint.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("WARNING: pytorch-grad-cam not available. Skipping Grad-CAM generation.")
        print("Install with: pip install grad-cam>=1.4.8")
        return

    # Sanity check class ordering
    assert test_dataset.class_to_idx == class_to_idx, (
        f"class_to_idx mismatch!\n  checkpoint: {class_to_idx}\n  dataset: {test_dataset.class_to_idx}"
    )

    target_layer = get_gradcam_target_layer(model)
    # GradCAM requires gradients — must NOT be inside torch.no_grad()
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Mean/std tensors for de-normalization
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name, class_idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        # Collect indices for this class
        class_indices = [
            i for i, (_, label) in enumerate(test_dataset.samples) if label == class_idx
        ]
        if not class_indices:
            print(f"  No test samples for class {class_name}, skipping.")
            continue

        selected = random.sample(class_indices, min(n_samples_per_class, len(class_indices)))

        fig, axes = plt.subplots(
            len(selected), 2, figsize=(6, 3 * len(selected))
        )
        if len(selected) == 1:
            axes = [axes]

        fig.suptitle(f"Grad-CAM: {class_name}", fontsize=14)

        for row, img_idx in enumerate(selected):
            img_tensor, _ = test_dataset[img_idx]  # (3, H, W), normalized
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # De-normalize for visualization (float32 in [0, 1])
            raw = img_tensor.squeeze(0).cpu() * std + mean
            raw = raw.permute(1, 2, 0).numpy().clip(0, 1).astype(np.float32)

            grayscale_cam = cam(
                input_tensor=img_tensor,
                targets=[ClassifierOutputTarget(class_idx)],
            )
            cam_overlay = show_cam_on_image(raw, grayscale_cam[0], use_rgb=True)

            axes[row][0].imshow(raw)
            axes[row][0].set_title("Original")
            axes[row][0].axis("off")

            axes[row][1].imshow(cam_overlay)
            axes[row][1].set_title("Grad-CAM")
            axes[row][1].axis("off")

        fig.tight_layout()
        out_path = output_dir / f"gradcam_{class_name.lower()}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Grad-CAM saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WBC CNN Classifier")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--data-dir", required=True, help="Root of the dataset")
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-gradcam", type=int, default=4, help="Grad-CAM samples per class")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Classes: {class_names}")

    # Data
    _, _, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Inference
    print("Running inference on test set...")
    preds, labels, probs = run_inference(model, test_loader, device)

    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    # Confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(labels, preds, class_names, output_path=cm_path)

    # Grad-CAM
    print(f"\nGenerating Grad-CAM ({args.n_gradcam} samples/class)...")
    from dataset import find_data_root
    root = find_data_root(args.data_dir)
    test_dataset = datasets.ImageFolder(
        root=str(root / "TEST"), transform=get_transforms("test")
    )
    generate_gradcam(
        model=model,
        test_dataset=test_dataset,
        class_to_idx=class_to_idx,
        output_dir=output_dir,
        n_samples_per_class=args.n_gradcam,
        device=device,
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
