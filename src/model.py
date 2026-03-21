"""
EfficientNet-B0 model for WBC classification via transfer learning.
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required: pip install timm>=0.9.0") from e


def create_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Create an EfficientNet-B0 model with a custom classification head.

    The backbone is loaded with ImageNet pretrained weights. timm replaces
    the original head with a Linear(1280, num_classes) automatically.

    Args:
        num_classes: Number of output classes (4 for WBC).
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        nn.Module ready for training.
    """
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.3,
        drop_path_rate=0.2,
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all parameters except the classification head (model.classifier).

    Used during the warm-up phase so only the randomly-initialized head trains,
    preventing early gradient updates from corrupting pretrained features.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all model parameters for full fine-tuning.

    Called after the warm-up phase. Use differential learning rates in the
    optimizer: backbone at lr*0.1, head at lr (see train.py).
    """
    for param in model.parameters():
        param.requires_grad = True


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """
    Return the last convolutional layer suitable for Grad-CAM visualization.

    For timm's EfficientNet-B0, this is model.conv_head — a 1x1 pointwise
    conv before global average pooling, which produces spatially-rich
    feature maps.
    """
    return model.conv_head


def count_parameters(model: nn.Module) -> dict:
    """
    Return total and trainable parameter counts.

    Returns:
        {'total': N, 'trainable': M}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
