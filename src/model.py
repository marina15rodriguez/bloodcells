import timm
import torch.nn as nn


def create_model(num_classes=4, pretrained=True):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
