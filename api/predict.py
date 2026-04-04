"""Inference pipeline for blood cell classification API.

Accepts an image file (JPEG/PNG), runs inference with EfficientNet-B0,
and returns the predicted class and per-class confidence scores.
"""

import io
import base64
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import create_model
from dataset import CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_model = None
_device = None

_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(checkpoint_path) -> None:
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = create_model(num_classes=len(CLASSES), pretrained=False).to(_device)
    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    _model.load_state_dict(state)
    _model.eval()
    print(f"Model loaded from {checkpoint_path} on {_device}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_prediction(image_bytes: bytes) -> dict:
    """Run classification on raw image bytes.

    Returns:
        dict with:
            'predicted_class'  : class name (e.g. 'NEUTROPHIL')
            'confidence'       : confidence of the top prediction (0–1)
            'probabilities'    : dict of {class_name: probability} for all 4 classes
            'original_png'     : base64-encoded PNG of the uploaded image (resized to 240×240)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Save resized original for display
    display = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    buf = io.BytesIO()
    display.save(buf, format="PNG")
    original_b64 = base64.b64encode(buf.getvalue()).decode()

    # Preprocess and infer
    tensor = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    top_idx = int(torch.tensor(probs).argmax())

    return {
        "predicted_class": CLASSES[top_idx],
        "confidence":      round(probs[top_idx], 4),
        "probabilities":   {cls: round(p, 4) for cls, p in zip(CLASSES, probs)},
        "original_png":    original_b64,
    }
