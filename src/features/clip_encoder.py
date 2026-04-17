"""
clip_encoder.py
---------------
Encodes cover images and text into a shared 512-dim CLIP embedding space.
Uses HuggingFace's implementation — no git install required.
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"

_model     = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        print("  Loading CLIP model (first run downloads ~600MB)...")
        _model     = CLIPModel.from_pretrained(MODEL_ID)
        _processor = CLIPProcessor.from_pretrained(MODEL_ID)
        _model.eval()
    return _model, _processor


def encode_image(image_path: str) -> np.ndarray:
    """
    Returns a normalised 512-dim float32 vector for a cover image.
    """
    model, processor = _load()
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().numpy().astype(np.float32)


def encode_text(text: str) -> np.ndarray:
    """
    Returns a normalised 512-dim float32 vector for a text string.
    Sits in the same space as encode_image — enabling cross-modal search.
    """
    model, processor = _load()
    inputs = processor(
        text=[text], return_tensors="pt", padding=True, truncation=True
    )

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().numpy().astype(np.float32)