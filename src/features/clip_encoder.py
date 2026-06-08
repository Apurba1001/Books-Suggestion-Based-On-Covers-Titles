"""
clip_encoder.py
---------------
Encodes cover images and text into a shared 512-dim CLIP embedding space.
Uses HuggingFace's implementation — no git install required.
"""
import logging
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
_model     = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        print("  Loading CLIP model (first run downloads ~600MB)...")
        _model     = CLIPModel.from_pretrained(MODEL_ID, force_download=False)
        _processor = CLIPProcessor.from_pretrained(MODEL_ID)
        _model.eval()
        print(f"  Model type: {type(_model)}")
        assert hasattr(_model, "get_image_features"), "Wrong model class loaded!"
    return _model, _processor


def encode_image(image_path: str) -> np.ndarray:
    model, processor = _load()
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs  = model.vision_model(pixel_values=inputs["pixel_values"])
        features = model.visual_projection(outputs.pooler_output)

    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze().detach().numpy().astype(np.float32)


def encode_text(text: str) -> np.ndarray:
    model, processor = _load()
    inputs = processor(
        text=[text], return_tensors="pt", padding=True, truncation=True
    )
    
    with torch.no_grad():
        outputs  = model.text_model(**inputs)
        features = model.text_projection(outputs.pooler_output)

    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze().detach().numpy().astype(np.float32)