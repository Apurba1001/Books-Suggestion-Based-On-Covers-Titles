"""
clip_encoder.py
---------------
Encodes cover images and text into a shared embedding space.
Supports HuggingFace CLIP, OpenCLIP, and SigLIP models.

Model ID formats:
    openai/clip-vit-base-patch32           — HF CLIP (baseline)
    openai/clip-vit-base-patch16           — HF CLIP (finer patches)
    openclip:ViT-B-32/laion2b_s34b_b79k   — OpenCLIP (LAION-2B)
    siglip:google/siglip-base-patch16-224  — SigLIP (sigmoid loss)
"""

import torch
import numpy as np
from PIL import Image

# ── State ──────────────────────────────────────────────────────────────────────

_model_id  = "openai/clip-vit-base-patch32"
_model     = None
_processor = None
_tokenizer = None   # only for openclip
_backend   = None   # "hf", "openclip", or "siglip"


def set_model(model_id: str):
    """Switch the model. Forces reload on next encode."""
    global _model_id, _model, _processor, _tokenizer, _backend
    if model_id != _model_id:
        _model_id  = model_id
        _model     = None
        _processor = None
        _tokenizer = None
        _backend   = None
        print(f"  Model set to: {model_id}")


def _load():
    global _model, _processor, _tokenizer, _backend
    if _model is not None:
        return

    if _model_id.startswith("openclip:"):
        _load_openclip()
    elif _model_id.startswith("siglip:"):
        _load_siglip()
    else:
        _load_hf()


def _load_hf():
    global _model, _processor, _backend
    from transformers import CLIPProcessor, CLIPModel
    print(f"  Loading HF CLIP: {_model_id}")
    _model     = CLIPModel.from_pretrained(_model_id)
    _processor = CLIPProcessor.from_pretrained(_model_id)
    _model.eval()
    _backend = "hf"
    print(f"  Backend: HuggingFace CLIP")


def _load_openclip():
    global _model, _processor, _tokenizer, _backend
    import open_clip
    spec = _model_id.split(":", 1)[1]
    arch, pretrained = spec.split("/", 1)
    print(f"  Loading OpenCLIP: arch={arch}, pretrained={pretrained}")
    _model, _, _processor = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    _tokenizer = open_clip.get_tokenizer(arch)
    _model.eval()
    _backend = "openclip"
    print(f"  Backend: open_clip")


def _load_siglip():
    global _model, _processor, _backend
    from transformers import SiglipModel, SiglipProcessor
    repo = _model_id.split(":", 1)[1]
    print(f"  Loading SigLIP: {repo}")
    _model     = SiglipModel.from_pretrained(repo)
    _processor = SiglipProcessor.from_pretrained(repo)
    _model.eval()
    _backend = "siglip"
    dim = _model.config.vision_config.hidden_size
    print(f"  Backend: SigLIP ({dim}-dim)")


def get_model_id() -> str:
    return _model_id


# ── Encoders ───────────────────────────────────────────────────────────────────

def encode_image(image_path: str) -> np.ndarray:
    """Returns a normalised float32 vector for a cover image."""
    _load()
    image = Image.open(image_path).convert("RGB")

    if _backend == "hf":
        inputs = _processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs  = _model.vision_model(pixel_values=inputs["pixel_values"])
            features = _model.visual_projection(outputs.pooler_output)

    elif _backend == "openclip":
        image_tensor = _processor(image).unsqueeze(0)
        with torch.no_grad():
            features = _model.encode_image(image_tensor)

    elif _backend == "siglip":
        inputs = _processor(images=image, return_tensors="pt")
        with torch.no_grad():
            vision_out = _model.vision_model(pixel_values=inputs["pixel_values"])
            features   = vision_out.pooler_output
            if hasattr(_model, "visual_projection"):
                features = _model.visual_projection(features)

    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze().detach().numpy().astype(np.float32)


def encode_text(text: str) -> np.ndarray:
    """Returns a normalised float32 vector in the same space as encode_image."""
    _load()

    if _backend == "hf":
        inputs = _processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs  = _model.text_model(**inputs)
            features = _model.text_projection(outputs.pooler_output)

    elif _backend == "openclip":
        tokens = _tokenizer([text])
        with torch.no_grad():
            features = _model.encode_text(tokens)

    elif _backend == "siglip":
        inputs = _processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_out = _model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
            features = text_out.pooler_output
            if hasattr(_model, "text_projection"):
                features = _model.text_projection(features)

    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze().detach().numpy().astype(np.float32)