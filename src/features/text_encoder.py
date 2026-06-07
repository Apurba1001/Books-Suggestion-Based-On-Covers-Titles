"""
text_encoder.py
---------------
Encodes text (OCR output or title+author fallback) into a 384-dim
semantic vector using a SentenceTransformer model.

Separate from CLIP text encoding — this model is trained specifically
on sentence-level semantic similarity, which makes it better for
matching themes and descriptions than CLIP's text encoder.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
MODEL_ID = "all-MiniLM-L6-v2"

_model = None


def _load():
    global _model
    if _model is None:
        print("  Loading SentenceTransformer (first run downloads ~90MB)...")
        _model = SentenceTransformer(MODEL_ID)
    return _model


def encode(text: str) -> np.ndarray:
    """
    Returns a normalised 384-dim float32 vector for the input text.
    If text is empty, returns a zero vector.
    """
    if not text or not text.strip():
        return np.zeros(384, dtype=np.float32)

    model    = _load()
    vector   = model.encode(text, normalize_embeddings=True)
    return vector.astype(np.float32)