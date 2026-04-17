"""
ocr.py
------
Extracts text visible on a book cover using EasyOCR.
Falls back to an empty string if nothing is detected.
"""

import easyocr
import numpy as np

_reader = None


def _load():
    global _reader
    if _reader is None:
        print("  Loading EasyOCR (first run downloads language model)...")
        # gpu=False keeps it simple; set to True if you have CUDA
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text(image_path: str) -> str:
    """
    Returns a single cleaned string of all text detected on the cover.
    Filters out very short fragments (single chars, noise).
    """
    reader  = _load()
    results = reader.readtext(image_path, detail=0, paragraph=True)

    # Flatten, strip, filter noise
    tokens = [t.strip() for t in results if len(t.strip()) > 1]
    return " ".join(tokens)