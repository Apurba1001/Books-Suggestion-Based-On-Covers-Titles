"""
retrieval.py
------------
Loads the pre-built embedding index and runs cosine similarity search
for a given query vector across all four tracks.

This module is stateful — call load_index() once at startup, then call
search() as many times as needed without reloading from disk.
"""

import json
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

EMBEDDINGS_DIR = Path("embeddings")
ROW_INDEX_PATH = EMBEDDINGS_DIR / "row_index.json"


# ── State ──────────────────────────────────────────────────────────────────────

_index      = None   # list of dicts, one per book
_clip_image = None   # (N, 512)
_clip_text  = None   # (N, 512)
_sentence   = None   # (N, 384)
_colors     = None   # (N, 15)


# ── Load ───────────────────────────────────────────────────────────────────────

def load_index():
    """Load all embedding matrices and row metadata into memory."""
    global _index, _clip_image, _clip_text, _sentence, _colors

    print("Loading embedding index...")

    with open(ROW_INDEX_PATH, encoding="utf-8") as f:
        _index = json.load(f)

    _clip_image = np.load(EMBEDDINGS_DIR / "clip_image.npy")
    _clip_text  = np.load(EMBEDDINGS_DIR / "clip_text.npy")
    _sentence   = np.load(EMBEDDINGS_DIR / "sentence.npy")
    _colors     = np.load(EMBEDDINGS_DIR / "colors.npy")

    print(f"  {len(_index)} books loaded across 4 tracks.\n")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cosine(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and every
    row in matrix. Both are expected to be already L2-normalised
    (CLIP and SentenceTransformer outputs are). Color vectors are not
    normalised, so we normalise here just in case.
    """
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return (m @ q).astype(np.float32)


def _top_k(scores: np.ndarray, k: int, exclude_idx: int = -1) -> list[dict]:
    """
    Return the top-k results as a list of dicts with rank, score, and metadata.
    Optionally excludes a row index (used to remove the query itself).
    """
    ranked = np.argsort(scores)[::-1]
    results = []
    for idx in ranked:
        if int(idx) == exclude_idx:
            continue
        results.append({
            "rank":     len(results) + 1,
            "score":    float(scores[idx]),
            "cover_id": _index[idx]["cover_id"],
            "title":    _index[idx]["title"],
            "authors":  _index[idx]["authors"],
            "genre":    _index[idx]["genre"],
            "filename": _index[idx]["filename"],
        })
        if len(results) == k:
            break
    return results


# ── Public API ─────────────────────────────────────────────────────────────────

def search(
    clip_image_vec: np.ndarray = None,
    clip_text_vec:  np.ndarray = None,
    sentence_vec:   np.ndarray = None,
    color_vec:      np.ndarray = None,
    k: int = 3,
    exclude_cover_id: int = None,
) -> dict:
    """
    Run cosine search across whichever tracks have a query vector.

    Returns a dict with keys:
        "visual"       — CLIP image → image similarity
        "cross_modal"  — CLIP text  → image similarity (OCR text vs covers)
        "semantic"     — SentenceTransformer text similarity
        "color"        — dominant color palette similarity

    Each value is a list of top-k result dicts.
    Pass exclude_cover_id to remove the query book from its own results
    (useful when the query image is already in the index).
    """
    if _index is None:
        raise RuntimeError("Call load_index() before search().")

    # Find the row to exclude (if query is in the index)
    exclude_idx = -1
    if exclude_cover_id is not None:
        for i, book in enumerate(_index):
            if book["cover_id"] == exclude_cover_id:
                exclude_idx = i
                break

    results = {}

    if clip_image_vec is not None:
        scores = _cosine(clip_image_vec, _clip_image)
        results["visual"] = _top_k(scores, k, exclude_idx)

    if clip_text_vec is not None:
        scores = _cosine(clip_text_vec, _clip_image)  # text query vs image index
        results["cross_modal"] = _top_k(scores, k, exclude_idx)

    if sentence_vec is not None:
        scores = _cosine(sentence_vec, _sentence)
        results["semantic"] = _top_k(scores, k, exclude_idx)

    if color_vec is not None:
        scores = _cosine(color_vec, _colors)
        results["color"] = _top_k(scores, k, exclude_idx)

    return results