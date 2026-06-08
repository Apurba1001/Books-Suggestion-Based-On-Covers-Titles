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

# ── State ──────────────────────────────────────────────────────────────────────

_index      = None   # list of dicts, one per book
_clip_image = None   # (N, 512)
_clip_text  = None   # (N, 512)
_sentence   = None   # (N, 384)
_colors     = None   # (N, 15)
_loaded_dir = None   # track which dir is currently loaded


# ── Load ───────────────────────────────────────────────────────────────────────

def load_index(embeddings_dir: str = "embeddings"):
    """
    Load all embedding matrices and row metadata into memory.

    Args:
        embeddings_dir: path to the embeddings folder.
                        Default: "embeddings"  (OpenLibrary dataset)
                        NYT:     "embeddings_nyt"
    """
    global _index, _clip_image, _clip_text, _sentence, _colors, _loaded_dir

    d = Path(embeddings_dir)

    if _loaded_dir == str(d):
        return  # already loaded, skip

    print(f"Loading embedding index from {d}/...")

    with open(d / "row_index.json", encoding="utf-8") as f:
        _index = json.load(f)

    _clip_image = np.load(d / "clip_image.npy")
    _clip_text  = np.load(d / "clip_text.npy")
    _sentence   = np.load(d / "sentence.npy")
    _colors     = np.load(d / "colors.npy")
    _loaded_dir = str(d)

    print(f"  {len(_index)} books loaded across 4 tracks.\n")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cosine(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return (m @ q).astype(np.float32)


def _top_k(scores: np.ndarray, k: int, exclude_idx: int = -1) -> list[dict]:
    ranked  = np.argsort(scores)[::-1]
    results = []
    for idx in ranked:
        if int(idx) == exclude_idx:
            continue
        results.append({
            "rank":     len(results) + 1,
            "score":    float(scores[idx]),
            "id":       _index[idx].get("id") or _index[idx].get("cover_id", ""),
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
    exclude_id: str = None,
) -> dict:
    """
    Run cosine search across whichever tracks have a query vector.

    Returns a dict with keys:
        "visual"       — CLIP image → image similarity
        "cross_modal"  — CLIP text  → image similarity
        "semantic"     — SentenceTransformer text similarity
        "color"        — dominant color palette similarity

    Each value is a list of top-k result dicts.
    Pass exclude_id (isbn or cover_id string) to remove the query book
    from its own results.
    """
    if _index is None:
        raise RuntimeError("Call load_index() before search().")

    exclude_idx = -1
    if exclude_id is not None:
        for i, book in enumerate(_index):
            book_id = book.get("id") or book.get("cover_id", "")
            if str(book_id) == str(exclude_id):
                exclude_idx = i
                break

    results = {}

    if clip_image_vec is not None:
        scores = _cosine(clip_image_vec, _clip_image)
        results["visual"] = _top_k(scores, k, exclude_idx)

    if clip_text_vec is not None:
        scores = _cosine(clip_text_vec, _clip_image)
        results["cross_modal"] = _top_k(scores, k, exclude_idx)

    if sentence_vec is not None:
        scores = _cosine(sentence_vec, _sentence)
        results["semantic"] = _top_k(scores, k, exclude_idx)

    if color_vec is not None:
        scores = _cosine(color_vec, _colors)
        results["color"] = _top_k(scores, k, exclude_idx)

    return results