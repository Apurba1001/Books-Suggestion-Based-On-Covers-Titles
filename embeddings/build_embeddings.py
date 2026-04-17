"""
build_embeddings.py
-------------------
Runs all feature extractors over the full dataset and saves vectors to disk.

Usage:
    python src/features/build_embeddings.py

Output (in embeddings/):
    clip_image.npy      — (N, 512)  CLIP image vectors
    clip_text.npy       — (N, 512)  CLIP text vectors (from OCR or title+author)
    sentence.npy        — (N, 384)  SentenceTransformer vectors
    colors.npy          — (N, 15)   dominant color palette vectors
    row_index.json      — maps row i in the .npy files → book in index.json
    index_updated.json  — index.json enriched with ocr_text field per book
"""

import sys
import json
import numpy as np
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.clip_encoder  import encode_image, encode_text
from src.features.ocr           import extract_text
from src.features.colors        import extract_palette
from src.features.text_encoder  import encode as encode_sentence

# ── Paths ──────────────────────────────────────────────────────────────────────

INDEX_PATH        = Path("data/index.json")
EMBEDDINGS_DIR    = Path("embeddings")
ROW_INDEX_PATH    = EMBEDDINGS_DIR / "row_index.json"
UPDATED_INDEX     = Path("data/index_updated.json")


# ── Helpers ────────────────────────────────────────────────────────────────────

def fallback_text(book: dict) -> str:
    """Title + author string used when OCR returns nothing."""
    authors = ", ".join(book.get("authors", []))
    return f"{book.get('title', '')} {authors}".strip()


def load_index() -> list[dict]:
    with open(INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    index = load_index()
    n     = len(index)
    print(f"Building embeddings for {n} books...\n")

    clip_image_vecs = np.zeros((n, 512),  dtype=np.float32)
    clip_text_vecs  = np.zeros((n, 512),  dtype=np.float32)
    sentence_vecs   = np.zeros((n, 384),  dtype=np.float32)
    color_vecs      = np.zeros((n, 15),   dtype=np.float32)

    row_index = []   # row i → book metadata (cover_id, title, genre, filename)

    for i, book in enumerate(index):
        img_path = book["filename"]
        print(f"[{i+1:03d}/{n}] {book['title'][:50]:<50}  ({book['genre']})")

        # ── OCR ──────────────────────────────────────────────────────────────
        try:
            ocr_text = extract_text(img_path)
        except Exception as e:
            print(f"         OCR failed: {e}")
            ocr_text = ""

        # Use OCR output if it found something, otherwise fall back to metadata
        text_for_encoding = ocr_text if ocr_text.strip() else fallback_text(book)
        book["ocr_text"] = ocr_text  # enrich the record

        # ── CLIP image ───────────────────────────────────────────────────────
        try:
            clip_image_vecs[i] = encode_image(img_path)
        except Exception as e:
            print(f"         CLIP image failed: {e}")

        # ── CLIP text ────────────────────────────────────────────────────────
        try:
            clip_text_vecs[i] = encode_text(text_for_encoding)
        except Exception as e:
            print(f"         CLIP text failed: {e}")

        # ── SentenceTransformer ──────────────────────────────────────────────
        try:
            sentence_vecs[i] = encode_sentence(text_for_encoding)
        except Exception as e:
            print(f"         SentenceTransformer failed: {e}")

        # ── Color palette ────────────────────────────────────────────────────
        try:
            color_vecs[i] = extract_palette(img_path)
        except Exception as e:
            print(f"         Color extraction failed: {e}")

        row_index.append({
            "row":      i,
            "cover_id": book["cover_id"],
            "title":    book["title"],
            "authors":  book["authors"],
            "genre":    book["genre"],
            "filename": book["filename"],
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(EMBEDDINGS_DIR / "clip_image.npy", clip_image_vecs)
    np.save(EMBEDDINGS_DIR / "clip_text.npy",  clip_text_vecs)
    np.save(EMBEDDINGS_DIR / "sentence.npy",   sentence_vecs)
    np.save(EMBEDDINGS_DIR / "colors.npy",     color_vecs)

    with open(ROW_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(row_index, f, indent=2, ensure_ascii=False)

    with open(UPDATED_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done. Saved to {EMBEDDINGS_DIR}/")
    print(f"  clip_image.npy  : {clip_image_vecs.shape}")
    print(f"  clip_text.npy   : {clip_text_vecs.shape}")
    print(f"  sentence.npy    : {sentence_vecs.shape}")
    print(f"  colors.npy      : {color_vecs.shape}")
    print(f"  row_index.json  : {len(row_index)} entries")


if __name__ == "__main__":
    main()