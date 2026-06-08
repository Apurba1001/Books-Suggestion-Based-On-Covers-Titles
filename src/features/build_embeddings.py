"""
build_embeddings.py
-------------------
Runs all feature extractors over a dataset and saves vectors to disk.

Usage:
    # Original OpenLibrary dataset
    uv run python embeddings/build_embeddings.py

    # NYT dataset
    uv run python embeddings/build_embeddings.py --index data/nyt_index.json --output embeddings_nyt

Output (in embeddings dir):
    clip_image.npy      — (N, 512)  CLIP image vectors
    clip_text.npy       — (N, 512)  CLIP text vectors (from OCR or title+author)
    sentence.npy        — (N, 384)  SentenceTransformer vectors
    colors.npy          — (N, 15)   dominant color palette vectors
    row_index.json      — maps row i in the .npy files → book in index
    index_updated.json  — index enriched with ocr_text field per book
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.clip_encoder  import encode_image, encode_text
from src.features.ocr           import extract_text
from src.features.colors        import extract_palette
from src.features.text_encoder  import encode as encode_sentence


# ── Helpers ────────────────────────────────────────────────────────────────────

def fallback_text(book: dict) -> str:
    authors = ", ".join(book.get("authors", []))
    return f"{book.get('title', '')} {authors}".strip()


def book_id(book: dict) -> str:
    """Return cover_id or isbn — whichever the dataset uses."""
    return str(book.get("cover_id") or book.get("isbn", ""))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",  default="data/index.json",
        help="Path to index JSON file (default: data/index.json)"
    )
    parser.add_argument(
        "--output", default="embeddings",
        help="Output directory for .npy files (default: embeddings)"
    )
    args = parser.parse_args()

    index_path     = Path(args.index)
    embeddings_dir = Path(args.output)
    row_index_path = embeddings_dir / "row_index.json"
    updated_index  = index_path.parent / f"{index_path.stem}_updated.json"

    if not index_path.exists():
        print(f"Error: index file not found at {index_path}")
        sys.exit(1)

    embeddings_dir.mkdir(exist_ok=True)

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    n = len(index)
    print(f"Index   : {index_path}")
    print(f"Output  : {embeddings_dir}/")
    print(f"Books   : {n}\n")

    clip_image_vecs = np.zeros((n, 512), dtype=np.float32)
    clip_text_vecs  = np.zeros((n, 512), dtype=np.float32)
    sentence_vecs   = np.zeros((n, 384), dtype=np.float32)
    color_vecs      = np.zeros((n, 15),  dtype=np.float32)
    row_index       = []

    for i, book in enumerate(index):
        img_path = book["filename"]
        print(f"[{i+1:03d}/{n}] {book['title'][:50]:<50}  ({book['genre']})")

        # ── OCR ──────────────────────────────────────────────────────────────
        try:
            ocr_text = extract_text(img_path)
        except Exception as e:
            print(f"         OCR failed: {e}")
            ocr_text = ""

        text_for_encoding = ocr_text if ocr_text.strip() else fallback_text(book)
        book["ocr_text"]  = ocr_text

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
            "id":       book_id(book),
            "title":    book["title"],
            "authors":  book["authors"],
            "genre":    book["genre"],
            "filename": book["filename"],
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(embeddings_dir / "clip_image.npy", clip_image_vecs)
    np.save(embeddings_dir / "clip_text.npy",  clip_text_vecs)
    np.save(embeddings_dir / "sentence.npy",   sentence_vecs)
    np.save(embeddings_dir / "colors.npy",     color_vecs)

    with open(row_index_path, "w", encoding="utf-8") as f:
        json.dump(row_index, f, indent=2, ensure_ascii=False)

    with open(updated_index, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done.")
    print(f"  clip_image.npy : {clip_image_vecs.shape}")
    print(f"  clip_text.npy  : {clip_text_vecs.shape}")
    print(f"  sentence.npy   : {sentence_vecs.shape}")
    print(f"  colors.npy     : {color_vecs.shape}")
    print(f"  row_index.json : {len(row_index)} entries")
    print(f"  updated index  : {updated_index}")


if __name__ == "__main__":
    main()