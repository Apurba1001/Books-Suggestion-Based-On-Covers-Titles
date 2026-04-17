"""
main.py
-------
End-to-end pipeline test. Takes a book cover image and prints the
top-3 recommendations for each track to the terminal.

Usage:
    python main.py <path_to_cover_image>

Example:
    python main.py data/covers/thriller_8739161.jpg
"""

import sys
from pathlib import Path

# ── Allow src imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.features.clip_encoder import encode_image, encode_text
from src.features.ocr          import extract_text
from src.features.colors       import extract_palette
from src.features.text_encoder import encode as encode_sentence
from src.search.retrieval      import load_index, search

TOP_K = 3

TRACK_LABELS = {
    "visual":       "Visual similarity   (cover style & imagery)",
    "cross_modal":  "Cross-modal         (OCR text → image search)",
    "semantic":     "Semantic similarity (theme & description)",
    "color":        "Color palette       (dominant colors)",
}


def run(image_path: str):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: file not found — {image_path}")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  Query image: {path.name}")
    print(f"{'═'*60}\n")

    # ── Feature extraction ────────────────────────────────────────────────────
    print("Extracting features from query image...")

    print("  [1/4] OCR...")
    ocr_text = extract_text(str(path))
    text_for_encoding = ocr_text if ocr_text.strip() else path.stem.replace("_", " ")
    print(f"        OCR detected: '{ocr_text[:80]}'" if ocr_text else "        OCR: nothing detected, using filename")

    print("  [2/4] CLIP image encoding...")
    clip_img_vec = encode_image(str(path))

    print("  [3/4] CLIP text encoding...")
    clip_txt_vec = encode_text(text_for_encoding)

    print("  [4/4] Sentence encoding + color palette...")
    sentence_vec = encode_sentence(text_for_encoding)
    color_vec    = extract_palette(str(path))

    # ── Search ────────────────────────────────────────────────────────────────
    print("\nSearching index...\n")
    load_index()

    results = search(
        clip_image_vec = clip_img_vec,
        clip_text_vec  = clip_txt_vec,
        sentence_vec   = sentence_vec,
        color_vec      = color_vec,
        k              = TOP_K,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    for track, label in TRACK_LABELS.items():
        hits = results.get(track, [])
        print(f"┌─ {label}")
        if not hits:
            print("│  No results.")
        for hit in hits:
            authors = ", ".join(hit["authors"]) if hit["authors"] else "Unknown"
            print(f"│  #{hit['rank']}  [{hit['score']:.3f}]  {hit['title'][:45]:<45}  by {authors[:30]}")
            print(f"│       Genre: {hit['genre']}   File: {Path(hit['filename']).name}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_cover_image>")
        print("Example: python main.py data/covers/thriller_8739161.jpg")
        sys.exit(1)

    run(sys.argv[1])