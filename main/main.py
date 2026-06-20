"""
main.py
-------
End-to-end pipeline test. Takes a book cover image and prints the
top-3 recommendations for each track to the terminal.

Usage:
    uv run python main.py <path_to_cover_image> [--embeddings embeddings_nyt]

Examples:
    uv run python main.py data/covers/thriller_8739161.jpg
    uv run python main.py data/covers_nyt/horror_9780385121675.jpg --embeddings embeddings_nyt
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.clip_encoder import encode_image, encode_text, set_model as set_clip_model
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


def run(image_path: str, embeddings_dir: str):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: file not found — {image_path}")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  Query image : {path.name}")
    print(f"  Embeddings  : {embeddings_dir}")
    print(f"{'═'*60}\n")

    print("Extracting features from query image...")

    print("  [1/4] OCR...")
    ocr_text = extract_text(str(path))
    text_for_encoding = ocr_text if ocr_text.strip() else path.stem.replace("_", " ")
    print(f"        OCR detected: '{ocr_text[:80]}'" if ocr_text else "        OCR: nothing detected")

    print("  [2/4] CLIP image encoding...")
    clip_img_vec = encode_image(str(path))

    print("  [3/4] CLIP text encoding...")
    clip_txt_vec = encode_text(text_for_encoding)

    print("  [4/4] Sentence encoding + color palette...")
    sentence_vec = encode_sentence(text_for_encoding)
    color_vec    = extract_palette(str(path))

    print("\nSearching index...\n")
    load_index(embeddings_dir)
    
    book_id = Path(image_path).stem.split("_", 1)[-1]  # e.g. "horror_9780385121675" → "9780385121675"

    results = search(
        clip_image_vec = clip_img_vec,
        clip_text_vec  = clip_txt_vec,
        sentence_vec   = sentence_vec,
        color_vec      = color_vec,
        k              = TOP_K,
        exclude_id     = book_id
    )

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
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to cover image")
    parser.add_argument("--embeddings", default="embeddings",
                        help="Embeddings directory (default: embeddings)")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                    help="CLIP model ID (must match the embeddings)")
    args = parser.parse_args()
    set_clip_model(args.clip_model)
    run(args.image, args.embeddings)