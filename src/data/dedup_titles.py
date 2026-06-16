"""
dedup_titles.py
---------------
Scans nyt_index.json for near-duplicate titles (same book, different
edition/ISBN/cover) using normalized title matching. Flags duplicates
and lets you choose which copy to keep based on image quality.

Usage:
    uv run python src/data/dedup_titles.py            # dry run, just reports
    uv run python src/data/dedup_titles.py --apply     # actually removes duplicates
"""

import json
import re
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

INDEX_PATH = Path("data/nyt_index.json")


def normalize_title(title: str) -> str:
    """Strip punctuation, lowercase, collapse whitespace, drop subtitles after colon."""
    title = title.split(":")[0]               # drop subtitles
    title = title.lower().strip()
    title = re.sub(r'[^\w\s]', '', title)      # strip punctuation
    title = re.sub(r'\s+', ' ', title)         # collapse whitespace
    return title.strip()


def image_quality_score(path: str) -> float:
    """
    Rough quality proxy: pixel std (higher = more visual detail/contrast)
    combined with resolution. Used to pick the better cover when duplicates
    are found.
    """
    try:
        img = Image.open(path)
        w, h = img.size
        arr = np.array(img.convert("RGB"))
        std = arr.std()
        resolution_score = min(w * h / (300 * 450), 2.0)  # cap bonus at 2x
        return float(std * resolution_score)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually remove duplicates (default: dry run)")
    args = parser.parse_args()

    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    print(f"Loaded {len(index)} books\n")

    # Group by normalized title
    groups = defaultdict(list)
    for book in index:
        key = normalize_title(book.get("title", ""))
        groups[key].append(book)

    duplicates_found = {k: v for k, v in groups.items() if len(v) > 1}

    if not duplicates_found:
        print("✓ No near-duplicate titles found.")
        return

    print(f"Found {len(duplicates_found)} groups with duplicate titles:\n")

    to_remove = []

    for norm_title, books in duplicates_found.items():
        print(f"── \"{books[0]['title']}\" ──")
        scored = []
        for book in books:
            score = image_quality_score(book["filename"])
            scored.append((score, book))
            print(f"  {book.get('isbn','?'):<16} score={score:6.1f}  {book['filename']}")

        # Keep the highest quality score, mark rest for removal
        scored.sort(key=lambda x: x[0], reverse=True)
        keep      = scored[0][1]
        remove    = [b for _, b in scored[1:]]

        print(f"  → keep:   {keep.get('isbn','?')}  ({keep['filename']})")
        for r in remove:
            print(f"  → remove: {r.get('isbn','?')}  ({r['filename']})")
            to_remove.append(r)
        print()

    print(f"Total to remove: {len(to_remove)}\n")

    if args.apply:
        remove_isbns = {b.get("isbn") for b in to_remove}
        cleaned = [b for b in index if b.get("isbn") not in remove_isbns]

        # Delete the actual cover image files
        for b in to_remove:
            p = Path(b["filename"])
            if p.exists():
                p.unlink()
                print(f"  deleted file: {p}")

        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Applied. {len(index)} → {len(cleaned)} books.")
        print("  Re-run build_embeddings.py to refresh the index.")
    else:
        print("Dry run only — no changes made. Re-run with --apply to remove duplicates.")


if __name__ == "__main__":
    main()