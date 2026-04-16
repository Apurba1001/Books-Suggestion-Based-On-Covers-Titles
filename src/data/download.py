"""
download.py
-----------
Fetches book cover images from OpenLibrary and builds data/index.json.

Usage:
    python src/data/download.py

Output:
    data/covers/<genre>_<cover_id>.jpg   — one JPEG per book
    data/index.json                       — metadata for every downloaded book
"""

import os
import json
import time
import requests
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

COVERS_DIR = Path("data/covers")
INDEX_PATH  = Path("data/index.json")

# How many books to request per genre.
# OpenLibrary may return fewer if covers are missing — that's fine.
SUBJECTS = {
    "thriller":           45,
    "science_fiction":    45,
    "fantasy":            45,
    "romance":            45,
    "biography":          45,
    "horror":             45,
    "historical_fiction": 45,
    "mystery":            45,
    "non-fiction":        45,
    "literary_fiction":   45,
}

# Minimum file size in bytes — OpenLibrary returns a 1×1 grey pixel
# placeholder (~807 bytes) when no cover exists. We skip those.
MIN_IMAGE_BYTES = 5_000

# Polite delay between requests (seconds)
REQUEST_DELAY = 0.3


# ── Helpers ────────────────────────────────────────────────────────────────────

def fetch_works(subject: str, limit: int) -> list[dict]:
    """Return a list of works from the OpenLibrary Subjects API."""
    url = f"https://openlibrary.org/subjects/{subject}.json?limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("works", [])
    except Exception as e:
        print(f"  [warn] could not fetch subject '{subject}': {e}")
        return []


def fetch_cover(cover_id: int, dest: Path) -> bool:
    """
    Download a cover image to dest. Returns True on success.
    Skips placeholder images (too small) and network errors.
    """
    url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        if len(r.content) < MIN_IMAGE_BYTES:
            return False  # placeholder pixel, skip
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  [warn] could not download cover {cover_id}: {e}")
        return False


def dedup(records: list[dict]) -> list[dict]:
    """Remove duplicate cover_ids, keeping the first occurrence."""
    seen = set()
    out  = []
    for r in records:
        if r["cover_id"] not in seen:
            seen.add(r["cover_id"])
            out.append(r)
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    COVERS_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    total_downloaded = 0

    for genre, limit in SUBJECTS.items():
        print(f"\n── {genre} (requesting {limit}) ──")
        works = fetch_works(genre, limit)
        genre_count = 0

        for book in works:
            cover_id = book.get("cover_id")
            if not cover_id:
                continue

            dest = COVERS_DIR / f"{genre}_{cover_id}.jpg"

            # Skip if already downloaded (allows re-running the script safely)
            if dest.exists():
                print(f"  skip  {dest.name} (already exists)")
                record_exists = any(r["cover_id"] == cover_id for r in all_records)
                if not record_exists:
                    all_records.append({
                        "cover_id": cover_id,
                        "title":    book.get("title", ""),
                        "authors":  [a["name"] for a in book.get("authors", [])],
                        "genre":    genre,
                        "filename": str(dest),
                    })
                genre_count += 1
                continue

            ok = fetch_cover(cover_id, dest)
            if not ok:
                continue

            all_records.append({
                "cover_id": cover_id,
                "title":    book.get("title", ""),
                "authors":  [a["name"] for a in book.get("authors", [])],
                "genre":    genre,
                "filename": str(dest),
            })

            genre_count += 1
            total_downloaded += 1
            print(f"  [{genre_count:02d}] {dest.name}  —  {book.get('title', '?')[:50]}")
            time.sleep(REQUEST_DELAY)

        print(f"  → {genre_count} covers for {genre}")

    # Deduplicate across genres (a book may appear in multiple subjects)
    before = len(all_records)
    all_records = dedup(all_records)
    dupes = before - len(all_records)

    # Save index
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done.")
    print(f"  Total in index : {len(all_records)}")
    print(f"  Duplicates removed : {dupes}")
    print(f"  Newly downloaded   : {total_downloaded}")
    print(f"  Index saved to     : {INDEX_PATH}")


if __name__ == "__main__":
    main()
