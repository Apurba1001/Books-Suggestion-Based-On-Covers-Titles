"""
redownload_covers.py
--------------------
For books in the index that have no cover file on disk,
try to re-download from OpenLibrary by ISBN.

Usage:
    uv run python src/data/redownload_covers.py
"""

import json
import time
import requests
from pathlib import Path
from collections import Counter

INDEX_PATH      = Path("data/nyt_index.json")
BACKUP_INDEX    = Path("data/nyt_index_before_placeholder_removal.json")
COVERS_DIR      = Path("data/covers_nyt")
MIN_IMAGE_BYTES = 15_000  # stricter than before


def fetch_from_openlibrary(isbn: str, dest: Path) -> bool:
    url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        if len(r.content) < MIN_IMAGE_BYTES:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def fetch_from_google(isbn: str, dest: Path) -> bool:
    """Try Google Books but with strict size check."""
    try:
        r = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params={"q": f"isbn:{isbn}", "maxResults": 1},
            timeout=10
        )
        items = r.json().get("items", [])
        if not items:
            return False
        links = items[0].get("volumeInfo", {}).get("imageLinks", {})
        url = links.get("large") or links.get("medium") or links.get("thumbnail")
        if not url:
            return False
        url = url.replace("zoom=1", "zoom=3").replace("&edge=curl", "")
        img = requests.get(url, timeout=10)
        if len(img.content) < MIN_IMAGE_BYTES:
            return False
        dest.write_bytes(img.content)
        return True
    except Exception:
        return False


def main():
    # Load the backup (pre-removal) index to get the removed books
    if not BACKUP_INDEX.exists():
        print(f"No backup index at {BACKUP_INDEX}")
        print("Creating one from the placeholder list...")
        # Fall back: scan for ISBNs referenced in embeddings but missing from current index
        print("Run this script right after removing placeholders.")
        return

    with open(BACKUP_INDEX, encoding="utf-8") as f:
        old_index = json.load(f)
    with open(INDEX_PATH, encoding="utf-8") as f:
        current_index = json.load(f)

    current_isbns = {b.get("isbn") for b in current_index}
    removed = [b for b in old_index if b.get("isbn") not in current_isbns]

    print(f"Current index : {len(current_index)} books")
    print(f"Removed books : {len(removed)} to re-download\n")

    recovered = []
    failed    = []

    for i, book in enumerate(removed):
        isbn = book.get("isbn", "")
        title = book.get("title", "?")
        dest = COVERS_DIR / f"{book['genre']}_{isbn}.jpg"

        print(f"[{i+1:03d}/{len(removed)}] {title[:45]:<45} ", end="", flush=True)

        # Try OpenLibrary first
        if fetch_from_openlibrary(isbn, dest):
            print("✓ OpenLibrary")
            book["filename"] = str(dest)
            recovered.append(book)
            time.sleep(0.3)
            continue

        time.sleep(0.3)

        # Try Google Books with stricter filter
        if fetch_from_google(isbn, dest):
            print("✓ Google Books")
            book["filename"] = str(dest)
            recovered.append(book)
            time.sleep(0.3)
            continue

        print("✗ no cover found")
        failed.append(book)
        time.sleep(0.3)

    # Merge recovered books back into index
    merged = current_index + recovered
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done.")
    print(f"  Recovered : {len(recovered)}")
    print(f"  Failed    : {len(failed)}")
    print(f"  Total now : {len(merged)}")

    counts = Counter(b["genre"] for b in merged)
    print("\n  Per genre:")
    for g in sorted(counts.keys()):
        print(f"    {g:<25} {counts[g]}")

    if failed:
        print(f"\n  Books without covers ({len(failed)}):")
        for b in failed:
            print(f"    {b.get('isbn','?'):<20} {b['title'][:45]}")


if __name__ == "__main__":
    main()