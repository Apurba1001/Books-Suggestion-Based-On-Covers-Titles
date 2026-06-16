"""
remove_books.py
---------------
Removes specific books from nyt_index.json by ISBN, and deletes
their cover image files. Use for manual curation cleanup.

Usage:
    uv run python src/data/remove_books.py
"""

import json
from pathlib import Path

INDEX_PATH = Path("data/nyt_index.json")

# ISBNs to remove — add more as needed
REMOVE_ISBNS = [
    # Audio duplicates (keep print edition, remove audio)
    "9783748087328",   # Dungeon Crawler Carl — audio-fiction
    "DBKADBL025354",   # The Deal — audio-fiction
    "9798217073443",   # Strangers — audio-nonfiction
    "9780593412701",   # The Body Keeps the Score — audio-nonfiction
    "9781668173107",   # Theo of Golden — audio-fiction
    "9780063204188",   # Remarkably Bright Creatures — audio-fiction
    "9798347810307",   # The Calamity Club — audio-fiction
    "9781668648230",   # The Land and Its People — audio-nonfiction
    "9781668657157",   # Liar's Kingdom — audio-nonfiction
    "9781668191569",   # Take Me to Your Leader — audio-nonfiction
    "9798217174072",   # London Falling — audio-nonfiction
    "9798217176397",   # Famesick — audio-nonfiction
    "9780063446557",   # Suicidal Empathy — audio-nonfiction
    "9780063360839",   # The Case for America — audio-nonfiction
    
    #print duplicates (same book, two print ISBNs)
    "9781250413581",   # The Final Target — hardcover duplicate
    
    #bad covers
    "6aEuAQAAIAAJ",   # Classic Memoirs (3-volume compilation, kept copy)
    "1jO1AAAAIAAJ",   # Classic Memoirs (3-volume compilation, duplicate copy)
]


def main():
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    print(f"Loaded {len(index)} books")

    removed = []
    kept    = []
    for book in index:
        if book.get("isbn") in REMOVE_ISBNS:
            removed.append(book)
        else:
            kept.append(book)

    for book in removed:
        path = Path(book["filename"])
        if path.exists():
            path.unlink()
            print(f"  deleted file : {path}")
        print(f"  removed entry: {book['title'][:60]}  ({book.get('isbn')})")

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done. {len(index)} → {len(kept)} books.")
    print("  Re-run build_embeddings.py to refresh the embedding index.")


if __name__ == "__main__":
    main()