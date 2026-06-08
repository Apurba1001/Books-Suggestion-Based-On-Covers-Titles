"""
download_nyt.py
---------------
Fetches NYT bestseller books and classifies them into your 10 genres
using the NYT list name + description text. No OpenLibrary subject
lookup needed. Covers are downloaded directly from NYT image URLs.

Deduplicates against:
  - books already in data/nyt_index.json (previous runs)
  - books already in data/index.json (original OpenLibrary dataset)

Usage:
    uv run python src/data/download_nyt.py
"""

import os
import json
import time
import requests
from pathlib import Path
from collections import defaultdict, Counter
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

NYT_API_KEY      = os.environ.get("NYT_API_KEY", "")
COVERS_DIR       = Path("data/covers_nyt")
INDEX_PATH       = Path("data/nyt_index.json")
OL_INDEX_PATH    = Path("data/index.json")       # original dataset
TARGET_PER_GENRE = 25
MIN_IMAGE_BYTES  = 5_000

# All valid NYT list slugs from your account
NYT_LISTS = [
    "combined-print-and-e-book-fiction",
    "combined-print-and-e-book-nonfiction",
    "hardcover-fiction",
    "hardcover-nonfiction",
    "trade-fiction-paperback",
    "young-adult-hardcover",
    "graphic-books-and-manga",
    "business-books",
    "advice-how-to-and-miscellaneous",
    "childrens-middle-grade-hardcover",
    "young-adult-paperback-monthly",
    "audio-fiction",
    "audio-nonfiction",
]

GENRE_PRIORITY = [
    "horror",
    "science_fiction",
    "fantasy",
    "romance",
    "mystery",
    "thriller",
    "historical_fiction",
    "biography",
    "non-fiction",
    "literary_fiction",
]

# Rules applied in order — first match wins
# Each rule: (genre, [keywords to match against list_name + description])
CLASSIFICATION_RULES = [
    ("horror",             ["horror", "ghost", "haunted", "supernatural", "vampire", "zombie", "occult", "terror", "evil", "demon"]),
    ("science_fiction",    ["science fiction", "sci-fi", "dystopia", "space", "alien", "futuristic", "robot", "cyberpunk", "time travel", "apocalyptic"]),
    ("fantasy",            ["fantasy", "magic", "wizard", "dragon", "witch", "sorcerer", "mythical", "fairy", "fae", "enchanted", "realm", "quest"]),
    ("romance",            ["romance", "romantic", "love story", "love stories", "falling in love", "relationship", "wedding", "affair"]),
    ("mystery",            ["mystery", "detective", "whodunit", "murder mystery", "cozy", "private investigator", "clues", "sleuth"]),
    ("thriller",           ["thriller", "suspense", "conspiracy", "espionage", "spy", "assassin", "psychological", "chase", "kidnap", "hostage"]),
    ("historical_fiction", ["historical", "world war", "civil war", "medieval", "ancient", "century", "era", "colonial", "tudor", "victorian"]),
    ("biography",          ["biography", "autobiography", "memoir", "life of", "true story", "her story", "his story", "growing up", "personal journey"]),
    ("non-fiction",        ["nonfiction", "non-fiction", "true crime", "self-help", "guide", "science", "history", "politics", "economics", "how to", "business"]),
]
# literary_fiction is the fallback for anything from a fiction list that didn't match


# ── NYT list → genre hint ──────────────────────────────────────────────────────
# Used as a secondary signal alongside description keywords

LIST_GENRE_HINT = {
    "combined-print-and-e-book-fiction":    "fiction",
    "hardcover-fiction":                    "fiction",
    "trade-fiction-paperback":              "fiction",
    "audio-fiction":                        "fiction",
    "combined-print-and-e-book-nonfiction": "nonfiction",
    "hardcover-nonfiction":                 "nonfiction",
    "audio-nonfiction":                     "nonfiction",
    "advice-how-to-and-miscellaneous":      "nonfiction",
    "business-books":                       "nonfiction",
    "young-adult-hardcover":                "fiction",
    "young-adult-paperback-monthly":        "fiction",
    "childrens-middle-grade-hardcover":     "fiction",
    "graphic-books-and-manga":              "fiction",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def classify(book: dict) -> str:
    """
    Classify a book into one of 10 genres using:
    1. Description text keywords
    2. NYT list name keywords
    3. Fallback based on list type (fiction/nonfiction)
    """
    description = (book.get("description") or "").lower()
    title       = (book.get("title") or "").lower()
    nyt_list    = (book.get("nyt_list") or "").lower()
    blob        = f"{title} {description} {nyt_list}"

    for genre, keywords in CLASSIFICATION_RULES:
        for kw in keywords:
            if kw in blob:
                return genre

    # Fallback
    hint = LIST_GENRE_HINT.get(book.get("nyt_list", ""), "fiction")
    if hint == "nonfiction":
        return "non-fiction"
    return "literary_fiction"


def fetch_nyt_overview() -> list[dict]:
    print("Fetching NYT overview (single request)...")
    url = f"https://api.nytimes.com/svc/books/v3/lists/overview.json?api-key={NYT_API_KEY}"
    r   = requests.get(url, timeout=15)
    r.raise_for_status()
    lists     = r.json()["results"]["lists"]
    all_books = []
    for lst in lists:
        slug = lst["list_name_encoded"]
        if slug in NYT_LISTS:
            for book in lst.get("books", []):
                book["nyt_list"] = slug
                all_books.append(book)
    print(f"  {len(all_books)} books fetched")
    return all_books


def load_existing_isbns() -> set[str]:
    """
    Load all ISBNs already downloaded in previous runs or in the
    original OpenLibrary dataset, to avoid duplicates.
    """
    seen = set()

    # Previous NYT runs
    if INDEX_PATH.exists():
        with open(INDEX_PATH, encoding="utf-8") as f:
            for book in json.load(f):
                isbn = book.get("isbn", "")
                if isbn:
                    seen.add(isbn)
        print(f"  {len(seen)} ISBNs already in nyt_index.json")

    # Original OpenLibrary dataset — no ISBNs but filenames overlap by title
    # We can't cross-match perfectly so we just skip ISBN duplication here.
    # If you later merge both indexes, deduplicate by title+author instead.

    return seen


def fetch_cover(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        if len(r.content) < MIN_IMAGE_BYTES:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"    [warn] cover failed: {e}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not NYT_API_KEY:
        print("Error: NYT_API_KEY not set in .env")
        return

    COVERS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing records to preserve them
    existing_records = []
    if INDEX_PATH.exists():
        with open(INDEX_PATH, encoding="utf-8") as f:
            existing_records = json.load(f)

    existing_isbns = load_existing_isbns()

    # Fetch NYT books
    nyt_books = fetch_nyt_overview()

    # Dedup within NYT response by ISBN
    seen_this_run = set()
    unique_books  = []
    for book in nyt_books:
        isbn = book.get("primary_isbn13") or book.get("primary_isbn10", "")
        if not isbn or isbn in existing_isbns or isbn in seen_this_run:
            continue
        seen_this_run.add(isbn)
        unique_books.append((isbn, book))

    print(f"  {len(unique_books)} new books after dedup\n")

    # Classify into genres
    genre_buckets = defaultdict(list)
    for isbn, book in unique_books:
        genre = classify(book)
        genre_buckets[genre].append((isbn, book))

    print("Genre distribution after classification:")
    for g in GENRE_PRIORITY:
        print(f"  {g:<25} {len(genre_buckets[g])}")

    # Enforce exclusivity by priority order
    claimed      = set()
    exclusive    = defaultdict(list)
    for genre in GENRE_PRIORITY:
        for isbn, book in genre_buckets[genre]:
            if isbn not in claimed:
                claimed.add(isbn)
                exclusive[genre].append((isbn, book))

    # Count what we already have per genre from existing records
    existing_per_genre = Counter(r["genre"] for r in existing_records)

    # Download covers and build new records
    new_records = []
    for genre in GENRE_PRIORITY:
        already     = existing_per_genre.get(genre, 0)
        still_need  = max(0, TARGET_PER_GENRE - already)
        candidates  = exclusive[genre][:still_need]

        print(f"\n── {genre}  (have {already}, need {still_need}, candidates {len(candidates)}) ──")

        for isbn, book in candidates:
            cover_url = book.get("book_image", "")
            dest      = COVERS_DIR / f"{genre}_{isbn}.jpg"

            if dest.exists():
                print(f"  skip {dest.name}")
            elif cover_url:
                ok = fetch_cover(cover_url, dest)
                if not ok:
                    continue
            else:
                print(f"  [warn] no image for {book.get('title','?')[:40]}")
                continue

            record = {
                "isbn":          isbn,
                "title":         book.get("title", ""),
                "authors":       [book.get("author", "")],
                "genre":         genre,
                "filename":      str(dest),
                "nyt_rank":      book.get("rank", 0),
                "nyt_list":      book.get("nyt_list", ""),
                "description":   book.get("description", ""),
                "publisher":     book.get("publisher", ""),
                "weeks_on_list": book.get("weeks_on_list", 0),
            }
            new_records.append(record)
            print(f"  ✓ {record['title'][:55]}")

    # Merge with existing and save
    all_records = existing_records + new_records
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done.")
    print(f"  New this run  : {len(new_records)}")
    print(f"  Total records : {len(all_records)}")
    print(f"  Index saved to: {INDEX_PATH}")

    final_counts = Counter(r["genre"] for r in all_records)
    print("\n  Final per genre:")
    for genre in GENRE_PRIORITY:
        n         = final_counts.get(genre, 0)
        shortfall = TARGET_PER_GENRE - n
        flag      = f"  ⚠ {shortfall} short" if shortfall > 0 else "  ✓"
        print(f"    {genre:<25} {n:>3}{flag}")


if __name__ == "__main__":
    main()