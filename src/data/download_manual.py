"""
download_manual.py
------------------
Downloads covers for a hardcoded list of specific well-known books
by ISBN. Used to fill gaps where automated queries return poor results.

Fetches cover from OpenLibrary first, falls back to Google Books.
Appends to data/nyt_index.json without duplicating existing titles.

Usage:
    uv run python src/data/download_manual.py
"""

import os
import json
import time
import requests
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "")
COVERS_DIR     = Path("data/covers_nyt")
INDEX_PATH     = Path("data/nyt_index.json")
MIN_IMAGE_BYTES = 5_000

# ── Hardcoded ISBNs ────────────────────────────────────────────────────────────
# Format: { "genre": [ ("ISBN-13", "Title", "Author"), ... ] }
# Add as many as you want — the script skips any already in the index.

MANUAL_BOOKS = {
    "horror": [
        ("9780385121675", "The Shining",                    "Stephen King"),
        ("9781501156700", "It",                             "Stephen King"),
        ("9780385333481", "Pet Sematary",                   "Stephen King"),
        ("9780385472579", "Misery",                         "Stephen King"),
        ("9780307743657", "Doctor Sleep",                   "Stephen King"),
        ("9780140817737", "Dracula",                        "Bram Stoker"),
        ("9780141439471", "Frankenstein",                   "Mary Shelley"),
        ("9780143039952", "The Haunting of Hill House",     "Shirley Jackson"),
        ("9780143105435", "We Have Always Lived in the Castle", "Shirley Jackson"),
        ("9780316769174", "The Exorcist",                   "William Peter Blatty"),
        ("9780060935474", "American Gods",                  "Neil Gaiman"),
    ],
    "historical_fiction": [
        ("9780451205766", "The Name of the Rose",           "Umberto Eco"),
        ("9780060931957", "Pillars of the Earth",           "Ken Follett"),
        ("9780007320271", "Wolf Hall",                      "Hilary Mantel"),
        ("9780525555360", "The Tattooist of Auschwitz",     "Heather Morris"),
        ("9780062409850", "All the Light We Cannot See",    "Anthony Doerr"),
        ("9780385490818", "Schindler's List",               "Thomas Keneally"),
        ("9780143124542", "The Book Thief",                 "Markus Zusak"),
        ("9780062409843", "The Alice Network",              "Kate Quinn"),
        ("9780062838667", "The Nightingale",                "Kristin Hannah"),
    ],
    "biography": [
    ("9781451648539", "Steve Jobs",              "Walter Isaacson"),
    ("9781501156700", "Einstein",                "Walter Isaacson"),
    ("9781501121586", "Becoming",                "Michelle Obama"),
    ("9780316017923", "The Immortal Life of Henrietta Lacks", "Rebecca Skloot"),
    ("9780316346627", "Born a Crime",            "Trevor Noah"),
    ("9780525559474", "I Am Malala",             "Malala Yousafzai"),
    ("9781476748542", "Leonardo da Vinci",       "Walter Isaacson"),
    ("9780385490818", "Long Walk to Freedom",    "Nelson Mandela"),
    ("9780679720201", "The Diary of a Young Girl", "Anne Frank"),
    ],
}


# ── Cover fetching ─────────────────────────────────────────────────────────────

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
    if not GOOGLE_API_KEY:
        return False
    try:
        r = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params={"q": f"isbn:{isbn}", "key": GOOGLE_API_KEY},
            timeout=10
        )
        items = r.json().get("items", [])
        if not items:
            return False
        image_links = items[0].get("volumeInfo", {}).get("imageLinks", {})
        url = (
            image_links.get("large") or
            image_links.get("medium") or
            image_links.get("thumbnail")
        )
        if not url:
            return False
        url = url.replace("zoom=1", "zoom=3").replace("&edge=curl", "")
        img = requests.get(url, timeout=10)
        img.raise_for_status()
        if len(img.content) < MIN_IMAGE_BYTES:
            return False
        dest.write_bytes(img.content)
        return True
    except Exception:
        return False


def fetch_cover(isbn: str, dest: Path) -> bool:
    """Try OpenLibrary first, fall back to Google Books."""
    if fetch_from_openlibrary(isbn, dest):
        return True
    time.sleep(0.3)
    return fetch_from_google(isbn, dest)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    COVERS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing index
    existing_records = []
    if INDEX_PATH.exists():
        with open(INDEX_PATH, encoding="utf-8") as f:
            existing_records = json.load(f)

    existing_titles = {r["title"].lower().strip() for r in existing_records}
    existing_isbns  = {r.get("isbn", "") for r in existing_records}

    print(f"Existing records: {len(existing_records)}")
    existing_counts = Counter(r["genre"] for r in existing_records)
    print("Current counts:")
    for genre, count in existing_counts.items():
        print(f"  {genre:<25} {count}")

    new_records = []

    for genre, books in MANUAL_BOOKS.items():
        print(f"\n── {genre} ──")
        for isbn, title, author in books:
            if isbn in existing_isbns:
                print(f"  skip (ISBN exists) {title}")
                continue
            if title.lower() in existing_titles:
                print(f"  skip (title exists) {title}")
                continue

            dest = COVERS_DIR / f"{genre}_{isbn}.jpg"

            if dest.exists():
                print(f"  skip (file exists) {title}")
            else:
                ok = fetch_cover(isbn, dest)
                if not ok:
                    print(f"  [warn] no cover found for {title} ({isbn})")
                    continue
                time.sleep(0.3)

            record = {
                "isbn":     isbn,
                "title":    title,
                "authors":  [author],
                "genre":    genre,
                "filename": str(dest),
                "source":   "manual",
            }
            new_records.append(record)
            existing_titles.add(title.lower())
            existing_isbns.add(isbn)
            print(f"  ✓ {title}")

    # Merge and save
    all_records = existing_records + new_records
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done.")
    print(f"  Added   : {len(new_records)}")
    print(f"  Total   : {len(all_records)}")

    final_counts = Counter(r["genre"] for r in all_records)
    print("\n  Final per genre:")
    for genre in MANUAL_BOOKS.keys():
        n = final_counts.get(genre, 0)
        print(f"    {genre:<25} {n}")


if __name__ == "__main__":
    main()