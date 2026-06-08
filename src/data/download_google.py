"""
download_google.py
------------------
Fetches well-known book covers from Google Books API by genre.
Tops up genres that are short in nyt_index.json to reach TARGET_PER_GENRE.
Deduplicates against both nyt_index.json and index.json.

Usage:
    uv run python src/data/download_google.py

Requires:
    GOOGLE_BOOKS_API_KEY in .env file
"""

import os
import json
import time
import requests
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

GOOGLE_API_KEY   = os.environ.get("GOOGLE_BOOKS_API_KEY", "")
COVERS_DIR       = Path("data/covers_nyt")
NYT_INDEX_PATH   = Path("data/nyt_index.json")
OL_INDEX_PATH    = Path("data/index.json")
TARGET_PER_GENRE = 25
MIN_IMAGE_BYTES  = 5_000
REQUEST_DELAY    = 0.5

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

# Tighter queries — anchor on known authors + genre to get actual novels
# Multiple queries tried in order until target is reached
GENRE_QUERIES = {
    "horror": [
        "inauthor:Stephen King subject:horror",
        "inauthor:Dean Koontz subject:horror",
        "inauthor:Peter Straub subject:horror",
        "inauthor:Shirley Jackson subject:horror",
        "inauthor:Joe Hill subject:horror",
        "subject:horror fiction novel",
    ],
    "science_fiction": [
        "inauthor:Philip K Dick subject:science fiction",
        "inauthor:Isaac Asimov subject:science fiction",
        "inauthor:Arthur C Clarke subject:science fiction",
        "inauthor:Frank Herbert subject:science fiction",
        "inauthor:Ursula Le Guin subject:science fiction",
        "inauthor:Andy Weir subject:science fiction",
        "subject:science fiction novel dystopia",
    ],
    "fantasy": [
        "inauthor:Brandon Sanderson subject:fantasy",
        "inauthor:Patrick Rothfuss subject:fantasy",
        "inauthor:Robin Hobb subject:fantasy",
        "inauthor:Neil Gaiman subject:fantasy",
        "inauthor:Terry Pratchett subject:fantasy",
        "inauthor:George R R Martin subject:fantasy",
        "subject:epic fantasy novel magic",
    ],
    "romance": [
        "inauthor:Nicholas Sparks subject:romance",
        "inauthor:Nora Roberts subject:romance",
        "inauthor:Colleen Hoover subject:romance",
        "inauthor:Julia Quinn subject:romance",
        "inauthor:Emily Henry subject:romance",
        "subject:romance novel love story",
    ],
    "mystery": [
        "inauthor:Agatha Christie subject:mystery",
        "inauthor:Gillian Flynn subject:mystery",
        "inauthor:Tana French subject:mystery",
        "inauthor:Michael Connelly subject:mystery",
        "inauthor:Lee Child subject:mystery",
        "subject:mystery detective novel fiction",
    ],
    "thriller": [
        "inauthor:John Grisham subject:thriller",
        "inauthor:Tom Clancy subject:thriller",
        "inauthor:Lee Child subject:thriller",
        "inauthor:James Patterson subject:thriller",
        "inauthor:Dan Brown subject:thriller",
        "inauthor:Harlan Coben subject:thriller",
        "subject:thriller suspense novel fiction",
    ],
    "historical_fiction": [
        "inauthor:Ken Follett subject:historical fiction",
        "inauthor:Hilary Mantel subject:historical fiction",
        "inauthor:Anthony Burgess subject:historical fiction",
        "inauthor:Bernard Cornwell subject:historical fiction",
        "inauthor:Philippa Gregory subject:historical fiction",
        "subject:historical fiction novel war",
    ],
    "biography": [
        "inauthor:Walter Isaacson subject:biography",
        "inauthor:Robert Caro subject:biography",
        "inauthor:Erik Larson subject:biography",
        "inauthor:David McCullough subject:biography",
        "subject:biography memoir autobiography personal",
    ],
    "non-fiction": [
        "inauthor:Malcolm Gladwell subject:nonfiction",
        "inauthor:Yuval Noah Harari subject:nonfiction",
        "inauthor:Michael Lewis subject:nonfiction",
        "inauthor:Erik Larson subject:true crime",
        "subject:true crime nonfiction narrative",
        "subject:popular science nonfiction",
    ],
    "literary_fiction": [
        "inauthor:Cormac McCarthy subject:fiction",
        "inauthor:Donna Tartt subject:fiction",
        "inauthor:Jonathan Franzen subject:fiction",
        "inauthor:Zadie Smith subject:literary fiction",
        "inauthor:Kazuo Ishiguro subject:literary fiction",
        "subject:literary fiction novel contemporary",
    ],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_existing_titles() -> set[str]:
    titles = set()
    for path in [NYT_INDEX_PATH, OL_INDEX_PATH]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for book in json.load(f):
                    title = book.get("title", "").lower().strip()
                    if title:
                        titles.add(title)
    print(f"  {len(titles)} existing titles loaded for dedup")
    return titles


def load_nyt_records() -> list[dict]:
    if NYT_INDEX_PATH.exists():
        with open(NYT_INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def fetch_google_books(query: str, max_results: int = 40) -> list[dict]:
    results = []
    start   = 0

    while len(results) < max_results:
        batch = min(40, max_results - len(results))
        try:
            r = requests.get(
                "https://www.googleapis.com/books/v1/volumes",
                params={
                    "q":            query,
                    "orderBy":      "relevance",
                    "maxResults":   batch,
                    "startIndex":   start,
                    "printType":    "books",
                    "langRestrict": "en",
                    "lr":           "lang_en",   # force English content
                    "key":          GOOGLE_API_KEY,
                },
                timeout=10
            )
            r.raise_for_status()
            items = r.json().get("items", [])
            if not items:
                break
            results.extend(items)
            start += len(items)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"    [warn] Google Books query failed: {e}")
            break

    return results


def is_reference_book(info: dict) -> bool:
    """
    Filter out academic/reference/non-novel books.
    Returns True if the book should be skipped.
    """
    title       = (info.get("title") or "").lower()
    description = (info.get("description") or "").lower()
    categories  = [c.lower() for c in info.get("categories", [])]

    # Skip obvious reference material
    reference_signals = [
        "dictionary", "encyclopedia", "cyclopedia", "anthology",
        "proceedings", "bulletin", "transactions", "magazine",
        "journal", "handbook", "textbook", "academic", "study guide",
        "how to write", "writing guide", "companion to",
        "introduction to", "history of", "criticism",
        "zeitschrift", "logik", "untersuchung",   # German language leakage
    ]
    for signal in reference_signals:
        if signal in title:
            return True

    # Skip if page count is suspiciously low (pamphlets) or very high (reference)
    page_count = info.get("pageCount", 200)
    if page_count < 80 or page_count > 1500:
        return True

    # Skip if categories suggest reference
    for cat in categories:
        if any(s in cat for s in ["reference", "study", "education", "juvenile"]):
            return True

    return False


def extract_isbn(volume: dict) -> str | None:
    for id_info in volume.get("volumeInfo", {}).get("industryIdentifiers", []):
        if id_info.get("type") == "ISBN_13":
            return id_info["identifier"]
    for id_info in volume.get("volumeInfo", {}).get("industryIdentifiers", []):
        if id_info.get("type") == "ISBN_10":
            return id_info["identifier"]
    return None


def fetch_cover(url: str, dest: Path) -> bool:
    url = url.replace("zoom=1", "zoom=3").replace("&edge=curl", "")
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
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_BOOKS_API_KEY not set in .env")
        return

    COVERS_DIR.mkdir(parents=True, exist_ok=True)

    existing_titles  = load_existing_titles()
    existing_records = load_nyt_records()
    existing_counts  = Counter(r["genre"] for r in existing_records)

    print(f"\nCurrent counts per genre:")
    for genre in GENRE_PRIORITY:
        n = existing_counts.get(genre, 0)
        print(f"  {genre:<25} {n:>3} / {TARGET_PER_GENRE}")

    new_records   = []
    seen_this_run = set()

    for genre in GENRE_PRIORITY:
        already    = existing_counts.get(genre, 0)
        still_need = max(0, TARGET_PER_GENRE - already)

        if still_need == 0:
            print(f"\n── {genre}  ✓ already full ──")
            continue

        print(f"\n── {genre}  (need {still_need} more) ──")
        collected = []

        for query in GENRE_QUERIES[genre]:
            if len(collected) >= still_need:
                break

            print(f"  Query: '{query}'")
            volumes = fetch_google_books(query, max_results=40)

            for vol in volumes:
                if len(collected) >= still_need:
                    break

                info  = vol.get("volumeInfo", {})
                title = info.get("title", "").strip()
                if not title:
                    continue

                # Filter reference/academic books
                if is_reference_book(info):
                    continue

                title_key = title.lower()
                if title_key in existing_titles or title_key in seen_this_run:
                    continue

                image_links = info.get("imageLinks", {})
                cover_url   = (
                    image_links.get("large") or
                    image_links.get("medium") or
                    image_links.get("thumbnail")
                )
                if not cover_url:
                    continue

                isbn = extract_isbn(vol) or vol.get("id", "")
                dest = COVERS_DIR / f"{genre}_{isbn}.jpg"

                if dest.exists():
                    print(f"  skip {dest.name}")
                else:
                    ok = fetch_cover(cover_url, dest)
                    if not ok:
                        continue

                authors = info.get("authors", ["Unknown"])
                record  = {
                    "isbn":        isbn,
                    "title":       title,
                    "authors":     authors,
                    "genre":       genre,
                    "filename":    str(dest),
                    "description": info.get("description", "")[:300],
                    "publisher":   info.get("publisher", ""),
                    "source":      "google_books",
                }
                collected.append(record)
                seen_this_run.add(title_key)
                existing_titles.add(title_key)
                print(f"  ✓ [{len(collected):02d}/{still_need}] {title[:55]}")

        new_records.extend(collected)
        shortfall = still_need - len(collected)
        flag      = f"  ⚠ {shortfall} short" if shortfall > 0 else ""
        print(f"  → added {len(collected)} for {genre}{flag}")

    # Merge and save
    all_records = existing_records + new_records
    with open(NYT_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done.")
    print(f"  New this run  : {len(new_records)}")
    print(f"  Total records : {len(all_records)}")

    final_counts = Counter(r["genre"] for r in all_records)
    print("\n  Final per genre:")
    for genre in GENRE_PRIORITY:
        n         = final_counts.get(genre, 0)
        shortfall = TARGET_PER_GENRE - n
        flag      = f"  ⚠ {shortfall} short" if shortfall > 0 else "  ✓"
        print(f"    {genre:<25} {n:>3}{flag}")


if __name__ == "__main__":
    main()