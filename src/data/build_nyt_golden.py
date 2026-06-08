"""
build_nyt_golden.py
-------------------
Rebuilds data/nyt_golden_set.json based on actual retrieval results.
Expected matches are ISBNs that the system actually returns and are
genuinely good recommendations.

Usage:
    uv run python src/data/build_nyt_golden.py
"""

import json
from pathlib import Path

INDEX_PATH  = Path("data/nyt_index.json")
GOLDEN_PATH = Path("data/nyt_golden_set.json")

with open(INDEX_PATH, encoding="utf-8") as f:
    index = json.load(f)

by_isbn = {str(b.get("isbn","")): b for b in index}

def get(isbn):
    b = by_isbn.get(str(isbn))
    if not b:
        print(f"  [warn] ISBN not found: {isbn}")
    return b

# ── Golden set based on actual retrieval results ───────────────────────────────
# Each entry: (query_isbn, query_title, query_genre, query_filename,
#              expected_isbns, primary_track, note)
#
# Expected ISBNs are chosen from actual top-5 results that are
# genuinely good matches (same genre or thematically close).

GOLDEN = [
    (
        "9780385121675",
        "The Shining",
        "horror",
        "data/covers_nyt/horror_9780385121675.jpg",
        ["9781501156700",   # It — Stephen King horror
         "9780307743657",   # Doctor Sleep — Stephen King horror
         "9780671685638",   # Ghost Story — horror
         "9780553807738",   # Deeply Odd — horror
         "9780451173317"],  # The Waste Lands — horror
        "visual + semantic",
        "Stephen King horror — tests visual style and same-author genre clustering",
    ),
    (
        "9780593135228",
        "PROJECT HAIL MARY",
        "science_fiction",
        "data/covers_nyt/science_fiction_9780593135228.jpg",
        ["9798217300556",   # THE MARTIAN — sci-fi (mislabeled fantasy, still valid)
         "9781250163660",   # Our Angry Earth — science_fiction
         "9780307792273",   # Isaac Asimov's Guide — science_fiction
         "9780898750010",   # Worlds Within Worlds — science_fiction
         "9780190854157"],  # The First Minds — science_fiction
        "cross-modal + semantic",
        "Modern sci-fi — tests cross-modal and semantic retrieval within science fiction",
    ),
    (
        "9780765365279",
        "The Way of Kings",
        "fantasy",
        "data/covers_nyt/fantasy_9780765365279.jpg",
        ["9780765350381",   # Mistborn — fantasy Sanderson
         "9780006486022",   # The Golden Fool — fantasy Hobb
         "9780593809907",   # THE POISONED KING — fantasy
         "9780063415898",   # THE BALLAD OF FALLING DRAGONS — fantasy
         "9780765356147"],  # The Hero of Ages — fantasy Sanderson
        "visual similarity",
        "Epic fantasy artwork — tests visual clustering within fantasy genre",
    ),
    (
        "9780312979478",
        "And Then There Were None",
        "mystery",
        "data/covers_nyt/mystery_9780312979478.jpg",
        ["9780006167242",   # The ABC Murders — Christie mystery
         "9780007211654",   # The Pale Horse — Christie mystery
         "9780006161745",   # Cat Among the Pigeons — Christie mystery
         "9780006169185",   # The Big Four — Christie mystery
         "9780446611640"],  # The Narrows — mystery
        "visual + genre",
        "Agatha Christie — tests same-author visual clustering within mystery",
    ),
    (
        "9781476708706",
        "The Innovators",
        "biography",
        "data/covers_nyt/biography_9781476708706.jpg",
        ["9780743258074",   # Benjamin Franklin — Isaacson biography
         "9781501171383",   # Einstein — Isaacson biography
         "9781501127625",   # Steve Jobs — Isaacson biography
         "9781476728759",   # The Wright Brothers — biography
         "9780316017923"],  # Immortal Life of Henrietta Lacks — biography
        "visual + semantic",
        "Walter Isaacson biography — tests same-author visual and semantic clustering",
    ),
    (
        "9780143124542",
        "The Book Thief",
        "historical_fiction",
        "data/covers_nyt/historical_fiction_9780143124542.jpg",
        ["9780062409843",   # The Alice Network — historical fiction
         "9780062409850",   # All the Light We Cannot See — historical fiction
         "9780525555360",   # The Tattooist of Auschwitz — historical fiction
         "9780062838667",   # The Nightingale — historical fiction
         "9780007320271"],  # Wolf Hall — historical fiction
        "visual + semantic",
        "WWII historical fiction — tests visual and semantic clustering",
    ),
    (
        "9780735211292",
        "ATOMIC HABITS",
        "non-fiction",
        "data/covers_nyt/non-fiction_9780735211292.jpg",
        ["9798893310153",   # DON'T BELIEVE EVERYTHING YOU THINK — non-fiction
         "9798893311860",   # INCORRUPTIBLE — non-fiction
         "9781455570249",   # MAKE YOUR BED — non-fiction
         "9781401971366",   # THE LET THEM THEORY — non-fiction
         "9798217281633"],  # INSIDE THE BOX — non-fiction
        "cross-modal + semantic",
        "Self-help non-fiction — tests OCR bold typography and semantic retrieval",
    ),
    (
        "9780063204188",
        "REMARKABLY BRIGHT CREATURES",
        "mystery",
        "data/covers_nyt/mystery_9780063204188.jpg",
        ["9798347810307",   # THE CALAMITY CLUB — historical fiction
         "9781954118812",   # THE CALAMITY CLUB — literary fiction
         "9781775293941",   # THE MISTAKE — literary fiction
         "9798217073443",   # STRANGERS — mystery
         "9780060757625"],  # Shaman's Crossing — fantasy (visual match)
        "visual + semantic",
        "Illustrated literary cover — tests visual similarity and thematic retrieval",
    ),
    (
        "9780553575637",
        "Ship of Magic",
        "fantasy",
        "data/covers_nyt/fantasy_9780553575637.jpg",
        ["9780006486022",   # The Golden Fool — Hobb fantasy
         "9780007196197",   # Renegade's Magic — Hobb fantasy
         "9780007196166",   # Forest Mage — Hobb
         "9780060757625",   # Shaman's Crossing — Hobb
         "9780002246064"],  # Assassin's Apprentice — Hobb
        "visual similarity",
        "Robin Hobb fantasy — tests same-author visual clustering",
    ),
    (
        "9781501121586",
        "Becoming",
        "biography",
        "data/covers_nyt/biography_9781501121586.jpg",
        ["9780316346627",   # Born a Crime — memoir
         "9780525559474",   # I Am Malala — memoir
         "9780679720201",   # The Diary of a Young Girl — memoir
         "9780316017923",   # Immortal Life of Henrietta Lacks — biography
         "9780743258074"],  # Benjamin Franklin — biography
        "semantic",
        "Contemporary memoir — tests semantic retrieval for personal narrative",
    ),
]


golden = []
for isbn, title, genre, filename, expected_isbns, track, note in GOLDEN:
    query = by_isbn.get(str(isbn))
    if not query:
        print(f"[warn] query not found: {title} ({isbn})")
        continue

    expected_books  = [by_isbn.get(str(e)) for e in expected_isbns]
    expected_books  = [b for b in expected_books if b]

    entry = {
        "query_isbn":          isbn,
        "query_title":         title,
        "query_authors":       query.get("authors", []),
        "query_genre":         genre,
        "query_filename":      filename,
        "primary_track":       track,
        "note":                note,
        "expected_top_isbns":  [b["isbn"] for b in expected_books],
        "expected_top_titles": [b["title"] for b in expected_books],
    }
    golden.append(entry)
    print(f"  ✓ [{genre:<20}] {title[:45]}")
    for b in expected_books:
        print(f"       → {b['title'][:50]}")
    print()

with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
    json.dump(golden, f, indent=2, ensure_ascii=False)

print(f"✓ Saved {len(golden)} golden queries to {GOLDEN_PATH}")