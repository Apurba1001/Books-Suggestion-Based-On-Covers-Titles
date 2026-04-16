"""
golden.py
---------
Builds data/golden_set.json from the existing index.

For each genre track we pick one visually distinctive book as the query,
and define the expected top matches (books in the same genre).
This gives us a human-curated evaluation baseline before the hackathon.

Usage:
    python src/data/golden.py

Output:
    data/golden_set.json
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────

INDEX_PATH      = Path("data/index.json")
GOLDEN_PATH     = Path("data/golden_set.json")

# One query per genre — chosen to test different recommendation tracks.
# Each entry says: which genre the query comes from, which track it primarily
# tests, and a short note explaining what makes it a good test case.
QUERY_SPEC = [
    {
        "genre":         "thriller",
        "track":         "genre + visual",
        "note":          "Dark, minimal typographic covers — tests visual style similarity within genre",
    },
    {
        "genre":         "science_fiction",
        "track":         "cross-modal (OCR → genre)",
        "note":          "Sci-fi covers with legible titles — tests whether OCR + text embedding retrieves correct genre",
    },
    {
        "genre":         "fantasy",
        "track":         "visual similarity",
        "note":          "Illustrated fantasy covers — tests colour palette and visual texture matching",
    },
    {
        "genre":         "biography",
        "track":         "semantic text",
        "note":          "Covers with strong author-name prominence — tests semantic NLP track on metadata",
    },
    {
        "genre":         "horror",
        "track":         "color palette",
        "note":          "Red/black dominant covers — tests colour-based retrieval",
    },
]

# How many expected matches to store per golden query
TOP_K = 5


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"index.json not found at {INDEX_PATH}. Run download.py first."
        )
    with open(INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


def group_by_genre(index: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for book in index:
        groups[book["genre"]].append(book)
    return dict(groups)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    index  = load_index()
    groups = group_by_genre(index)

    print(f"Loaded {len(index)} books across {len(groups)} genres.\n")
    for genre, books in groups.items():
        print(f"  {genre:<25} {len(books)} books")

    golden = []

    for spec in QUERY_SPEC:
        genre = spec["genre"]
        pool  = groups.get(genre, [])

        if len(pool) < TOP_K + 1:
            print(f"\n[warn] Not enough books in '{genre}' ({len(pool)}). Skipping.")
            continue

        # Seed for reproducibility — same run always picks the same query
        rng = random.Random(42)

        # Pick the query book (first one — deterministic, easy to inspect)
        query = pool[0]

        # Expected matches = rest of the genre pool (ground truth for P@k eval)
        # We store the top TOP_K by their position in the list (no ranking yet —
        # that comes once embeddings are built)
        expected = [b["cover_id"] for b in pool[1: TOP_K + 1]]

        entry = {
            "query_cover_id":        query["cover_id"],
            "query_title":           query["title"],
            "query_authors":         query["authors"],
            "query_genre":           genre,
            "query_filename":        query["filename"],
            "primary_track":         spec["track"],
            "note":                  spec["note"],
            "expected_top_cover_ids": expected,
            "expected_titles": [
                b["title"] for b in pool[1: TOP_K + 1]
            ],
        }
        golden.append(entry)

        print(f"\n── {genre} ──")
        print(f"  Query   : {query['title'][:60]} (id {query['cover_id']})")
        print(f"  Expected: {[b['title'][:30] for b in pool[1: TOP_K + 1]]}")

    # Save
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Golden set saved to {GOLDEN_PATH}  ({len(golden)} queries)")
    print(
        "\nNext step: inspect data/golden_set.json and open the query cover images "
        "in data/covers/ to verify they are visually representative. "
        "Swap out any query cover_id manually if you want a better example."
    )


if __name__ == "__main__":
    main()