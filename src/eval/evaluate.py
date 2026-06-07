"""
evaluate.py
-----------
Evaluates the retrieval system against the golden set.
Computes Precision@3 (P@3) for each track and prints a summary table.

Usage:
    python src/eval/evaluate.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.clip_encoder import encode_image, encode_text
from src.features.ocr          import extract_text
from src.features.colors       import extract_palette
from src.features.text_encoder import encode as encode_sentence
from src.search.retrieval      import load_index, search

# ── Paths ──────────────────────────────────────────────────────────────────────

GOLDEN_PATH = Path("data/golden_set.json")
TOP_K       = 3

TRACKS = ["visual", "cross_modal", "semantic", "color"]


# ── Metric ─────────────────────────────────────────────────────────────────────

def precision_at_k(results: list[dict], expected_ids: list, k: int) -> float:
    """
    P@k = (number of expected cover_ids in top-k results) / k
    """
    retrieved = [r["cover_id"] for r in results[:k]]
    hits = sum(1 for cid in retrieved if cid in expected_ids)
    return hits / k


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not GOLDEN_PATH.exists():
        print(f"Error: golden set not found at {GOLDEN_PATH}. Run golden.py first.")
        sys.exit(1)

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    load_index()

    print(f"\n{'═'*70}")
    print(f"  Golden Set Evaluation  —  P@{TOP_K} per track")
    print(f"{'═'*70}\n")

    # Accumulate scores per track across all queries
    track_scores = {t: [] for t in TRACKS}

    for entry in golden:
        img_path     = entry["query_filename"]
        expected_ids = set(entry["expected_top_cover_ids"])
        cover_id     = entry["query_cover_id"]

        print(f"── Query: {entry['query_title'][:55]}  [{entry['query_genre']}]")
        print(f"   Track tested: {entry['primary_track']}")
        print(f"   Note: {entry['note']}")

        if not Path(img_path).exists():
            print(f"   [warn] Image not found: {img_path}. Skipping.\n")
            continue

        # Extract features
        ocr_text     = extract_text(img_path)
        text         = ocr_text if ocr_text.strip() else entry["query_title"]
        clip_img_vec = encode_image(img_path)
        clip_txt_vec = encode_text(text)
        sentence_vec = encode_sentence(text)
        color_vec    = extract_palette(img_path)

        results = search(
            clip_image_vec    = clip_img_vec,
            clip_text_vec     = clip_txt_vec,
            sentence_vec      = sentence_vec,
            color_vec         = color_vec,
            k                 = TOP_K,
            exclude_cover_id  = cover_id,
        )

        for track in TRACKS:
            hits   = results.get(track, [])
            p_at_k = precision_at_k(hits, expected_ids, TOP_K)
            track_scores[track].append(p_at_k)

            hit_titles = [r["title"][:30] for r in hits if r["cover_id"] in expected_ids]
            status     = "✓" if p_at_k > 0 else "✗"
            print(f"   {status} {track:<15} P@{TOP_K}={p_at_k:.2f}   hits: {hit_titles}")

        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"{'─'*40}")
    print(f"  Summary  —  mean P@{TOP_K} across {len(golden)} queries")
    print(f"{'─'*40}")
    for track in TRACKS:
        scores = track_scores[track]
        if scores:
            mean = sum(scores) / len(scores)
            bar  = "█" * int(mean * 20)
            print(f"  {track:<15} {mean:.2f}  {bar}")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    main()