"""
evaluate.py
-----------
Evaluates the retrieval system against a golden set.
Computes Precision@3 (P@3) for each track and prints a summary table.

Usage:
    # Original dataset
    uv run python src/eval/evaluate.py

    # NYT dataset
    uv run python src/eval/evaluate.py --golden data/nyt_golden_set.json --embeddings embeddings_nyt
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.clip_encoder import encode_image, encode_text, set_model as set_clip_model
from src.features.ocr          import extract_text
from src.features.colors       import extract_palette
from src.features.text_encoder import encode as encode_sentence
from src.search.retrieval      import load_index, search

TOP_K  = 3
TRACKS = ["visual", "cross_modal", "semantic", "color"]


def precision_at_k(results: list[dict], expected_ids: list, k: int) -> float:
    retrieved = [r["id"] for r in results[:k]]
    hits = sum(1 for rid in retrieved if rid in expected_ids)
    return hits / k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden",     default="data/golden_set.json")
    parser.add_argument("--embeddings", default="embeddings")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                        help="CLIP model ID (must match the embeddings)")
    args = parser.parse_args()

    set_clip_model(args.clip_model) 

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"Error: golden set not found at {golden_path}")
        sys.exit(1)

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    load_index(args.embeddings)

    print(f"\n{'═'*70}")
    print(f"  Golden Set Evaluation  —  P@{TOP_K} per track")
    print(f"  Golden : {golden_path}")
    print(f"  Index  : {args.embeddings}")
    print(f"{'═'*70}\n")

    track_scores = {t: [] for t in TRACKS}

    for entry in golden:
        img_path = entry.get("query_filename", "")
        # Support both cover_id and isbn based golden sets
        query_id     = str(entry.get("query_isbn") or entry.get("query_cover_id", ""))
        expected_ids = set(
            str(x) for x in (
                entry.get("expected_top_isbns") or
                entry.get("expected_top_cover_ids") or []
            )
        )

        print(f"── Query: {entry.get('query_title','?')[:55]}  [{entry.get('query_genre','?')}]")
        print(f"   Track tested : {entry.get('primary_track','?')}")
        print(f"   Note         : {entry.get('note','')}")

        if not Path(img_path).exists():
            print(f"   [warn] Image not found: {img_path}. Skipping.\n")
            continue

        ocr_text     = extract_text(img_path)
        text         = ocr_text if ocr_text.strip() else entry.get("query_title", "")
        clip_img_vec = encode_image(img_path)
        clip_txt_vec = encode_text(text)
        sentence_vec = encode_sentence(text)
        color_vec    = extract_palette(img_path)

        results = search(
            clip_image_vec = clip_img_vec,
            clip_text_vec  = clip_txt_vec,
            sentence_vec   = sentence_vec,
            color_vec      = color_vec,
            k              = TOP_K,
            exclude_id     = query_id,
        )

        for track in TRACKS:
            hits   = results.get(track, [])
            p_at_k = precision_at_k(hits, expected_ids, TOP_K)
            track_scores[track].append(p_at_k)
            hit_titles = [r["title"][:30] for r in hits if r["id"] in expected_ids]
            status     = "✓" if p_at_k > 0 else "✗"
            print(f"   {status} {track:<15} P@{TOP_K}={p_at_k:.2f}   hits: {hit_titles}")

        print()

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