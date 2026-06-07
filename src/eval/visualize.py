"""
visualize.py
------------
Project high-dimensional embedding vectors down to 2D and plot them.
Points are colored by genre so you can see if the embeddings cluster
by genre (which means they capture meaningful structure).

Usage:
    uv run python src/eval/visualize.py
    uv run python src/eval/visualize.py --track sentence --method tsne
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Paths ──────────────────────────────────────────────────────────────────────

EMBEDDINGS_DIR = Path("embeddings")
ROW_INDEX_PATH = EMBEDDINGS_DIR / "row_index.json"

TRACKS = {
    "clip_image":  "CLIP image vectors (visual similarity)",
    "clip_text":   "CLIP text vectors (from OCR)",
    "sentence":    "SentenceTransformer vectors (semantic)",
    "colors":      "Color palette vectors",
}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track",  choices=TRACKS.keys(), default="clip_image")
    ap.add_argument("--method", choices=["pca", "tsne"], default="pca")
    args = ap.parse_args()

    # Load
    with open(ROW_INDEX_PATH, encoding="utf-8") as f:
        row_index = json.load(f)
    vectors = np.load(EMBEDDINGS_DIR / f"{args.track}.npy")
    genres  = [book["genre"] for book in row_index]

    print(f"Loaded {vectors.shape[0]} vectors of shape {vectors.shape[1]}")
    print(f"Projecting to 2D using {args.method.upper()}...")

    # Project to 2D
    if args.method == "pca":
        proj = PCA(n_components=2).fit_transform(vectors)
    else:
        # t-SNE is slower but often shows nicer clusters
        proj = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(vectors)

    # Plot
    unique_genres = sorted(set(genres))
    colors        = plt.cm.tab10(np.linspace(0, 1, len(unique_genres)))

    plt.figure(figsize=(12, 8))
    for genre, color in zip(unique_genres, colors):
        mask = [g == genre for g in genres]
        pts  = proj[mask]
        plt.scatter(pts[:, 0], pts[:, 1], c=[color], label=genre, alpha=0.7, s=40)

    plt.title(f"{TRACKS[args.track]}  —  {args.method.upper()} projection")
    plt.legend(loc="best", fontsize=9)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()

    out = Path(f"embeddings/plot_{args.track}_{args.method}.png")
    plt.savefig(out, dpi=120)
    print(f"Saved plot to {out}")
    plt.show()


if __name__ == "__main__":
    main()