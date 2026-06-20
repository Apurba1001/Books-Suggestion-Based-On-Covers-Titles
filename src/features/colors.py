"""
colors.py
---------
Extracts a dominant color palette from a cover image using k-means.

Improvements over baseline:
  1. LAB color space (perceptually uniform distances)
  2. Cluster weights (proportion of pixels per color)
  3. Border cropping (removes scan artifacts)
  4. Sorted by weight (most dominant color first)

Output: flat vector of shape (N_COLORS * 4,) — L, A, B, weight per cluster.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2lab

N_COLORS    = 5
RESIZE_TO   = (100, 100)
CROP_RATIO  = 0.10       # crop 10% from each edge
RANDOM_SEED = 42


def extract_palette(image_path: str) -> np.ndarray:
    """
    Returns a float32 vector of shape (N_COLORS * 4,).
    Each cluster: L, A, B (normalised), weight (0-1).
    """
    image = Image.open(image_path).convert("RGB").resize(RESIZE_TO)
    pixels = np.array(image)  # (100, 100, 3)

    # ── Crop borders ─────────────────────────────────────────────────────
    h, w = pixels.shape[:2]
    t = int(h * CROP_RATIO)
    b = h - t
    l = int(w * CROP_RATIO)
    r = w - l
    pixels = pixels[t:b, l:r]

    # ── Convert to LAB ───────────────────────────────────────────────────
    # rgb2lab expects float [0,1] input
    pixels_float = pixels.astype(np.float64) / 255.0
    lab_pixels   = rgb2lab(pixels_float)  # shape: (h, w, 3)

    # Normalise LAB to [0, 1] range for consistent cosine similarity
    # L: [0, 100], A: [-128, 127], B: [-128, 127]
    lab_flat = lab_pixels.reshape(-1, 3)
    lab_flat[:, 0] = lab_flat[:, 0] / 100.0            # L → [0, 1]
    lab_flat[:, 1] = (lab_flat[:, 1] + 128.0) / 255.0  # A → [0, 1]
    lab_flat[:, 2] = (lab_flat[:, 2] + 128.0) / 255.0  # B → [0, 1]

    # ── K-means clustering ───────────────────────────────────────────────
    kmeans = KMeans(n_clusters=N_COLORS, random_state=RANDOM_SEED, n_init="auto")
    labels = kmeans.fit_predict(lab_flat)
    centers = kmeans.cluster_centers_  # (N_COLORS, 3)

    # ── Compute cluster weights (proportion of pixels) ───────────────────
    total = len(labels)
    weights = np.array([
        np.sum(labels == i) / total for i in range(N_COLORS)
    ])  # shape: (N_COLORS,)

    # ── Sort by weight descending (most dominant first) ──────────────────
    order   = np.argsort(weights)[::-1]
    centers = centers[order]
    weights = weights[order]

    # ── Build output vector: [L, A, B, weight] per cluster ───────────────
    result = np.hstack([centers, weights.reshape(-1, 1)])  # (N_COLORS, 4)
    return result.flatten().astype(np.float32)  # shape: (N_COLORS * 4,)


def get_palette_dim() -> int:
    """Return the dimensionality of the palette vector."""
    return N_COLORS * 4