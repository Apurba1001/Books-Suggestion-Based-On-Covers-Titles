"""
colors.py
---------
Extracts a dominant color palette from a cover image using k-means.
Produces a flat 15-dim vector (5 colors × RGB) for color-based similarity.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

N_COLORS    = 5       # number of dominant colors to extract
RESIZE_TO   = (100, 100)  # downsample before clustering — much faster, same result
RANDOM_SEED = 42


def extract_palette(image_path: str) -> np.ndarray:
    """
    Returns a float32 vector of shape (15,) — five RGB colors flattened.
    Values are in [0, 1].
    """
    image = Image.open(image_path).convert("RGB").resize(RESIZE_TO)
    pixels = np.array(image).reshape(-1, 3) / 255.0  # shape: (10000, 3)

    kmeans = KMeans(n_clusters=N_COLORS, random_state=RANDOM_SEED, n_init="auto")
    kmeans.fit(pixels)

    # Sort clusters by brightness so the vector is order-stable across images
    centers = kmeans.cluster_centers_
    brightness = centers.mean(axis=1)
    centers = centers[np.argsort(brightness)]

    return centers.flatten().astype(np.float32)  # shape: (15,)