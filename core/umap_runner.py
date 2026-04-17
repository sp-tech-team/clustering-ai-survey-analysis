"""UMAP projections: high-dimensional (for clustering) and 3-D (for visualisation)."""

import numpy as np
from typing import Callable, Optional, Tuple

import umap

from config import (
    UMAP_N_COMPONENTS_HIGH, UMAP_N_COMPONENTS_VIS,
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_N_EPOCHS,
)


def run_umap(
    embeddings: np.ndarray,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (umap_high, umap_3d).

    umap_high : (n, UMAP_N_COMPONENTS_HIGH)  — used for HDBSCAN
    umap_3d   : (n, 3)                       — used for 3-D scatter
    """
    if progress_cb:
        progress_cb(0, 2, f"UMAP {embeddings.shape[1]}d → {UMAP_N_COMPONENTS_HIGH}d …")

    reducer_high = umap.UMAP(
        n_components=UMAP_N_COMPONENTS_HIGH,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        n_epochs=UMAP_N_EPOCHS,
        random_state=42,
        low_memory=True,
    )
    umap_high = reducer_high.fit_transform(embeddings)

    if progress_cb:
        progress_cb(1, 2, f"UMAP {embeddings.shape[1]}d → 3d …")

    reducer_3d = umap.UMAP(
        n_components=UMAP_N_COMPONENTS_VIS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=0.1,
        metric="cosine",
        n_epochs=UMAP_N_EPOCHS,
        random_state=42,
        low_memory=True,
    )
    umap_3d = reducer_3d.fit_transform(embeddings)

    if progress_cb:
        progress_cb(2, 2, "UMAP complete")

    return umap_high.astype(np.float32), umap_3d.astype(np.float32)
