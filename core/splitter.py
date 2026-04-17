"""
Split a single cluster by re-running HDBSCAN on its points only.
Returns new assignment mapping + skeleton cluster info (no LLM titles yet).
"""

import numpy as np
from typing import Dict, List, Tuple

import hdbscan

from config import HDBSCAN_MIN_SAMPLES


def split_cluster(
    cluster_id: int,
    point_indices: List[int],
    umap_high: np.ndarray,
    next_cluster_id: int,
) -> Tuple[Dict[int, int], List[int]]:
    """
    Re-cluster the points belonging to `cluster_id`.

    Returns:
        assignments : {point_idx: new_cluster_id}  (outliers within the split
                      are assigned to the largest new sub-cluster)
        new_ids     : list of new cluster IDs created
    """
    if len(point_indices) < 4:
        raise ValueError("Cluster too small to split (need ≥ 4 points).")

    sub_emb = umap_high[point_indices]
    min_cs  = max(3, len(point_indices) // 5)

    sub_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cs,
        min_samples=min(HDBSCAN_MIN_SAMPLES, min_cs),
        metric="euclidean",
        cluster_selection_method="eom",
        gen_min_span_tree=True,
    )
    sub_labels = sub_clusterer.fit_predict(sub_emb)

    unique_sub = sorted(set(sub_labels))
    n_named    = [c for c in unique_sub if c != -1]

    if len(n_named) < 2:
        raise ValueError("HDBSCAN could not find ≥ 2 sub-clusters. Try a larger cluster or different parameters.")

    # Map sub-cluster indices 0, 1, 2… → globally unique IDs
    id_map = {sc: next_cluster_id + i for i, sc in enumerate(n_named)}
    # Outliers within the split → assigned to the largest sub-cluster
    largest = max(n_named, key=lambda c: int((sub_labels == c).sum()))
    id_map[-1] = id_map[largest]

    assignments = {
        point_indices[i]: id_map[int(sub_labels[i])]
        for i in range(len(point_indices))
    }
    new_ids = list(id_map[c] for c in n_named)
    return assignments, new_ids
