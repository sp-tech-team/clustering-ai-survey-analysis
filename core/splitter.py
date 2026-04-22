"""Split a single cluster by re-running HDBSCAN locally on its points."""

import numpy as np
from typing import Dict, List, Tuple

import hdbscan

from config import HDBSCAN_MIN_SAMPLES, SECONDARY_MEMBERSHIP_PERCENTILE, SECONDARY_MEMBERSHIP_FLOOR
from core.clusterer import compute_cluster_thresholds, assign_clusters_from_scores


def split_cluster(
    cluster_id: int,
    point_indices: List[int],
    umap_high: np.ndarray,
    next_cluster_id: int,
) -> Tuple[Dict[int, int], List[int], Dict[int, List[Tuple[int, float]]], Dict[int, float]]:
    """
    Re-cluster the points belonging to `cluster_id`.

    Returns:
        assignments      : {point_idx: new_cluster_id or -1}
        new_ids          : list of new cluster IDs created
        qualifying_map   : {point_idx: [(new_cluster_id, score), ...]}
        threshold_map    : {new_cluster_id: threshold}
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
        prediction_data=True,
    )
    sub_labels = sub_clusterer.fit_predict(sub_emb)

    unique_sub = sorted(set(sub_labels))
    n_named    = [c for c in unique_sub if c != -1]

    if len(n_named) < 2:
        raise ValueError("HDBSCAN could not find ≥ 2 sub-clusters. Try a larger cluster or different parameters.")

    membership = hdbscan.all_points_membership_vectors(sub_clusterer).astype(np.float32)
    raw_cids = np.array(n_named, dtype=np.int32)
    thresholds = compute_cluster_thresholds(
        membership,
        sub_labels,
        raw_cids,
        percentile=SECONDARY_MEMBERSHIP_PERCENTILE,
        floor=SECONDARY_MEMBERSHIP_FLOOR,
    )
    local_labels, local_qualifying = assign_clusters_from_scores(membership, raw_cids, thresholds)

    # Map sub-cluster indices 0, 1, 2… → globally unique IDs
    id_map = {sc: next_cluster_id + i for i, sc in enumerate(n_named)}

    assignments = {
        point_indices[i]: id_map[int(local_labels[i])] if int(local_labels[i]) != -1 else -1
        for i in range(len(point_indices))
    }
    qualifying_map = {
        point_indices[i]: [(id_map[int(cid)], score) for cid, score in qualifying]
        for i, qualifying in local_qualifying.items()
    }
    threshold_map = {id_map[int(raw_cid)]: float(thresholds[col_idx]) for col_idx, raw_cid in enumerate(raw_cids)}
    new_ids = list(id_map[c] for c in n_named)
    return assignments, new_ids, qualifying_map, threshold_map
