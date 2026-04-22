"""
HDBSCAN clustering + exemplar-based representative extraction.
Returns base cluster state dicts ready to be saved to DB.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

import hdbscan
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from config import (
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
    N_REPRESENTATIVES, N_OUTLIER_SAMPLE,
    SECONDARY_MEMBERSHIP_PERCENTILE, SECONDARY_MEMBERSHIP_FLOOR,
    SECONDARY_CENTROID_PERCENTILE,
)


# ── HDBSCAN ───────────────────────────────────────────────────────────────────

def run_hdbscan(
    umap_high: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
) -> Tuple[hdbscan.HDBSCAN, np.ndarray]:
    """
    Returns (fitted_clusterer, cluster_labels).
    prediction_data=True is required for soft-membership vectors.
    gen_min_span_tree=True is required for exemplars_.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        gen_min_span_tree=True,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(umap_high)
    return clusterer, labels


# ── Representative extraction ─────────────────────────────────────────────────

def _exemplar_indices(clusterer: hdbscan.HDBSCAN, cluster_id: int, umap_high: np.ndarray) -> List[int]:
    try:
        vecs    = clusterer.exemplars_[cluster_id]
        indices = []
        for v in vecs:
            diffs = np.abs(umap_high - v).sum(axis=1)
            indices.append(int(np.argmin(diffs)))
        return list(set(indices))
    except (IndexError, AttributeError):
        return []


def _centroid_nearest(indices: List[int], emb: np.ndarray, n: int) -> List[int]:
    sub      = emb[indices]
    centroid = sub.mean(axis=0, keepdims=True)
    dists    = cosine_distances(centroid, sub)[0]
    return [indices[i] for i in np.argsort(dists)[:n]]


def extract_representatives(
    clusterer: hdbscan.HDBSCAN,
    labels: np.ndarray,
    umap_high: np.ndarray,
    embeddings: np.ndarray,
    unique_clusters: List[int],
    n_reps: int = N_REPRESENTATIVES,
    n_outlier: int = N_OUTLIER_SAMPLE,
) -> Dict[int, List[int]]:
    """Returns {cluster_id: [point_indices]}."""
    reps = {}
    for cid in unique_clusters:
        all_idx = np.where(labels == cid)[0].tolist()
        if cid == -1:
            rep_idx = _centroid_nearest(all_idx, embeddings, n_outlier)
        else:
            ex_idx = _exemplar_indices(clusterer, cid, umap_high)
            if len(ex_idx) >= n_reps:
                rep_idx = ex_idx[:n_reps]
            else:
                remaining = [i for i in all_idx if i not in set(ex_idx)]
                pad       = _centroid_nearest(remaining, umap_high, n_reps - len(ex_idx)) if remaining else []
                rep_idx   = ex_idx + pad
        reps[cid] = rep_idx
    return reps


# ── Cluster summary dict builder (pre-LLM) ────────────────────────────────────

def build_base_cluster_list(labels: np.ndarray, unique_clusters: List[int]) -> List[dict]:
    """Build skeleton summary dicts (no LLM titles yet)."""
    total = len(labels)
    return [
        {
            "cluster_id": int(cid),
            "title":       "Other Themes" if cid == -1 else f"Cluster {cid}",
            "description": "",
            "sentiment":   "unknown",
            "n_points":    int((labels == cid).sum()),
            "pct":         round(int((labels == cid).sum()) / total * 100, 1),
        }
        for cid in unique_clusters
    ]


# ── Soft membership ────────────────────────────────────────────────────────────

def compute_soft_membership(
    clusterer: hdbscan.HDBSCAN,
    labels: np.ndarray,
    named_clusters: List[int],
    percentile: int = SECONDARY_MEMBERSHIP_PERCENTILE,
    floor: float    = SECONDARY_MEMBERSHIP_FLOOR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      membership   — float32 (n_points, n_named_clusters)
      cids         — int32   (n_named_clusters,)  column → original HDBSCAN cluster id
      thresholds   — float64 (n_named_clusters,)  per-cluster assignment threshold
    """
    membership = hdbscan.all_points_membership_vectors(clusterer)  # (n_pts, n_named)
    cids       = np.array(named_clusters, dtype=np.int32)
    thresholds = np.zeros(len(named_clusters), dtype=np.float64)
    for col_idx, cid in enumerate(named_clusters):
        primary_mask = labels == cid
        if primary_mask.sum() > 0:
            raw_pct = float(np.percentile(membership[primary_mask, col_idx], percentile))
        else:
            raw_pct = 1.0
        thresholds[col_idx] = max(raw_pct, floor)
    return membership.astype(np.float32), cids, thresholds


def aggregate_membership_by_cluster(
    membership: np.ndarray,
    raw_cids: np.ndarray,
    cluster_map: Dict[int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse raw-cluster membership columns into canonical cluster columns via max."""
    cluster_map = cluster_map or {}
    grouped: Dict[int, List[int]] = {}
    for col_idx, raw_cid in enumerate(raw_cids):
        canonical_cid = int(cluster_map.get(int(raw_cid), int(raw_cid)))
        grouped.setdefault(canonical_cid, []).append(col_idx)

    canonical_cids = np.array(sorted(grouped), dtype=np.int32)
    canonical_scores = np.zeros((membership.shape[0], len(canonical_cids)), dtype=np.float32)
    for out_col, canonical_cid in enumerate(canonical_cids):
        source_cols = grouped[int(canonical_cid)]
        canonical_scores[:, out_col] = membership[:, source_cols].max(axis=1)
    return canonical_scores, canonical_cids


def compute_cluster_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    cids: np.ndarray,
    percentile: int = SECONDARY_MEMBERSHIP_PERCENTILE,
    floor: float = SECONDARY_MEMBERSHIP_FLOOR,
) -> np.ndarray:
    """Compute one threshold per cluster from all current primary members."""
    thresholds = np.zeros(len(cids), dtype=np.float64)
    for col_idx, cid in enumerate(cids):
        primary_mask = labels == int(cid)
        if primary_mask.sum() > 0:
            raw_pct = float(np.percentile(scores[primary_mask, col_idx], percentile))
        else:
            raw_pct = 1.0
        thresholds[col_idx] = max(raw_pct, floor)
    return thresholds


def assign_clusters_from_scores(
    scores: np.ndarray,
    cids: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, float]]]]:
    """Assign each point to its best qualifying cluster and retain all qualifiers."""
    labels = np.full(scores.shape[0], -1, dtype=np.int32)
    qualifying_map: Dict[int, List[Tuple[int, float]]] = {}

    for row_idx in range(scores.shape[0]):
        qualifying = [
            (int(cids[col_idx]), float(scores[row_idx, col_idx]))
            for col_idx in range(len(cids))
            if float(scores[row_idx, col_idx]) >= float(thresholds[col_idx])
        ]
        qualifying.sort(key=lambda item: item[1], reverse=True)
        if qualifying:
            labels[row_idx] = int(qualifying[0][0])
            qualifying_map[row_idx] = qualifying

    return labels, qualifying_map


# ── Centroid cosine secondary assignment ──────────────────────────────────────

def compute_centroid_thresholds(
    embeddings: np.ndarray,
    rep_indices: Dict[int, List[int]],
    labels: np.ndarray,
    named_merged_clusters: List[int],
    unique_raw_clusters: List[int],
    merge_map: Dict[int, int],
    percentile: int = SECONDARY_CENTROID_PERCENTILE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-cluster centroid vectors and cosine-similarity thresholds for
    secondary cluster assignment.

    Centroid for each canonical cluster is the mean of its representative points'
    embeddings (pooled across all raw sub-clusters that merged into it).
    Threshold = Nth percentile of cosine similarity to centroid among ALL primary
    members of that canonical cluster (robust, not just reps).

    Returns:
      centroids   — float32 (n_clusters, emb_dim)  one centroid per canonical cluster
      cids        — int32   (n_clusters,)           canonical cluster ID per row
      thresholds  — float64 (n_clusters,)           per-cluster cosine sim threshold
    """
    # Build merged_to_raw: canonical id → list of contributing raw cluster ids
    merged_to_raw: Dict[int, List[int]] = {j: [] for j in named_merged_clusters}
    for raw in unique_raw_clusters:
        canonical = merge_map.get(raw, raw)
        if canonical in merged_to_raw:
            merged_to_raw[canonical].append(raw)

    emb_dim    = embeddings.shape[1]
    n_clusters = len(named_merged_clusters)
    centroids  = np.zeros((n_clusters, emb_dim), dtype=np.float32)
    thresholds = np.zeros(n_clusters, dtype=np.float64)
    cids       = np.array(named_merged_clusters, dtype=np.int32)

    for col_idx, j in enumerate(named_merged_clusters):
        # Pool representative indices from all raw clusters feeding this canonical
        all_rep_idx: List[int] = []
        for raw in merged_to_raw.get(j, [j]):
            all_rep_idx.extend(rep_indices.get(raw, []))
        all_rep_idx = list(set(all_rep_idx))

        if not all_rep_idx:
            # Fallback: use all primary members directly
            all_rep_idx = np.where(labels == j)[0].tolist()

        if not all_rep_idx:
            thresholds[col_idx] = 1.0  # no data — effectively disabled
            continue

        centroid_j = embeddings[all_rep_idx].mean(axis=0)
        centroids[col_idx] = centroid_j.astype(np.float32)

        # Threshold from ALL primary members' similarities (robust percentile)
        primary_mask = labels == j
        if primary_mask.sum() > 0:
            sims = cosine_similarity(
                embeddings[primary_mask], centroid_j.reshape(1, -1)
            ).ravel()
            thresholds[col_idx] = float(np.percentile(sims, percentile))
        else:
            thresholds[col_idx] = 1.0  # effectively disabled

    return centroids, cids, thresholds
