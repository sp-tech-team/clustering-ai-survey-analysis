from __future__ import annotations

from typing import Dict, List

import numpy as np

from config import SECONDARY_CENTROID_PERCENTILE, EXPORT_CENTROID_THRESHOLD_MARGIN, MAX_SECONDARY_CLUSTERS


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def compute_export_centroid_assignments(
    embeddings: np.ndarray | None,
    labels: np.ndarray,
    state,
    max_secondary_clusters: int = MAX_SECONDARY_CLUSTERS,
    percentile: int = SECONDARY_CENTROID_PERCENTILE,
    threshold_margin: float = EXPORT_CENTROID_THRESHOLD_MARGIN,
) -> tuple[np.ndarray, Dict[int, List[int]], dict]:
    export_labels = labels.copy().astype(np.int32)
    secondary_map: Dict[int, List[int]] = {}
    diagnostics = {
        "available": False,
        "cluster_count": 0,
        "outliers_before": int(np.sum(labels == -1)),
        "outliers_absorbed": 0,
        "outliers_after": int(np.sum(labels == -1)),
        "points_with_secondaries": 0,
        "total_secondary_links": 0,
        "max_secondary_links": 0,
        "threshold_min": None,
        "threshold_median": None,
        "threshold_max": None,
    }

    active_cids = [cid for cid in state.active_ids if cid in state.info and state.info[cid].is_active]
    diagnostics["cluster_count"] = int(len(active_cids))
    if embeddings is None or len(active_cids) == 0 or embeddings.shape[0] != len(labels):
        return export_labels, secondary_map, diagnostics

    normalized = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    cids = np.array(active_cids, dtype=np.int32)
    similarities = np.zeros((normalized.shape[0], len(cids)), dtype=np.float32)
    thresholds = np.ones(len(cids), dtype=np.float64)

    for col_idx, cid in enumerate(cids):
        primary_idx = np.where(labels == int(cid))[0]
        if len(primary_idx) == 0:
            continue

        centroid = _normalize_vector(normalized[primary_idx].mean(axis=0))
        if not np.any(centroid):
            continue

        sims = normalized @ centroid
        similarities[:, col_idx] = sims.astype(np.float32)
        raw_threshold = float(np.percentile(sims[primary_idx], percentile))
        thresholds[col_idx] = max(-1.0, min(1.0, raw_threshold - float(threshold_margin)))

    valid_thresholds = thresholds[np.isfinite(thresholds)]
    if len(valid_thresholds):
        diagnostics["available"] = True
        diagnostics["threshold_min"] = float(np.min(valid_thresholds))
        diagnostics["threshold_median"] = float(np.median(valid_thresholds))
        diagnostics["threshold_max"] = float(np.max(valid_thresholds))

    for row_idx in range(len(labels)):
        current_label = int(labels[row_idx])
        if current_label != -1 and (current_label not in state.info or not state.info[current_label].is_active):
            continue

        qualifying = [
            (int(cids[col_idx]), float(similarities[row_idx, col_idx]))
            for col_idx in range(len(cids))
            if float(similarities[row_idx, col_idx]) >= float(thresholds[col_idx])
        ]
        qualifying.sort(key=lambda item: item[1], reverse=True)

        export_primary = current_label
        if current_label == -1 and qualifying:
            export_primary = int(qualifying[0][0])
            export_labels[row_idx] = export_primary
            diagnostics["outliers_absorbed"] += 1

        secondaries = [cid for cid, _score in qualifying if cid != export_primary][:max_secondary_clusters]
        if secondaries:
            secondary_map[row_idx] = secondaries

    diagnostics["outliers_after"] = int(np.sum(export_labels == -1))
    diagnostics["points_with_secondaries"] = int(len(secondary_map))
    diagnostics["total_secondary_links"] = int(sum(len(cids) for cids in secondary_map.values()))
    diagnostics["max_secondary_links"] = int(max((len(cids) for cids in secondary_map.values()), default=0))
    return export_labels, secondary_map, diagnostics