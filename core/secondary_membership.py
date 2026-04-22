from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from config import MAX_SECONDARY_CLUSTERS


def replay_secondary_memberships(
    mem_data: dict | None,
    edits: list,
    state,
    max_secondary_clusters: int = MAX_SECONDARY_CLUSTERS,
) -> tuple[Dict[int, List[int]], dict]:
    secondary_map: Dict[int, List[int]] = {}
    diagnostics = {
        "membership_available": mem_data is not None,
        "cached_cluster_count": 0,
        "active_points": int(len(state.labels)),
        "assigned_points": int(np.sum(state.labels != -1)),
        "outliers_remaining": int(np.sum(state.labels == -1)),
        "points_with_secondaries": 0,
        "total_secondary_links": 0,
        "max_secondary_links": 0,
    }
    if mem_data is None:
        return secondary_map, diagnostics

    qualifying_scores: dict[int, dict[int, float]] = {}
    membership = mem_data["membership"]
    cids = mem_data["cids"]
    thresholds = mem_data["thresholds"]
    diagnostics["cached_cluster_count"] = int(len(cids))

    for row_idx in range(membership.shape[0]):
        qualifying_scores[row_idx] = {
            int(cids[col_idx]): float(membership[row_idx, col_idx])
            for col_idx in range(len(cids))
            if float(membership[row_idx, col_idx]) >= float(thresholds[col_idx])
        }

    for edit in edits:
        payload = edit.payload or {}
        if edit.edit_type == "join":
            from_ids = [int(cid) for cid in payload.get("from_ids", [])]
            to_id = int(payload.get("to_id", from_ids[0])) if from_ids else None
            if to_id is None:
                continue
            for point_scores in qualifying_scores.values():
                merged_scores = [point_scores.get(cid) for cid in from_ids if cid in point_scores]
                merged_scores = [score for score in merged_scores if score is not None]
                if merged_scores:
                    point_scores[to_id] = max([point_scores.get(to_id, float("-inf"))] + merged_scores)
                for cid in from_ids:
                    if cid != to_id:
                        point_scores.pop(cid, None)
        elif edit.edit_type == "split":
            point_rows = {int(pair[0]) for pair in payload.get("new_assignments", [])}
            local_qualifying = payload.get("local_qualifying", {})
            for row_idx in point_rows:
                qualifying_scores[row_idx] = {
                    int(cluster_id): float(score)
                    for cluster_id, score in local_qualifying.get(str(row_idx), [])
                }

    for row_idx in range(len(state.labels)):
        primary_final = int(state.labels[row_idx])
        point_scores = qualifying_scores.get(row_idx, {})
        active_qualifying = [
            (score, cid)
            for cid, score in point_scores.items()
            if cid in state.info and state.info[cid].is_active
        ]
        active_qualifying.sort(reverse=True)
        secondaries = [
            cid for score, cid in active_qualifying
            if cid != primary_final
        ][:max_secondary_clusters]
        if secondaries:
            secondary_map[row_idx] = secondaries

    diagnostics["points_with_secondaries"] = int(len(secondary_map))
    diagnostics["total_secondary_links"] = int(sum(len(cids) for cids in secondary_map.values()))
    diagnostics["max_secondary_links"] = int(max((len(cids) for cids in secondary_map.values()), default=0))
    return secondary_map, diagnostics