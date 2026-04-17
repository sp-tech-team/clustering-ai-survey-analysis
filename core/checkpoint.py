"""
Session checkpoint helpers.

load_session_state() is the single entry-point for the analysis page.
It returns everything needed to render the current state.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

import core.cache as cache
from core.state import ClusterState, reconstruct
from config import EMBEDDING_MODEL
from db.queries import (
    get_session, get_points, get_clusters,
    get_cluster_assignments, get_all_edits,
)


def arrays_ready(csv_hash: str, embedding_model: str = EMBEDDING_MODEL) -> bool:
    return cache.exists(csv_hash, embedding_model)


def load_session_state(session_id: str) -> Optional[dict]:
    """
    Returns a dict with all data needed by the analysis page, or None if
    the session doesn't exist.

    Keys:
      session     : AnalysisSession ORM object
      phase       : int
      arrays      : {embeddings, umap_high, umap_3d} or None if phase < 1
      points      : list[Point]
      state       : ClusterState or None if phase < 2
      n_edits     : int
    """
    session = get_session(session_id)
    if session is None:
        return None

    phase  = session.phase
    points = get_points(session_id)

    # Load numpy arrays from disk/memory cache
    embedding_model = session.embedding_model or EMBEDDING_MODEL
    arrays = cache.load(session.csv_hash, embedding_model) if phase >= 1 else None

    # Reconstruct cluster state
    cluster_state = None
    if phase >= 2 and arrays is not None:
        db_clusters = get_clusters(session_id)
        db_assigns  = get_cluster_assignments(session_id)
        edits       = get_all_edits(session_id)

        # Base labels array from DB assignments
        n = len(points)
        base_labels = np.full(n, -1, dtype=np.int32)
        for a in db_assigns:
            # point_id is 1-indexed (autoincrement) — use position
            idx = a.point_id - points[0].id  # offset to 0-based
            if 0 <= idx < n:
                base_labels[idx] = a.cluster_id

        base_info = {
            c.cluster_id: {
                "title":       c.title,
                "description": c.description,
                "sentiment":   c.sentiment,
                "n_points":    c.n_points,
                "theme_name":  c.theme_name,
                "is_active":   c.is_active,
            }
            for c in db_clusters
        }

        cluster_state = reconstruct(base_labels, base_info, edits)

    from db.queries import count_edits
    return {
        "session":       session,
        "phase":         phase,
        "arrays":        arrays,
        "points":        points,
        "cluster_state": cluster_state,
        "n_edits":       count_edits(session_id),
    }
