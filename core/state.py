"""
ClusterState — mutable in-memory representation of the current cluster layout.

It is always derived by replaying the edit log on top of the base HDBSCAN state.
Never persisted directly; reconstructed on every load (takes < 1 s).

Edit payloads (stored in DB cluster_edits.payload):

  join    : {from_ids, to_id, title, description, sentiment}
  split   : {from_id, new_assignments: [[point_idx, new_cid], …],
             new_cluster_info: {str(cid): {title, description, sentiment}}}
  rename  : {cluster_id, title, description}
  exclude : {cluster_id, reason}          → is_active=False, status="excluded"
  unexclude: {cluster_id}                 → is_active=True
  theme   : {cluster_ids, theme_name}
"""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ClusterInfo:
    cluster_id:  int
    title:       str
    description: str
    sentiment:   str
    n_points:    int
    theme_name:  Optional[str] = None
    is_active:   bool          = True
    status:      str           = "active"   # active | excluded


class ClusterState:
    """
    Parameters
    ----------
    base_labels     : np.ndarray (n_points,) — raw HDBSCAN output
    base_cluster_info : dict[int, dict]      — from DB clusters table
    """

    def __init__(
        self,
        base_labels:      np.ndarray,
        base_cluster_info: Dict[int, dict],
    ):
        self.labels: np.ndarray = base_labels.copy().astype(np.int32)
        self.info: Dict[int, ClusterInfo] = {
            k: ClusterInfo(**{
                "cluster_id":  k,
                "title":       v.get("title", f"Cluster {k}"),
                "description": v.get("description", ""),
                "sentiment":   v.get("sentiment", "unknown"),
                "n_points":    int(v.get("n_points", 0)),
                "theme_name":  v.get("theme_name"),
                "is_active":   v.get("is_active", True),
                "status":      v.get("status", "active"),
            })
            for k, v in base_cluster_info.items()
        }
        self._next_id: int = max((k for k in base_cluster_info if k >= 0), default=0) + 1

    # ── Edit application ──────────────────────────────────────────────────────

    def apply(self, edit_type: str, payload: dict) -> None:
        handler = {
            "join":      self._join,
            "split":     self._split,
            "rename":    self._rename,
            "exclude":   self._exclude,
            "unexclude": self._unexclude,
            "theme":     self._theme,
        }.get(edit_type)
        if handler:
            handler(payload)

    def _join(self, p: dict) -> None:
        from_ids = p["from_ids"]
        to_id    = p["to_id"]
        for fid in from_ids:
            self.labels[self.labels == fid] = to_id
            if fid != to_id:
                self.info.pop(fid, None)
        if to_id in self.info:
            ci = self.info[to_id]
            ci.title       = p["title"]
            ci.description = p.get("description", ci.description)
            ci.sentiment   = p.get("sentiment", ci.sentiment)
            ci.n_points    = int((self.labels == to_id).sum())

    def _split(self, p: dict) -> None:
        from_id = p["from_id"]
        self.info.pop(from_id, None)

        for point_idx, new_cid in p["new_assignments"]:
            self.labels[int(point_idx)] = int(new_cid)

        for cid_str, ci_dict in p["new_cluster_info"].items():
            cid = int(cid_str)
            self.info[cid] = ClusterInfo(
                cluster_id=cid,
                title=ci_dict.get("title", f"Cluster {cid}"),
                description=ci_dict.get("description", ""),
                sentiment=ci_dict.get("sentiment", "unknown"),
                n_points=int((self.labels == cid).sum()),
                theme_name=None,
                is_active=True,
            )
            self._next_id = max(self._next_id, cid + 1)

    def _rename(self, p: dict) -> None:
        cid = p["cluster_id"]
        if cid in self.info:
            self.info[cid].title       = p["title"]
            self.info[cid].description = p.get("description", self.info[cid].description)

    def _exclude(self, p: dict) -> None:
        cid = p["cluster_id"]
        if cid in self.info:
            self.info[cid].is_active = False
            self.info[cid].status    = p.get("reason", "excluded")

    def _unexclude(self, p: dict) -> None:
        cid = p["cluster_id"]
        if cid in self.info:
            self.info[cid].is_active = True
            self.info[cid].status    = "active"

    def _theme(self, p: dict) -> None:
        name = p["theme_name"]
        for cid in p["cluster_ids"]:
            if cid in self.info:
                self.info[cid].theme_name = name

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def active_ids(self) -> List[int]:
        return sorted(k for k, v in self.info.items() if v.is_active and k != -1)

    @property
    def next_id(self) -> int:
        return self._next_id

    def cluster_for_point(self, idx: int) -> int:
        return int(self.labels[idx])

    def point_indices_for_cluster(self, cid: int) -> List[int]:
        return list(np.where(self.labels == cid)[0])

    def to_sidebar_items(self) -> List[dict]:
        """Serialisable list for the sidebar checklist."""
        items = []
        for cid in self.active_ids:
            ci = self.info[cid]
            items.append({
                "cluster_id":  cid,
                "title":       ci.title,
                "n_points":    ci.n_points,
                "sentiment":   ci.sentiment,
                "theme_name":  ci.theme_name,
            })
        # Outliers last
        if -1 in self.info and self.info[-1].is_active:
            ci = self.info[-1]
            items.append({
                "cluster_id":  -1,
                "title":       "Outliers / Ungrouped",
                "n_points":    ci.n_points,
                "sentiment":   "neutral",
                "theme_name":  None,
            })
        return items


# ── Replay helper ──────────────────────────────────────────────────────────────

def reconstruct(
    base_labels: np.ndarray,
    base_cluster_info: Dict[int, dict],
    edits: list,           # list of ClusterEdit ORM objects
) -> ClusterState:
    """Replay edit log on top of base HDBSCAN state."""
    state = ClusterState(base_labels, base_cluster_info)
    for edit in edits:
        state.apply(edit.edit_type, edit.payload)
    return state
