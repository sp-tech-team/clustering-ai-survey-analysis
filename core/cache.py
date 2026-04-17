"""
In-memory + on-disk cache for numpy arrays.

Directory layout:
    cache/{csv_hash}/{embedding_model}/
      embeddings.npy
      umap_high.npy
      umap_3d.npy

The module-level dict `_mem` keeps arrays in RAM across requests so they
are not re-read from disk on every callback. The cache is populated/read
via load() and save(); everything else just calls get().
"""

import json
import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config import CACHE_DIR, EMBEDDING_MODEL

_mem:  Dict[str, Dict[str, np.ndarray]] = {}
_lock: threading.Lock = threading.Lock()

_KEYS = ("embeddings", "umap_high", "umap_3d", "point_ids")


def _cache_id(csv_hash: str, embedding_model: str) -> str:
    return f"{csv_hash}:{embedding_model}"


def _dir(csv_hash: str, embedding_model: str = EMBEDDING_MODEL) -> Path:
    return CACHE_DIR / csv_hash / embedding_model


def exists(csv_hash: str, embedding_model: str = EMBEDDING_MODEL) -> bool:
    """True when all three .npy files are present on disk."""
    d = _dir(csv_hash, embedding_model)
    return all((d / f"{k}.npy").exists() for k in _KEYS)


def save(csv_hash: str, arrays: Dict[str, np.ndarray], embedding_model: str = EMBEDDING_MODEL) -> None:
    """Persist arrays to disk and populate memory cache."""
    d = _dir(csv_hash, embedding_model)
    d.mkdir(parents=True, exist_ok=True)

    # Write all files before updating the in-memory cache so callers
    # that call exists() + load() see a consistent state.
    for k in _KEYS:
        np.save(d / f"{k}.npy", arrays[k])

    with _lock:
        _mem[_cache_id(csv_hash, embedding_model)] = {k: arrays[k] for k in _KEYS}


def load(csv_hash: str, embedding_model: str = EMBEDDING_MODEL) -> Optional[Dict[str, np.ndarray]]:
    """
    Return arrays from memory cache if available, otherwise load from disk.
    Returns None if files don't exist.
    """
    cache_id = _cache_id(csv_hash, embedding_model)
    with _lock:
        if cache_id in _mem:
            return _mem[cache_id]

    d = _dir(csv_hash, embedding_model)
    if not exists(csv_hash, embedding_model):
        return None

    arrays = {k: np.load(d / f"{k}.npy") for k in _KEYS}
    with _lock:
        _mem[cache_id] = arrays
    return arrays


def get(csv_hash: str, key: str, embedding_model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    """Convenience: return a single array."""
    data = load(csv_hash, embedding_model)
    return data[key] if data else None


def evict(csv_hash: str, embedding_model: str = EMBEDDING_MODEL) -> None:
    """Remove from memory cache (disk files are kept)."""
    with _lock:
        _mem.pop(_cache_id(csv_hash, embedding_model), None)


# ── Soft-membership cache (written once in phase 2) ───────────────────────────
# Files stored alongside the phase-1 arrays under the same hash directory:
#   membership.npy          — float32 (n_active, n_named_clusters)
#   membership_cids.npy     — int32   (n_named_clusters,)  column→cluster_id
#   membership_thresholds.npy — float64 (n_named_clusters,) per-cluster thresholds
#   membership_meta.json    — {"merge_map": {str: int}}    initial auto-merge map

_MEM_KEYS = ("membership", "membership_cids", "membership_thresholds")


def membership_exists(csv_hash: str, embedding_model: str = EMBEDDING_MODEL) -> bool:
    d = _dir(csv_hash, embedding_model)
    return all((d / f"{k}.npy").exists() for k in _MEM_KEYS)


def save_membership(
    csv_hash: str,
    membership: np.ndarray,
    cids: np.ndarray,
    thresholds: np.ndarray,
    merge_map: Dict[int, int],
    embedding_model: str = EMBEDDING_MODEL,
) -> None:
    d = _dir(csv_hash, embedding_model)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "membership.npy",            membership.astype(np.float32))
    np.save(d / "membership_cids.npy",       cids.astype(np.int32))
    np.save(d / "membership_thresholds.npy", thresholds.astype(np.float64))
    meta = {"merge_map": {str(k): int(v) for k, v in merge_map.items()}}
    (d / "membership_meta.json").write_text(json.dumps(meta))


def load_membership(csv_hash: str, embedding_model: str = EMBEDDING_MODEL) -> Optional[Dict]:
    """Returns dict with keys: membership, cids, thresholds, merge_map."""
    d = _dir(csv_hash, embedding_model)
    if not membership_exists(csv_hash, embedding_model):
        return None
    meta_path = d / "membership_meta.json"
    merge_map = {}
    if meta_path.exists():
        raw = json.loads(meta_path.read_text())
        merge_map = {int(k): int(v) for k, v in raw.get("merge_map", {}).items()}
    return {
        "membership":  np.load(d / "membership.npy"),
        "cids":        np.load(d / "membership_cids.npy"),
        "thresholds":  np.load(d / "membership_thresholds.npy"),
        "merge_map":   merge_map,
    }


# ── Centroid cosine cache (written once in phase 2) ───────────────────────────
# Files stored alongside the phase-1 arrays under the same hash directory:
#   centroid_vectors.npy    — float32 (n_clusters, emb_dim) centroid per canonical cluster
#   centroid_cids.npy       — int32   (n_clusters,)  row → canonical cluster id
#   centroid_thresholds.npy — float64 (n_clusters,)  per-cluster cosine similarity threshold

_CENT_KEYS = ("centroid_vectors", "centroid_cids", "centroid_thresholds")


def centroids_exist(csv_hash: str, embedding_model: str = EMBEDDING_MODEL) -> bool:
    d = _dir(csv_hash, embedding_model)
    return all((d / f"{k}.npy").exists() for k in _CENT_KEYS)


def save_centroids(
    csv_hash: str,
    centroids: np.ndarray,
    cids: np.ndarray,
    thresholds: np.ndarray,
    embedding_model: str = EMBEDDING_MODEL,
) -> None:
    d = _dir(csv_hash, embedding_model)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "centroid_vectors.npy",    centroids.astype(np.float32))
    np.save(d / "centroid_cids.npy",       cids.astype(np.int32))
    np.save(d / "centroid_thresholds.npy", thresholds.astype(np.float64))


def load_centroids(csv_hash: str, embedding_model: str = EMBEDDING_MODEL) -> Optional[Dict]:
    """Returns dict with keys: centroids, cids, thresholds — or None if not cached."""
    d = _dir(csv_hash, embedding_model)
    if not centroids_exist(csv_hash, embedding_model):
        return None
    return {
        "centroids":  np.load(d / "centroid_vectors.npy"),
        "cids":       np.load(d / "centroid_cids.npy"),
        "thresholds": np.load(d / "centroid_thresholds.npy"),
    }
