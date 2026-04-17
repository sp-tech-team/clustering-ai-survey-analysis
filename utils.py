"""Shared utility helpers."""

import hashlib
from typing import List

import pandas as pd
import plotly.express as px

# ── Color palette ─────────────────────────────────────────────────────────────

_PALETTE = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.G10
    + px.colors.qualitative.Alphabet
)
OUTLIER_COLOR  = "#9E9E9E"
EXCLUDED_COLOR = "#BDBDBD"


def cluster_color(cluster_id: int, ordered_ids: List[int]) -> str:
    """Return a deterministic hex color for a cluster_id."""
    if cluster_id == -1:
        return OUTLIER_COLOR
    try:
        idx = ordered_ids.index(cluster_id)
        return _PALETTE[idx % len(_PALETTE)]
    except ValueError:
        return "#607D8B"


# ── CSV hashing ───────────────────────────────────────────────────────────────

def hash_csv(content_bytes: bytes) -> str:
    return hashlib.sha256(content_bytes).hexdigest()


# ── Column auto-detection ─────────────────────────────────────────────────────

_ID_HINTS    = {"id", "timestamp", "email", "respondent", "name", "user", "response_id", "#"}
_RESP_HINTS  = {"response", "answer", "feedback", "comment", "text", "message", "reflection"}


def guess_id_col(headers: List[str]) -> str | None:
    """Heuristic: pick the column that looks most like an identifier."""
    lowered = [h.lower() for h in headers]
    for hint in _ID_HINTS:
        for i, h in enumerate(lowered):
            if hint in h:
                return headers[i]
    return headers[0] if headers else None


def guess_response_col(headers: List[str]) -> str | None:
    """Heuristic: pick the column that looks most like a free-text response."""
    lowered = [h.lower() for h in headers]
    for hint in _RESP_HINTS:
        for i, h in enumerate(lowered):
            if hint in h:
                return headers[i]
    # Fallback: last column
    return headers[-1] if headers else None


# ── Sentiment colour ──────────────────────────────────────────────────────────

SENTIMENT_COLORS = {
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral":  "#2196F3",
    "mixed":    "#FF9800",
    "unknown":  "#9E9E9E",
}


def sentiment_color(sentiment: str) -> str:
    return SENTIMENT_COLORS.get(sentiment.lower(), "#9E9E9E")
