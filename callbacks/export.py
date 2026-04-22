"""Export callback helpers and modal controls.

Flow:
    btn-export     -> opens modal-export
    export-confirm -> follows attachment URL
    export-cancel  -> closes modal
"""

import io
from urllib.parse import quote

import numpy as np
import pandas as pd

from dash import Input, Output, State, callback, callback_context, no_update

from db.queries import get_session, get_points, get_clusters, get_cluster_assignments, get_all_edits
from core.state import reconstruct
from core.cache import load as cache_load, load_membership
from core.secondary_membership import replay_secondary_memberships
from config import EMBEDDING_MODEL, MAX_SECONDARY_CLUSTERS


# ── Open / close modal ────────────────────────────────────────────────────────

@callback(
    Output("modal-export",           "is_open"),
    Input("btn-export",              "n_clicks"),
    Input("export-cancel",           "n_clicks"),
    Input("export-confirm",          "n_clicks"),
    State("modal-export",            "is_open"),
    State("session-id-store",        "data"),
    prevent_initial_call=True,
)
def toggle_export_modal(open_n, cancel_n, confirm_n, is_open, session_id):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "btn-export":
        return True
    return False


@callback(
    Output("export-confirm",        "href"),
    Input("session-id-store",       "data"),
)
def update_export_href(session_id):
    if not session_id:
        return "#"

    return f"/api/export/{quote(str(session_id), safe='')}"


def build_export_workbook(session_id: str) -> tuple[bytes, str]:
    sess = get_session(session_id)
    if sess is None:
        raise ValueError(f"Unknown session: {session_id}")

    points      = get_points(session_id)
    clusters    = get_clusters(session_id)
    assignments = get_cluster_assignments(session_id)
    edits       = get_all_edits(session_id)

    base_labels = np.array([a.cluster_id for a in assignments], dtype=int)
    base_info   = {
        c.cluster_id: {
            "title": c.title, "description": c.description,
            "sentiment": c.sentiment, "theme_name": c.theme_name,
            "n_points": c.n_points, "is_active": c.is_active,
        }
        for c in clusters
    }
    state          = reconstruct(base_labels, base_info, edits)
    total_uploaded = sess.n_points or len(points)
    active_pts     = [p for p in points if p.status == "active"]
    embedding_model = sess.embedding_model or EMBEDDING_MODEL

    # ── Load UMAP coords for representative selection ─────────────────────────
    cache    = cache_load(sess.csv_hash, embedding_model)
    umap_3d  = cache["umap_3d"] if cache is not None else None  # (N_active, 3)

    def _get_reps(cid: int, n: int = 5) -> list[str]:
        """Return up to n response texts closest to the cluster centroid."""
        idxs = np.where(state.labels == cid)[0]
        if len(idxs) == 0:
            return []
        if umap_3d is not None and len(idxs) > n:
            coords   = umap_3d[idxs]
            centroid = coords.mean(axis=0)
            order    = np.argsort(np.linalg.norm(coords - centroid, axis=1))
            idxs     = idxs[order[:n]]
        else:
            idxs = idxs[:n]
        return [active_pts[i].response_text for i in idxs]

    # ── Compute secondary assignments via unified membership replay ───────────
    # secondary_map: point array index -> list of secondary cluster ids
    secondary_map: dict[int, list[int]] = {}
    mem_data = load_membership(sess.csv_hash, embedding_model)
    secondary_map, _ = replay_secondary_memberships(
        mem_data,
        edits,
        state,
        max_secondary_clusters=MAX_SECONDARY_CLUSTERS,
    )

    # ── Sheet 1: Responses — 3 columns only ──────────────────────────────────
    # theme column: primary cluster title (+ secondary titles separated by ", ")
    #   outlier  → "Outliers"
    #   any filtered/excluded → "no/low response"
    rows = []
    for idx, (pt, lbl) in enumerate(zip(active_pts, state.labels)):
        is_outlier = int(lbl) == -1
        if is_outlier:
            theme_val = "Outliers"
        else:
            ci = state.info.get(int(lbl))
            if ci is None or not ci.is_active:
                theme_val = "no/low response"
            else:
                primary = ci.title
                if idx in secondary_map:
                    all_themes = [primary] + [state.info[cid].title for cid in secondary_map[idx] if cid in state.info]
                else:
                    all_themes = [primary]
                theme_val = ", ".join(t for t in all_themes if t)
        rows.append({
            sess.id_col:       pt.orig_id,
            sess.response_col: pt.response_text,
            "theme":           theme_val,
        })

    for pt in points:
        if pt.status != "active":
            rows.append({
                sess.id_col:       pt.orig_id,
                sess.response_col: pt.response_text,
                "theme":           "no/low response",
            })

    df_responses = pd.DataFrame(rows)

    # ── Sheet 2: Cluster Summary ──────────────────────────────────────────────
    # Columns: theme, theme description, count, pct_of_total, rep_1…rep_5
    N_REPS = 5
    rep_cols = [f"rep_{i+1}" for i in range(N_REPS)]

    def _summary_row(title, description, count, pct, reps):
        row = {"theme": title, "theme description": description,
               "count": count, "pct_of_total": pct}
        for i, col in enumerate(rep_cols):
            row[col] = reps[i] if i < len(reps) else ""
        return row

    summary_rows = []
    for cid in sorted(state.active_ids,
                      key=lambda c: int((state.labels == c).sum()), reverse=True):
        ci    = state.info.get(cid)
        n     = int((state.labels == cid).sum())
        reps  = _get_reps(cid)
        summary_rows.append(_summary_row(
            title       = ci.title if ci else "",
            description = ci.description if ci else "",
            count       = n,
            pct         = round(100 * n / total_uploaded, 1),
            reps        = reps,
        ))

    n_outliers = int((state.labels == -1).sum())
    if n_outliers:
        summary_rows.append(_summary_row(
            title="Outliers", description="Responses not assigned to any cluster",
            count=n_outliers, pct=round(100 * n_outliers / total_uploaded, 1), reps=[],
        ))

    excluded_cids = {cid for cid, ci in state.info.items() if not ci.is_active and cid != -1}
    n_low_filter  = sum(
        1
        for p in points
        if p.status in ("low_info_structural", "low_info_llm", "low_info_user")
    )
    n_excluded    = sum(int((base_labels == cid).sum()) for cid in excluded_cids)
    n_low_total   = n_low_filter + n_excluded
    if n_low_total:
        summary_rows.append(_summary_row(
            title="no/low response",
            description="Filtered responses + user-excluded clusters",
            count=n_low_total, pct=round(100 * n_low_total / total_uploaded, 1), reps=[],
        ))

    df_summary = pd.DataFrame(summary_rows)

    # ── Write Excel ────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_responses.to_excel(writer, sheet_name="Responses",       index=False)
        df_summary.to_excel(  writer, sheet_name="Cluster Summary", index=False)

    filename_root = (sess.session_name or "clustered_export").strip().replace(" ", "_")
    filename = f"{filename_root}_clustered.xlsx"
    return buf.getvalue(), filename

