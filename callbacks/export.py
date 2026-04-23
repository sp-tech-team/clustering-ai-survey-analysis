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

from dash import Input, Output, State, callback, callback_context, html, no_update
import dash_bootstrap_components as dbc

from db.queries import get_session, get_points, get_clusters, get_cluster_assignments, get_all_edits
from core.state import reconstruct
from core.cache import load as cache_load
from core.export_centroid import compute_export_centroid_assignments
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


def _load_export_state(session_id: str):
    sess = get_session(session_id)
    if sess is None:
        raise ValueError(f"Unknown session: {session_id}")

    points = get_points(session_id)
    clusters = get_clusters(session_id)
    assignments = get_cluster_assignments(session_id)
    edits = get_all_edits(session_id)
    embedding_model = sess.embedding_model or EMBEDDING_MODEL
    cache = cache_load(sess.csv_hash, embedding_model)

    assignment_by_pid = {a.point_id: a.cluster_id for a in assignments}
    active_points = [p for p in points if p.status == "active" and p.id in assignment_by_pid]
    base_info = {
        c.cluster_id: {
            "title": c.title,
            "description": c.description,
            "sentiment": c.sentiment,
            "theme_name": c.theme_name,
            "n_points": c.n_points,
            "is_active": c.is_active,
        }
        for c in clusters
    }

    embeddings = None
    umap_3d = None
    if cache is not None and "point_ids" in cache:
        point_ids = list(cache["point_ids"])
        active_id_set = {p.id for p in active_points}
        active_mask = np.array([pid in active_id_set for pid in point_ids], dtype=bool)
        active_point_ids = [pid for pid in point_ids if pid in active_id_set and pid in assignment_by_pid]
        pid_to_point = {p.id: p for p in active_points}
        ordered_active_points = [pid_to_point[pid] for pid in active_point_ids]
        base_labels = np.array([assignment_by_pid[pid] for pid in active_point_ids], dtype=np.int32)
        if "embeddings" in cache:
            embeddings = cache["embeddings"][active_mask]
        if "umap_3d" in cache:
            umap_3d = cache["umap_3d"][active_mask]
    else:
        ordered_active_points = active_points
        base_labels = np.array([assignment_by_pid[p.id] for p in ordered_active_points], dtype=np.int32)

    state = reconstruct(base_labels, base_info, edits)
    return {
        "session": sess,
        "points": points,
        "clusters": clusters,
        "assignments": assignments,
        "state": state,
        "active_points": ordered_active_points,
        "embeddings": embeddings,
        "umap_3d": umap_3d,
    }


def build_export_preview(session_id: str) -> dict:
    export_state = _load_export_state(session_id)
    export_labels, secondary_map, diagnostics = compute_export_centroid_assignments(
        export_state["embeddings"],
        export_state["state"].labels,
        export_state["state"],
        max_secondary_clusters=MAX_SECONDARY_CLUSTERS,
    )
    diagnostics["points_exported"] = int(len(export_labels))
    diagnostics["points_with_secondaries"] = int(len(secondary_map))
    return diagnostics


@callback(
    Output("export-preview-content", "children"),
    Input("cluster-refresh-store", "data"),
    Input("session-id-store", "data"),
)
def render_export_preview(_refresh, session_id):
    if not session_id:
        return html.Div("No session loaded.", className="text-muted small")

    diagnostics = build_export_preview(session_id)
    if not diagnostics["available"]:
        return dbc.Alert(
            "Embedding cache is unavailable, so this export will use the current cluster labels without centroid refinement.",
            color="warning",
            className="mb-0 py-2 small",
        )

    threshold_text = (
        f"Thresholds: min {diagnostics['threshold_min']:.3f}, "
        f"median {diagnostics['threshold_median']:.3f}, max {diagnostics['threshold_max']:.3f}"
    )
    return dbc.Card(
        dbc.CardBody([
            html.Div("Export preview", className="fw-semibold small mb-2"),
            html.Div(f"Edited active clusters: {diagnostics['cluster_count']}", className="small"),
            html.Div(f"Other Themes before export: {diagnostics['outliers_before']}", className="small"),
            html.Div(f"Moved out of Other Themes at export: {diagnostics['outliers_absorbed']}", className="small"),
            html.Div(f"Other Themes remaining in export: {diagnostics['outliers_after']}", className="small"),
            html.Div(f"Points with secondary themes: {diagnostics['points_with_secondaries']}", className="small"),
            html.Div(f"Secondary links: {diagnostics['total_secondary_links']} total, max {diagnostics['max_secondary_links']} on one point", className="small"),
            html.Div(threshold_text, className="text-muted small mt-1"),
        ]),
        className="border-0 bg-light",
    )


def build_export_workbook(session_id: str) -> tuple[bytes, str]:
    export_state = _load_export_state(session_id)
    sess = export_state["session"]
    points = export_state["points"]
    assignments = export_state["assignments"]
    state = export_state["state"]
    total_uploaded = sess.n_points or len(points)
    active_pts = export_state["active_points"]
    umap_3d = export_state["umap_3d"]

    export_labels, secondary_map, _ = compute_export_centroid_assignments(
        export_state["embeddings"],
        state.labels,
        state,
        max_secondary_clusters=MAX_SECONDARY_CLUSTERS,
    )

    def _get_reps(cid: int, n: int = 5) -> list[str]:
        """Return up to n response texts closest to the cluster centroid."""
        idxs = np.where(export_labels == cid)[0]
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

    # ── Sheet 1: Responses — 3 columns only ──────────────────────────────────
    # theme column: primary cluster title (+ secondary titles separated by ", ")
    #   outlier / other-themes bucket → "Other Themes"
    #   any filtered/excluded → "no/low response"
    rows = []
    for idx, pt in enumerate(active_pts):
        export_label = int(export_labels[idx])
        is_outlier = export_label == -1
        if is_outlier:
            theme_val = "Other Themes"
        else:
            ci = state.info.get(export_label)
            if ci is None or not ci.is_active:
                theme_val = "no/low response"
            elif ci.theme_name == "Other Themes":
                theme_val = "Other Themes"
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
    inactive_active_ids = {pt.id for idx, pt in enumerate(active_pts) if int(state.labels[idx]) != -1 and not state.info.get(int(state.labels[idx]), None).is_active}
    if inactive_active_ids:
        rows = [row for row in rows if row[sess.id_col] not in {str(pid) for pid in inactive_active_ids}]
        for pt in active_pts:
            if pt.id in inactive_active_ids:
                rows.append({
                    sess.id_col: pt.orig_id,
                    sess.response_col: pt.response_text,
                    "theme": "no/low response",
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
    n_other_themes = int(np.sum(export_labels == -1))
    for cid in sorted(state.active_ids,
                      key=lambda c: int((export_labels == c).sum()), reverse=True):
        ci    = state.info.get(cid)
        if ci and ci.theme_name == "Other Themes":
            n_other_themes += int((export_labels == cid).sum())
            continue
        n     = int((export_labels == cid).sum())
        if n == 0:
            continue
        reps  = _get_reps(cid)
        summary_rows.append(_summary_row(
            title       = ci.title if ci else "",
            description = ci.description if ci else "",
            count       = n,
            pct         = round(100 * n / total_uploaded, 1),
            reps        = reps,
        ))

    if n_other_themes:
        summary_rows.append(_summary_row(
            title="Other Themes", description="Responses left in the other themes bucket after export refinement",
            count=n_other_themes, pct=round(100 * n_other_themes / total_uploaded, 1), reps=[],
        ))

    excluded_cids = {cid for cid, ci in state.info.items() if not ci.is_active and cid != -1}
    n_low_filter  = sum(
        1
        for p in points
        if p.status in ("low_info_structural", "low_info_llm", "low_info_user")
    )
    n_excluded    = sum(int((state.labels == cid).sum()) for cid in excluded_cids)
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

