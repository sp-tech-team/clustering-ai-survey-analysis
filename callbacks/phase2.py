"""Phase 2 callbacks: interactive editing (join / split / rename / theme / exclude / undo).

Clustering start, Phase 2 worker, and progress polling have been moved to
callbacks/phase_controller.py to avoid Dash duplicate-output errors.
"""

import json
import random

from dash import (
    Input, Output, State, callback, html, no_update,
    callback_context, ALL,
)
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go

import tasks
from db.queries import (
    get_session, advance_phase, get_points,
    save_clusters, save_cluster_assignments,
    get_clusters, get_cluster_assignments,
    log_edit, get_all_edits, undo_last_edit,
)
from core.state import reconstruct
from config import EMBEDDING_MODEL
from core.cache import load as cache_get
from utils import cluster_color
from layout.components import cluster_list_item, action_buttons


# ─────────────────────────────────────────────────────────────────────────────
# Render cluster list
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("cluster-list-container", "children"),
    Output("cluster-count-badge",    "children"),
    Output("clustering-diagnostics-container", "children"),
    Input("cluster-refresh-store",   "data"),
    State("session-id-store",        "data"),
    State("selection-store",         "data"),
    prevent_initial_call=True,
)
def render_cluster_list(refresh, session_id, sel_data):
    if not session_id:
        return "", "", ""

    clusters = get_clusters(session_id)
    edits    = get_all_edits(session_id)
    if not clusters:
        return html.P("No clusters yet.", className="text-muted small p-2"), "0", ""

    assignments = get_cluster_assignments(session_id)
    base_labels = np.array([a.cluster_id for a in assignments], dtype=int)
    base_info   = {
        c.cluster_id: {
            "title": c.title, "description": c.description,
            "sentiment": c.sentiment, "theme_name": c.theme_name,
            "n_points": c.n_points, "is_active": c.is_active,
        }
        for c in clusters
    }
    state = reconstruct(base_labels, base_info, edits)

    sess = get_session(session_id)
    total_uploaded = sess.n_points if sess else max(len(base_labels), 1)

    # Sort descending by count
    active_ids_sorted = sorted(
        state.active_ids,
        key=lambda cid: int((state.labels == cid).sum()),
        reverse=True,
    )

    items = []
    for cid in active_ids_sorted:
        ci = state.info.get(cid)
        n  = int((state.labels == cid).sum())
        pct = round(100 * n / total_uploaded, 1)
        items.append(cluster_list_item(
            cluster_id=cid,
            title=ci.title if ci else f"Cluster {cid}",
            n_points=n,
            pct=pct,
            theme_name=ci.theme_name if ci else None,
            description=ci.description if ci else None,
        ))

    # Footer: outliers + low-info (includes user-excluded clusters)
    all_points   = get_points(session_id)
    n_outliers   = int((state.labels == -1).sum())
    n_other_themes = sum(
        int((state.labels == cid).sum())
        for cid, ci in state.info.items()
        if cid != -1 and ci.is_active and ci.theme_name == "Other Themes"
    )
    # Low-info from filtering
    n_low_filter = sum(
        1
        for p in all_points
        if p.status in ("low_info_structural", "low_info_llm", "low_info_user")
    )
    # Points whose cluster was marked inactive (user excluded)
    excluded_cids = {cid for cid, ci in state.info.items() if not ci.is_active and cid != -1}
    assignments_all = get_cluster_assignments(session_id)
    base_for_excl   = np.array([a.cluster_id for a in assignments_all], dtype=int)
    n_excluded   = sum(int((base_for_excl == cid).sum()) for cid in excluded_cids)
    n_low_info   = n_low_filter + n_excluded

    footer = []
    if n_outliers:
        pct_o = round(100 * n_outliers / total_uploaded, 1)
        footer.append(dbc.ListGroupItem(
            [html.Span("○ Outliers", className="text-muted small me-2"),
             dbc.Badge(f"{n_outliers} ({pct_o}%)", color="light", text_color="dark")],
            style={"padding": "5px 10px"},
        ))
    if n_other_themes:
        pct_t = round(100 * n_other_themes / total_uploaded, 1)
        footer.append(dbc.ListGroupItem(
            [html.Span("◌ Other Themes", className="text-muted small me-2"),
             dbc.Badge(f"{n_other_themes} ({pct_t}%)", color="light", text_color="dark")],
            style={"padding": "5px 10px"},
        ))
    if n_low_info:
        pct_l = round(100 * n_low_info / total_uploaded, 1)
        footer.append(dbc.ListGroupItem(
            [html.Span("✕ Low-info filtered", className="text-muted small me-2"),
             dbc.Badge(f"{n_low_info} ({pct_l}%)", color="light", text_color="dark")],
            style={"padding": "5px 10px"},
        ))

    return dbc.ListGroup(items + footer, flush=True), str(len(active_ids_sorted)), ""


# ─────────────────────────────────────────────────────────────────────────────
# Track checkbox selections → update selection-store + action buttons
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("selection-store",          "data",   allow_duplicate=True),
    Output("action-buttons-container", "children"),
    Input({"type": "cluster-check", "index": ALL}, "value"),
    State({"type": "cluster-check", "index": ALL}, "id"),
    State("selection-store",           "data"),
    prevent_initial_call=True,
)
def sync_selection(values, ids, sel_data):
    selected = [
        id_obj["index"]
        for id_obj, val in zip(ids, values)
        if val
    ]
    data = {**(sel_data or {}), "selected_cluster_ids": selected, "clicked_point_idx": None}
    return data, action_buttons(len(selected))


def _get_active_projection_state(session_id):
    if not session_id:
        return None

    sess = get_session(session_id)
    if not sess:
        return None

    arrays = cache_get(sess.csv_hash, sess.embedding_model or EMBEDDING_MODEL)
    if arrays is None:
        return None

    assignments = get_cluster_assignments(session_id)
    edits = get_all_edits(session_id)
    clusters = get_clusters(session_id)
    if not assignments:
        return {
            "umap_3d": arrays["umap_3d"],
            "state": None,
            "active_points_ordered": [],
            "active_ids": [],
        }

    point_ids = list(arrays["point_ids"])
    points = get_points(session_id)
    active_points = [p for p in points if p.status == "active"]
    active_id_set = {p.id for p in active_points}
    active_mask = np.array([pid in active_id_set for pid in point_ids], dtype=bool)
    active_point_ids = [pid for pid in point_ids if pid in active_id_set]
    assignment_by_pid = {a.point_id: a.cluster_id for a in assignments}
    pid_to_point = {p.id: p for p in active_points}

    if not active_point_ids or any(pid not in assignment_by_pid for pid in active_point_ids):
        return None

    umap_3d = arrays["umap_3d"][active_mask]
    active_points_ordered = [pid_to_point[pid] for pid in active_point_ids]
    base_labels = np.array([assignment_by_pid[pid] for pid in active_point_ids], dtype=int)
    base_info = {
        c.cluster_id: {
            "title": c.title, "description": c.description,
            "sentiment": c.sentiment, "theme_name": c.theme_name,
            "n_points": c.n_points, "is_active": c.is_active,
        }
        for c in clusters
    }
    state = reconstruct(base_labels, base_info, edits)
    return {
        "umap_3d": umap_3d,
        "state": state,
        "active_points_ordered": active_points_ordered,
        "active_ids": state.active_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3-D scatter plot
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("scatter-3d",           "figure"),
    Input("cluster-refresh-store", "data"),
    Input("selection-store",       "data"),
    State("session-id-store",      "data"),
    prevent_initial_call=True,
)
def render_scatter(refresh, sel_data, session_id):
    try:
        return _render_scatter_inner(session_id, sel_data)
    except Exception as exc:
        import traceback
        fig = go.Figure()
        fig.add_annotation(text=f"Graph error: {exc}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="red"))
        return fig


def _render_scatter_inner(session_id, sel_data):
    projection = _get_active_projection_state(session_id)
    if projection is None:
        return go.Figure()

    if projection["state"] is None:
        umap_3d = projection["umap_3d"]
        # Phase 1 only — show grey cloud
        fig = go.Figure(go.Scatter3d(
            x=umap_3d[:, 0], y=umap_3d[:, 1], z=umap_3d[:, 2],
            mode="markers",
            marker=dict(size=3, color="#adb5bd", opacity=0.7),
            hoverinfo="skip",
        ))
        _apply_layout(fig, "3-D Embedding (no clusters yet)")
        return fig

    umap_3d = projection["umap_3d"]
    active_points_ordered = projection["active_points_ordered"]
    state = projection["state"]
    selected = set(sel_data.get("selected_cluster_ids", []) if sel_data else [])
    active_ids = projection["active_ids"]
    clicked_idx = sel_data.get("clicked_point_idx") if sel_data else None

    traces = []
    for cid in active_ids:
        ci = state.info.get(cid)
        color = cluster_color(cid, active_ids)
        opacity = 1.0 if not selected or cid in selected else 0.05

        mask = state.labels == cid
        pt_subset = [p for i, p in enumerate(active_points_ordered) if state.labels[i] == cid]
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue

        coords = umap_3d[idxs]
        cl_title = ci.title if ci else f"Cluster {cid}"
        custom = [
            [p.response_text[:120], p.orig_id, cl_title, p.response_text, int(idx)]
            for idx, p in zip(idxs, pt_subset)
        ]

        traces.append(go.Scatter3d(
            name=ci.title if ci else f"Cluster {cid}",
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=4, color=color, opacity=opacity),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "ID: %{customdata[1]}<br>"
                "Cluster: %{customdata[2]}<extra></extra>"
            ),
        ))

    # Outliers
    outlier_mask = state.labels == -1
    if outlier_mask.any():
        idxs = np.where(outlier_mask)[0]
        coords = umap_3d[idxs]
        traces.append(go.Scatter3d(
            name="Outliers",
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=2, color="#adb5bd", opacity=0.3),
            hoverinfo="skip",
        ))

    fig = go.Figure(traces)
    _apply_layout(fig, "")
    return fig


def _apply_layout(fig: go.Figure, title: str):
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title=title,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showticklabels=False, title=""),
        ),
        legend=dict(
            orientation="v", x=1, y=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#dee2e6", borderwidth=1,
        ),
        paper_bgcolor="white",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Point detail on click
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("selection-store",        "data", allow_duplicate=True),
    Input("scatter-3d",              "clickData"),
    State("selection-store",         "data"),
    prevent_initial_call=True,
)
def set_clicked_point(click_data, sel_data):
    if not click_data:
        return no_update
    pt = click_data["points"][0]
    custom = pt.get("customdata", ["", "", "", "", None])
    point_idx = custom[4] if len(custom) > 4 else None
    if point_idx is None:
        return no_update
    return {**(sel_data or {}), "clicked_point_idx": int(point_idx)}


@callback(
    Output("selection-store",        "data", allow_duplicate=True),
    Input("point-nav-left",          "n_clicks"),
    Input("point-nav-right",         "n_clicks"),
    State("selection-store",         "data"),
    State("session-id-store",        "data"),
    prevent_initial_call=True,
)
def navigate_cluster_points(left_clicks, right_clicks, sel_data, session_id):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    projection = _get_active_projection_state(session_id)
    if projection is None or projection["state"] is None:
        return no_update

    state = projection["state"]
    current_idx = (sel_data or {}).get("clicked_point_idx")
    selected = (sel_data or {}).get("selected_cluster_ids", [])

    if current_idx is not None and 0 <= int(current_idx) < len(state.labels):
        cid = int(state.labels[int(current_idx)])
    elif len(selected) == 1:
        cid = int(selected[0])
    else:
        return no_update

    cluster_indices = np.where(state.labels == cid)[0].tolist()
    if not cluster_indices:
        return no_update

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if current_idx not in cluster_indices:
        next_idx = cluster_indices[-1] if trigger == "point-nav-left" else cluster_indices[0]
    else:
        pos = cluster_indices.index(current_idx)
        delta = -1 if trigger == "point-nav-left" else 1
        next_idx = cluster_indices[(pos + delta) % len(cluster_indices)]

    return {**(sel_data or {}), "clicked_point_idx": int(next_idx)}


@callback(
    Output("point-response-display", "children"),
    Output("point-id-display",       "children"),
    Output("point-cluster-display",  "children"),
    Output("point-detail-collapse",  "is_open"),
    Input("selection-store",         "data"),
    State("session-id-store",        "data"),
    prevent_initial_call=True,
)
def show_point_detail(sel_data, session_id):
    clicked_idx = (sel_data or {}).get("clicked_point_idx")
    if clicked_idx is None:
        return "", "", "", False
    projection = _get_active_projection_state(session_id)
    if projection is None or projection["state"] is None:
        return "", "", "", False

    active_points_ordered = projection["active_points_ordered"]
    state = projection["state"]
    point_idx = int(clicked_idx)
    if point_idx < 0 or point_idx >= len(active_points_ordered):
        return "", "", "", False

    point = active_points_ordered[point_idx]
    cid = int(state.labels[point_idx])
    ci = state.info.get(cid)
    cluster_label = ci.title if ci else ("Outliers" if cid == -1 else f"Cluster {cid}")
    return point.response_text, f"ID: {point.orig_id}", f"Cluster: {cluster_label}", bool(point.response_text)


# ─────────────────────────────────────────────────────────────────────────────
# JOIN – open modal with LLM suggestion
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("modal-join",        "is_open",    allow_duplicate=True),
    Output("join-title-input",  "value"),
    Output("join-desc-input",   "value"),
    Output("join-preview-text", "children"),
    Output("join-working-store","data"),
    Input("btn-join",           "n_clicks"),
    State("session-id-store",   "data"),
    State("selection-store",    "data"),
    prevent_initial_call=True,
)
def open_join_modal(n, session_id, sel_data):
    if not n or not session_id:
        return False, "", "", "", {}

    selected = sel_data.get("selected_cluster_ids", []) if sel_data else []
    if len(selected) < 2:
        return False, "", "", "", {}

    clusters = get_clusters(session_id)
    cmap = {c.cluster_id: c for c in clusters}

    titles = [cmap[cid].title for cid in selected if cid in cmap]
    preview = "Joining: " + " + ".join(titles)

    # LLM merge suggestion
    try:
        sess = get_session(session_id)
        projection = _get_active_projection_state(session_id)
        if projection is None or projection["state"] is None:
            raise RuntimeError("Projection state unavailable")

        from openai import OpenAI
        from core.llm import summarise_join
        client = OpenAI(api_key=sess.api_key)

        state = projection["state"]
        active_points_ordered = projection["active_points_ordered"]
        umap_3d = projection["umap_3d"]

        cluster_inputs = []
        pooled_texts = []
        for cid in selected:
            if cid not in cmap:
                continue
            idxs = np.where(state.labels == cid)[0]
            if len(idxs) == 0:
                continue
            if len(idxs) > 10:
                coords = umap_3d[idxs]
                centroid = coords.mean(axis=0)
                order = np.argsort(np.linalg.norm(coords - centroid, axis=1))
                idxs = idxs[order[:10]]
            rep_texts = [active_points_ordered[int(i)].response_text for i in idxs if int(i) < len(active_points_ordered)]
            cluster_inputs.append({
                "title": cmap[cid].title,
                "count": int((state.labels == cid).sum()),
            })
            pooled_texts.extend(rep_texts)

        sampled_pool = random.sample(pooled_texts, min(10, len(pooled_texts))) if pooled_texts else []

        suggestion  = summarise_join(
            client,
            cluster_inputs,
            sampled_pool,
            question=sess.response_col or "",
        )
        title_val = suggestion.get("title", titles[0])
        desc_val  = suggestion.get("description", "")
    except Exception:
        title_val, desc_val = titles[0], ""

    working = {"from_ids": selected}
    return True, title_val, desc_val, preview, working


@callback(
    Output("modal-join",           "is_open",  allow_duplicate=True),
    Output("cluster-refresh-store","data",     allow_duplicate=True),
    Output("selection-store",      "data",     allow_duplicate=True),
    Input("join-confirm",          "n_clicks"),
    Input("join-cancel",           "n_clicks"),
    State("join-title-input",      "value"),
    State("join-desc-input",       "value"),
    State("join-working-store",    "data"),
    State("session-id-store",      "data"),
    State("cluster-refresh-store", "data"),
    prevent_initial_call=True,
)
def confirm_join(confirm, cancel, title, desc, working, session_id, refresh):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "join-cancel" or not working:
        return False, no_update, no_update

    from_ids = working.get("from_ids", [])
    if not from_ids or not session_id:
        return False, no_update, no_update

    to_id = from_ids[0]
    payload = {
        "from_ids": from_ids,
        "to_id": to_id,
        "title": title or f"Cluster {to_id}",
        "description": desc or "",
    }
    log_edit(session_id, "join", payload)
    return False, (refresh or 0) + 1, {"selected_cluster_ids": [], "clicked_point_idx": None}


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("cluster-refresh-store","data",     allow_duplicate=True),
    Output("notification-toast",  "header",   allow_duplicate=True),
    Output("notification-toast",  "children", allow_duplicate=True),
    Output("notification-toast",  "is_open",  allow_duplicate=True),
    Input("btn-split",            "n_clicks"),
    State("session-id-store",     "data"),
    State("selection-store",      "data"),
    State("cluster-refresh-store","data"),
    prevent_initial_call=True,
)
def do_split(n, session_id, sel_data, refresh):
    if not n or not session_id:
        return no_update, no_update, no_update, False

    selected = sel_data.get("selected_cluster_ids", []) if sel_data else []
    if len(selected) != 1:
        return no_update, "Split", "Select exactly one cluster to split.", True

    cid = selected[0]
    try:
        sess    = get_session(session_id)
        arrays  = cache_get(sess.csv_hash, sess.embedding_model or EMBEDDING_MODEL)
        if arrays is None:
            raise RuntimeError("Cache missing")
        point_ids_all = list(arrays["point_ids"])
        assignments = get_cluster_assignments(session_id)
        edits       = get_all_edits(session_id)
        clusters    = get_clusters(session_id)
        points = get_points(session_id)
        active_pts = [p for p in points if p.status == "active"]
        active_id_set = {p.id for p in active_pts}
        active_mask = np.array([pid in active_id_set for pid in point_ids_all], dtype=bool)
        active_point_ids = [pid for pid in point_ids_all if pid in active_id_set]
        assignment_by_pid = {a.point_id: a.cluster_id for a in assignments}
        pid_to_point = {p.id: p for p in active_pts}

        if not active_point_ids or any(pid not in assignment_by_pid for pid in active_point_ids):
            raise RuntimeError("Active points and cluster assignments are out of sync")

        umap_high = arrays["umap_high"][active_mask]
        active_pts_ordered = [pid_to_point[pid] for pid in active_point_ids]
        base_labels = np.array([assignment_by_pid[pid] for pid in active_point_ids], dtype=int)
        base_info   = {c.cluster_id: {"title": c.title, "description": c.description,
                                       "sentiment": c.sentiment, "theme_name": c.theme_name,
                                       "n_points": c.n_points, "is_active": c.is_active}
                        for c in clusters}
        state = reconstruct(base_labels, base_info, edits)

        point_indices = list(np.where(state.labels == cid)[0])
        if len(point_indices) < 6:
            return no_update, "Split", f"Cluster {cid} has too few points to split.", True

        next_id = max((k for k in state.info if k >= 0), default=0) + 1
        from core.splitter import split_cluster
        new_assignments, new_ids = split_cluster(
            cid,
            point_indices,
            umap_high,
            next_id,
        )

        # LLM summarise new sub-clusters
        from openai import OpenAI
        from core.llm import summarise_cluster
        client = OpenAI(api_key=sess.api_key)

        new_cluster_info = {}
        for ncid in new_ids:
            idxs = [pt_idx for pt_idx, assigned_cid in new_assignments.items() if assigned_cid == ncid]
            texts = [active_pts_ordered[i].response_text for i in idxs[:10] if i < len(active_pts_ordered)]
            info  = summarise_cluster(client, texts)
            new_cluster_info[str(ncid)] = info

        payload = {
            "from_id": int(cid),
            "new_assignments": [[int(k), int(v)] for k, v in new_assignments.items()],
            "new_cluster_info": new_cluster_info,
        }
        log_edit(session_id, "split", payload)
        return (refresh or 0) + 1, "Split done", f"Cluster {cid} split into {len(new_ids)} sub-clusters.", True

    except Exception as exc:
        return no_update, "Error", str(exc), True


# ─────────────────────────────────────────────────────────────────────────────
# EXCLUDE / LOW-INFO
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("cluster-refresh-store","data",     allow_duplicate=True),
    Input("btn-low-info",         "n_clicks"),
    State("session-id-store",     "data"),
    State("selection-store",      "data"),
    State("cluster-refresh-store","data"),
    prevent_initial_call=True,
)
def exclude_cluster(n, session_id, sel_data, refresh):
    if not n or not session_id:
        return no_update
    selected = sel_data.get("selected_cluster_ids", []) if sel_data else []
    if len(selected) != 1:
        return no_update
    log_edit(session_id, "exclude", {"cluster_id": selected[0], "set_to": "low_info"})
    return (refresh or 0) + 1


@callback(
    Output("cluster-refresh-store","data",     allow_duplicate=True),
    Input("btn-other-themes",     "n_clicks"),
    State("session-id-store",     "data"),
    State("selection-store",      "data"),
    State("cluster-refresh-store","data"),
    prevent_initial_call=True,
)
def assign_other_themes(n, session_id, sel_data, refresh):
    if not n or not session_id:
        return no_update
    selected = sel_data.get("selected_cluster_ids", []) if sel_data else []
    if not selected:
        return no_update
    log_edit(session_id, "theme", {"cluster_ids": [int(cid) for cid in selected], "theme_name": "Other Themes"})
    return (refresh or 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
# RENAME modal
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("modal-rename",         "is_open",  allow_duplicate=True),
    Output("rename-title-input",   "value",    allow_duplicate=True),
    Output("rename-desc-input",    "value",    allow_duplicate=True),
    Input("btn-rename",            "n_clicks"),
    State("session-id-store",      "data"),
    State("selection-store",       "data"),
    prevent_initial_call=True,
)
def open_rename_modal(n, session_id, sel_data):
    if not n or not session_id:
        return False, "", ""
    selected = sel_data.get("selected_cluster_ids", []) if sel_data else []
    if len(selected) != 1:
        return False, "", ""
    cid = selected[0]
    clusters = get_clusters(session_id)
    cmap = {c.cluster_id: c for c in clusters}
    c = cmap.get(cid)
    return True, (c.title if c else ""), (c.description if c else "")


@callback(
    Output("modal-rename",         "is_open",  allow_duplicate=True),
    Output("cluster-refresh-store","data",     allow_duplicate=True),
    Input("rename-save",           "n_clicks"),
    Input("rename-cancel",         "n_clicks"),
    State("rename-title-input",    "value"),
    State("rename-desc-input",     "value"),
    State("session-id-store",      "data"),
    State("selection-store",       "data"),
    State("cluster-refresh-store", "data"),
    prevent_initial_call=True,
)
def confirm_rename(save, cancel, title, desc, session_id, sel_data, refresh):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "rename-cancel":
        return False, no_update
    selected = sel_data.get("selected_cluster_ids", []) if sel_data else []
    if not selected or not session_id:
        return False, no_update
    payload = {"cluster_id": selected[0], "title": title or "", "description": desc or ""}
    log_edit(session_id, "rename", payload)
    return False, (refresh or 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
# UNDO
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("cluster-refresh-store","data",     allow_duplicate=True),
    Input("btn-undo",             "n_clicks"),
    State("session-id-store",     "data"),
    State("cluster-refresh-store","data"),
    prevent_initial_call=True,
)
def do_undo(n, session_id, refresh):
    if not n or not session_id:
        return no_update
    undo_last_edit(session_id)
    return (refresh or 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
# RECLUSTER modal open / cancel
# (confirm is handled by phase_controller to avoid duplicate output conflicts)
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("modal-recluster", "is_open", allow_duplicate=True),
    Input("btn-recluster",    "n_clicks"),
    Input("recluster-cancel", "n_clicks"),
    State("modal-recluster",  "is_open"),
    prevent_initial_call=True,
)
def toggle_recluster_modal(open_clicks, cancel_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "btn-recluster":
        return True
    if btn == "recluster-cancel":
        return False
    return no_update


@callback(
    Output("modal-recluster", "is_open", allow_duplicate=True),
    Input("recluster-confirm", "n_clicks"),
    prevent_initial_call=True,
)
def close_recluster_modal_on_confirm(n):
    if n:
        return False
    return no_update
