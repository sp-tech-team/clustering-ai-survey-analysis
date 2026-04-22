"""Reusable Dash UI components."""

from dash import html
import dash_bootstrap_components as dbc

from utils import sentiment_color, SENTIMENT_COLORS


# ── Phase badge ───────────────────────────────────────────────────────────────

_PHASE_LABELS = {
    0: ("Uploaded",   "secondary"),
    1: ("Embedded",   "info"),
    2: ("Clustered",  "primary"),
    3: ("Editing",    "success"),
}


def phase_badge(phase: int) -> dbc.Badge:
    label, color = _PHASE_LABELS.get(phase, ("Unknown", "dark"))
    return dbc.Badge(label, color=color, className="ms-2")


# ── Sentiment dot ──────────────────────────────────────────────────────────────

def sentiment_dot(sentiment: str) -> html.Span:
    color = sentiment_color(sentiment)
    return html.Span(
        "●",
        style={"color": color, "fontSize": "0.75rem", "marginRight": "4px"},
        title=sentiment,
    )


# ── Cluster list item ─────────────────────────────────────────────────────────

def cluster_list_item(cluster_id: int, title: str, n_points: int,
                      pct: float, theme_name: str | None = None,
                      description: str | None = None) -> dbc.ListGroupItem:
    count_badge = dbc.Badge(f"{n_points} ({pct}%)", color="light", text_color="dark",
                            className="ms-auto me-1", style={"fontSize": "0.65rem"})

    title_id = {"type": "cluster-title", "index": cluster_id}
    tooltip = dbc.Tooltip(
        description or "No summary available.",
        target=title_id,
        placement="right",
        style={"maxWidth": "320px"},
    ) if True else None

    return dbc.ListGroupItem(
        [
            dbc.Checkbox(
                id={"type": "cluster-check", "index": cluster_id},
                className="me-2",
                style={"display": "inline-block"},
            ),
            html.Span(title, id=title_id, className="small",
                      style={"cursor": "help", "borderBottom": "1px dotted #999"}),
            count_badge,
            tooltip,
        ],
        id={"type": "cluster-item", "index": cluster_id},
        action=True,
        style={"padding": "6px 10px", "cursor": "default"},
    )


# ── Progress card ──────────────────────────────────────────────────────────────

def progress_card(message: str = "Processing…", value: int = 0,
                  steps: list | None = None) -> dbc.Card:
    step_items = []
    for s in (steps or []):
        _s       = s if isinstance(s, dict) else vars(s)
        s_status = _s.get("status", "pending")
        s_name   = _s.get("name",   "")
        s_detail = _s.get("detail", "")
        icon  = {"done": "✓", "running": "⌛", "error": "✗", "pending": "○"}.get(s_status, "○")
        color = {"done": "text-success", "running": "text-primary",
                 "error": "text-danger", "pending": "text-muted"}.get(s_status, "text-muted")
        step_items.append(
            html.Li(
                [
                    html.Span(icon, className=f"{color} me-2"),
                    html.Span(s_name),
                    html.Span(f" — {s_detail}", className="text-muted ms-1") if s_detail else "",
                ],
                className="small",
            )
        )

    return dbc.Card(
        dbc.CardBody([
            html.H5("Processing…", className="card-title"),
            dbc.Progress(value=value, striped=True, animated=value < 100,
                         color="primary", className="mb-3"),
            html.P(message, className="text-muted small mb-2"),
            html.Ul(step_items, className="list-unstyled mb-0") if step_items else "",
        ]),
        className="shadow-sm",
    )


# ── Action buttons row ────────────────────────────────────────────────────────

def action_buttons(n_selected: int = 0) -> html.Div:
    disabled_join  = n_selected < 2
    disabled_other = n_selected != 1

    return html.Div([
        dbc.Button("Join",        id="btn-join",    color="primary",   size="sm",
                   disabled=disabled_join,  className="me-1 mb-1"),
        dbc.Button("Split",       id="btn-split",   color="warning",   size="sm",
                   disabled=disabled_other, className="me-1 mb-1"),
        dbc.Button("Low-Info",    id="btn-low-info", color="secondary", size="sm",
                   disabled=disabled_other, className="me-1 mb-1"),
        dbc.Button("Rename",      id="btn-rename",  color="light",     size="sm",
                   disabled=disabled_other, className="me-1 mb-1"),
    ], className="d-flex flex-wrap gap-1")


# ── Modals ────────────────────────────────────────────────────────────────────

def rename_modal() -> dbc.Modal:
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Rename Cluster")),
        dbc.ModalBody([
            dbc.Label("Title"),
            dbc.Input(id="rename-title-input", placeholder="Cluster title…", maxLength=80),
            dbc.Label("Description (optional)", className="mt-2"),
            dbc.Textarea(id="rename-desc-input", placeholder="2-3 sentence description…", rows=3),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="rename-cancel", color="secondary", className="me-2"),
            dbc.Button("Save",   id="rename-save",   color="primary"),
        ]),
    ], id="modal-rename", is_open=False)


def join_confirm_modal() -> dbc.Modal:
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Join Clusters")),
        dbc.ModalBody([
            html.P(id="join-preview-text", className="text-muted small"),
            dbc.Label("New title (LLM suggestion — edit freely)"),
            dbc.Input(id="join-title-input", maxLength=80),
            dbc.Label("Description", className="mt-2"),
            dbc.Textarea(id="join-desc-input", rows=3),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel",  id="join-cancel",  color="secondary", className="me-2"),
            dbc.Button("Confirm", id="join-confirm", color="primary"),
        ]),
    ], id="modal-join", is_open=False, size="lg")


def export_confirm_modal() -> dbc.Modal:
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Export Clusters")),
        dbc.ModalBody([
            html.P(
                "Include secondary cluster assignments?",
                className="mb-1",
            ),
            html.P(
                "Each response will have additional columns listing any other clusters "
                "it qualifies for based on soft membership scores. "
                "This only affects the exported file — the cluster view is unchanged.",
                className="text-muted small",
            ),
            dbc.Checklist(
                options=[{"label": "Include secondary cluster columns", "value": "yes"}],
                value=[],
                id="export-secondary-check",
                switch=True,
                className="mt-2",
            ),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel",   id="export-cancel",  color="secondary", className="me-2"),
            dbc.Button(
                "Download",
                id="export-confirm",
                color="success",
                href="#",
                target="_blank",
                external_link=True,
            ),
        ]),
    ], id="modal-export", is_open=False)


def recluster_confirm_modal() -> dbc.Modal:
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Re-cluster")),
        dbc.ModalBody([
            html.P(
                "Clusters currently marked Low-Info will be permanently excluded, "
                "and clustering will re-run on the remaining responses.",
            ),
            dbc.Alert(
                [
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    "All joins, splits, renames, and theme assignments will be discarded. "
                    "This cannot be undone.",
                ],
                color="warning", className="small mb-0",
            ),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel",    id="recluster-cancel",  color="secondary", className="me-2"),
            dbc.Button("Re-cluster", id="recluster-confirm", color="danger"),
        ]),
    ], id="modal-recluster", is_open=False)
