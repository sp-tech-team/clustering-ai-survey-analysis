"""Main analysis page layout (Phase 1–3)."""

from dash import html, dcc
import dash_bootstrap_components as dbc

from layout.components import (
    phase_badge, action_buttons, progress_card,
    rename_modal, join_confirm_modal, export_confirm_modal,
    recluster_confirm_modal,
)


def analysis_layout(session_id: str) -> html.Div:
    return html.Div([
        # ── Top navbar ─────────────────────────────────────────────────────────
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand([
                    html.I(className="bi bi-diagram-3 me-2"),
                    "Survey Cluster Analysis",
                ], href="/"),
                dbc.Nav([
                    dbc.NavItem(html.Span(id="session-name-display", className="navbar-text me-2 small")),
                    dbc.NavItem(html.Div(id="phase-badge-display")),
                    dbc.NavItem(dbc.Button(
                        [html.I(className="bi bi-arrow-repeat me-1"), "Re-cluster"],
                        id="btn-recluster", color="warning", size="sm",
                        className="ms-3",
                    ), id="recluster-btn-container", style={"display": "none"}),
                ], className="ms-auto"),
            ], fluid=True),
            color="primary", dark=True, className="mb-0",
        ),

        # ── Phase 1 progress overlay ───────────────────────────────────────────
        dbc.Collapse(
            dbc.Container(
                dbc.Row(dbc.Col(
                    html.Div(id="phase1-progress-container"),
                    width={"size": 6, "offset": 3},
                )),
                className="mt-4",
            ),
            id="phase1-overlay", is_open=False,
        ),

        # ── Phase 2 trigger ────────────────────────────────────────────────────
        dbc.Collapse(
            dbc.Container(dbc.Row(dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.H5("Embeddings ready — run cluster analysis now", className="card-title"),
                    html.P("This will run HDBSCAN and use the LLM to label each cluster.",
                           className="text-muted small"),
                    dbc.Button(
                        [html.I(className="bi bi-diagram-3 me-2"), "Run Clustering"],
                        id="run-clustering-btn", color="success", size="lg",
                    ),
                    html.Div(id="phase2-progress-container", className="mt-3"),
                ]), className="text-center shadow-sm"), width={"size": 6, "offset": 3}
            )), className="mt-4"),
            id="phase2-trigger-panel", is_open=False,
        ),

        # ── Main editing view ──────────────────────────────────────────────────
        dbc.Collapse(
            dbc.Container([
                dbc.Row([
                    # ── Sidebar ────────────────────────────────────────────────
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.Span(id="cluster-count-badge", className="fw-bold"),
                                html.Span(" clusters", className="text-muted small ms-1"),
                            ]),
                            dbc.CardBody([
                                # Action buttons
                                html.Div(id="action-buttons-container",
                                         children=action_buttons(0)),
                                html.Hr(className="my-2"),

                                # Cluster list (scrollable)
                                html.Div(
                                    id="cluster-list-container",
                                    style={
                                        "maxHeight": "calc(100vh - 340px)",
                                        "overflowY": "auto",
                                    },
                                ),

                                html.Hr(className="my-2"),
                                dbc.Row([
                                    dbc.Col(dbc.Button(
                                        [html.I(className="bi bi-arrow-counterclockwise me-1"), "Undo"],
                                        id="btn-undo", color="outline-secondary", size="sm",
                                        className="w-100",
                                    ), width=6),
                                    dbc.Col(dbc.Button(
                                        [html.I(className="bi bi-download me-1"), "Export"],
                                        id="btn-export", color="outline-success", size="sm",
                                        className="w-100",
                                    ), width=6),
                                ]),
                            ]),
                        ], className="shadow-sm h-100"),
                    ], width=3),

                    # ── Main plot area ─────────────────────────────────────────
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(
                                    id="scatter-3d",
                                    style={"height": "calc(100vh - 260px)"},
                                    config={
                                        "displayModeBar": True,
                                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                    },
                                ),
                            ], className="p-1"),
                        ], className="shadow-sm"),

                        # Point detail panel
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(html.Small(id="point-id-display",
                                                       className="text-muted"), width=3),
                                    dbc.Col(html.Small(id="point-cluster-display",
                                                       className="text-muted"), width=9),
                                ]),
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Button(
                                            [html.I(className="bi bi-arrow-left")],
                                            id="point-nav-left",
                                            color="outline-secondary",
                                            size="sm",
                                            className="me-2",
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            [html.I(className="bi bi-arrow-right")],
                                            id="point-nav-right",
                                            color="outline-secondary",
                                            size="sm",
                                        ),
                                        width="auto",
                                    ),
                                ], className="g-2 mt-2 mb-1"),
                                html.P(id="point-response-display",
                                       className="mb-0 mt-1 small"),
                            ]), className="shadow-sm mt-2"),
                            id="point-detail-collapse", is_open=False,
                        ),
                    ], width=9),
                ], className="mt-3 mb-3", style={"minHeight": "calc(100vh - 110px)"}),
            ], fluid=True),
            id="main-edit-panel", is_open=False,
        ),

        # ── Re-cluster progress (shown during re-clustering) ─────────────────
        dbc.Collapse(
            dbc.Container(dbc.Row(dbc.Col(
                html.Div(id="recluster-progress-container"),
                width={"size": 6, "offset": 3},
            )), className="mt-4"),
            id="recluster-progress-collapse", is_open=False,
        ),

        # ── Modals ─────────────────────────────────────────────────────────────
        rename_modal(),
        join_confirm_modal(),
        export_confirm_modal(),
        recluster_confirm_modal(),

        # ── Notification toast ─────────────────────────────────────────────────
        dbc.Toast(
            id="notification-toast",
            header="",
            is_open=False,
            dismissable=True,
            icon="success",
            style={"position": "fixed", "bottom": 20, "right": 20, "width": 350, "zIndex": 9999},
        ),

        # ── Hidden stores ──────────────────────────────────────────────────────
        dcc.Store(id="session-id-store",       data=session_id),
        dcc.Store(id="selection-store",        data={"selected_cluster_ids": [], "clicked_point_idx": None}),
        dcc.Store(id="cluster-refresh-store",  data=0),      # increment to force list/plot re-render
        dcc.Store(id="phase-store",            data={"phase": 0}),
        dcc.Store(id="task-store",             data={"status": "idle", "task_type": None}),
        dcc.Store(id="join-working-store",     data={}),      # holds LLM draft during join modal
    ])
