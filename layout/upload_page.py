"""Upload / home page layout."""

from dash import html, dcc
import dash_bootstrap_components as dbc

from config import AVAILABLE_EMBEDDING_MODELS, EMBEDDING_MODEL


def upload_layout() -> html.Div:
    return html.Div([
        dbc.Container([
            dbc.Row(dbc.Col(
                html.Div([
                    html.H2("Survey Cluster Analysis", className="display-6 fw-bold text-primary"),
                    html.P("Upload a CSV of survey responses, pick your columns, and let the tool"
                           " cluster and label them for you.", className="text-muted"),
                ], className="text-center py-4")
            )),

            # ── Upload card ────────────────────────────────────────────────
            dbc.Row(dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.H5("1 · Upload CSV", className="mb-0")),
                    dbc.CardBody([
                        dcc.Upload(
                            id="csv-upload",
                            children=html.Div([
                                html.I(className="bi bi-cloud-upload fs-1 text-muted"),
                                html.P("Drag & drop your CSV file here, or click to browse",
                                       className="mb-0 text-muted small"),
                            ], className="text-center p-4"),
                            style={
                                "borderWidth": "2px", "borderStyle": "dashed",
                                "borderRadius": "8px", "borderColor": "#adb5bd",
                                "cursor": "pointer",
                            },
                            accept=".csv",
                        ),
                        html.Div(id="upload-feedback", className="mt-2"),
                    ]),
                ], className="shadow-sm mb-3"),
            width={"size": 8, "offset": 2})),

            # ── Column selector ────────────────────────────────────────────
            dbc.Row(dbc.Col(
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader(html.H5("2 · Select Columns", className="mb-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("ID column"),
                                    dbc.Select(id="col-id-select", options=[], value=None),
                                    dbc.FormText("Unique row identifier (e.g. Timestamp, Email)"),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Response column"),
                                    dbc.Select(id="col-response-select", options=[], value=None),
                                    dbc.FormText("Free-text responses to cluster"),
                                ], width=6),
                            ], className="mb-3"),

                            html.H6("Preview (first 5 rows)", className="text-muted small"),
                            html.Div(id="csv-preview-table"),
                        ]),
                    ], className="shadow-sm mb-3"),
                    id="column-collapse", is_open=False,
                ),
            width={"size": 8, "offset": 2})),

            # ── Settings ───────────────────────────────────────────────────
            dbc.Row(dbc.Col(
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader(html.H5("3 · Settings", className="mb-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Session name"),
                                    dbc.Input(id="session-name-input",
                                              placeholder="e.g. Sadhanapada 2025 Q3…",
                                              maxLength=80),
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Embedding model"),
                                    dbc.Select(
                                        id="embedding-model-select",
                                        options=[
                                            {"label": model, "value": model}
                                            for model in AVAILABLE_EMBEDDING_MODELS
                                        ],
                                        value=EMBEDDING_MODEL,
                                        persistence=True,
                                        persistence_type="local",
                                    ),
                                    dbc.FormText("Used for this run's embeddings and clustering cache"),
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("OpenAI API key"),
                                    dbc.Input(id="api-key-input", type="password",
                                              placeholder="sk-… (leave blank to use .env)",
                                              persistence=True, persistence_type="local"),
                                    dbc.FormText("Stored only in your browser's localStorage — leave blank if OPENAI_API_KEY is set in .env"),
                                ], width=4),
                            ]),
                        ]),
                    ], className="shadow-sm mb-4"),
                    id="settings-collapse", is_open=False,
                ),
            width={"size": 8, "offset": 2})),

            # ── Start button ───────────────────────────────────────────────
            dbc.Row(dbc.Col(
                html.Div([
                    dbc.Button(
                        [html.I(className="bi bi-cpu me-2"), "Start Analysis"],
                        id="start-analysis-btn",
                        color="primary",
                        size="lg",
                        disabled=True,
                        className="px-5",
                    ),
                    html.Div(id="start-error-msg", className="mt-2 text-danger small"),
                ], className="text-center mb-5"),
            width={"size": 8, "offset": 2})),

            # ── Past sessions ──────────────────────────────────────────────
            dbc.Row(dbc.Col(
                html.Div(id="past-sessions-panel"),
            width={"size": 8, "offset": 2})),

        ], fluid=False, className="pt-4"),

        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Delete Session")),
            dbc.ModalBody(id="delete-session-modal-body"),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="delete-session-cancel", color="secondary", className="me-2"),
                dbc.Button("Delete", id="delete-session-confirm", color="danger"),
            ]),
        ], id="delete-session-modal", is_open=False),

        # hidden stores / redirect trigger
        dcc.Store(id="upload-data-store"),   # {headers, nrows, csv_b64}
        dcc.Store(id="sessions-refresh-store", data=0),
        dcc.Store(id="delete-session-target-store", data={}),
        dcc.Store(id="delete-session-click-store", data=0),
        dcc.Location(id="redirect-url", refresh=True),
    ])
