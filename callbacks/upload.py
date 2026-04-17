"""Upload page callbacks."""

import base64
import io
import uuid

import pandas as pd
from dash import Input, Output, State, callback, html, no_update, ALL, callback_context
import dash_bootstrap_components as dbc

from config import OPENAI_API_KEY as _ENV_API_KEY, EMBEDDING_MODEL as _DEFAULT_EMBEDDING_MODEL
from db.queries import create_session, bulk_insert_points, list_sessions, delete_session, get_session
from utils import hash_csv, guess_id_col, guess_response_col


# ── Parse uploaded CSV ─────────────────────────────────────────────────────────

@callback(
    Output("upload-data-store",   "data"),
    Output("upload-feedback",     "children"),
    Output("column-collapse",     "is_open"),
    Output("settings-collapse",   "is_open"),
    Input("csv-upload",           "contents"),
    State("csv-upload",           "filename"),
    prevent_initial_call=True,
)
def parse_upload(contents, filename):
    if contents is None:
        return no_update, no_update, no_update, no_update

    _, b64 = contents.split(",", 1)
    csv_bytes = base64.b64decode(b64)

    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as exc:
        return (
            no_update,
            dbc.Alert(f"Could not parse CSV: {exc}", color="danger"),
            False, False,
        )

    headers = list(df.columns)
    store = {
        "csv_b64": b64,
        "headers": headers,
        "nrows": len(df),
        "csv_hash": hash_csv(csv_bytes),
        "id_col_guess": guess_id_col(headers),
        "resp_col_guess": guess_response_col(headers),
    }
    feedback = dbc.Alert(
        [html.I(className="bi bi-check-circle me-2"),
         f"Loaded {len(df):,} rows — {filename}"],
        color="success", className="py-2",
    )
    return store, feedback, True, True


# ── Populate column dropdowns ──────────────────────────────────────────────────

@callback(
    Output("col-id-select",       "options"),
    Output("col-id-select",       "value"),
    Output("col-response-select", "options"),
    Output("col-response-select", "value"),
    Input("upload-data-store",    "data"),
    prevent_initial_call=True,
)
def populate_columns(store):
    if not store:
        return [], None, [], None
    opts = [{"label": h, "value": h} for h in store["headers"]]
    return opts, store["id_col_guess"], opts, store["resp_col_guess"]


# ── Live preview table ─────────────────────────────────────────────────────────

@callback(
    Output("csv-preview-table",     "children"),
    Input("col-id-select",          "value"),
    Input("col-response-select",    "value"),
    State("upload-data-store",      "data"),
    prevent_initial_call=True,
)
def update_preview(id_col, resp_col, store):
    if not store or not id_col or not resp_col:
        return ""
    csv_bytes = base64.b64decode(store["csv_b64"])
    df = pd.read_csv(io.BytesIO(csv_bytes))
    cols = list({id_col, resp_col})
    preview = df[cols].head(5)
    return dbc.Table.from_dataframe(
        preview, striped=True, bordered=True, hover=True, size="sm",
        className="mb-0",
    )


# ── Enable start button ────────────────────────────────────────────────────────

@callback(
    Output("start-analysis-btn",  "disabled"),
    Input("col-id-select",        "value"),
    Input("col-response-select",  "value"),
    Input("embedding-model-select","value"),
    Input("api-key-input",        "value"),
    State("upload-data-store",    "data"),
)
def toggle_start_btn(id_col, resp_col, embedding_model, api_key, store):
    effective_key = api_key or _ENV_API_KEY
    ready = bool(store and id_col and resp_col and id_col != resp_col and embedding_model and effective_key)
    return not ready


# ── Start analysis ──────────────────────────────────────────────────────────────

@callback(
    Output("redirect-url",        "href"),
    Output("start-error-msg",     "children"),
    Input("start-analysis-btn",   "n_clicks"),
    State("upload-data-store",    "data"),
    State("col-id-select",        "value"),
    State("col-response-select",  "value"),
    State("session-name-input",   "value"),
    State("embedding-model-select","value"),
    State("api-key-input",        "value"),
    prevent_initial_call=True,
)
def start_analysis(n_clicks, store, id_col, resp_col, session_name, embedding_model, api_key):
    if not n_clicks or not store:
        return no_update, no_update

    csv_bytes = base64.b64decode(store["csv_b64"])
    df = pd.read_csv(io.BytesIO(csv_bytes))

    session_id = str(uuid.uuid4())
    name = session_name or f"Session {session_id[:8]}"
    effective_key = api_key or _ENV_API_KEY
    effective_model = embedding_model or _DEFAULT_EMBEDDING_MODEL

    try:
        create_session(
            session_id=session_id,
            csv_hash=store["csv_hash"],
            id_col=id_col,
            response_col=resp_col,
            session_name=name,
            n_points=len(df),
            api_key=effective_key,
            embedding_model=effective_model,
        )
        rows = list(zip(df[id_col].astype(str), df[resp_col].astype(str)))
        bulk_insert_points(session_id, rows)
    except Exception as exc:
        return no_update, f"Error creating session: {exc}"

    return f"/analysis/{session_id}", ""


# ── Past sessions panel ─────────────────────────────────────────────────────────

@callback(
    Output("past-sessions-panel", "children"),
    Input("upload-data-store",    "data"),  # refresh after upload
    Input("sessions-refresh-store", "data"),
)
def show_past_sessions(_store, _refresh):
    sessions = list_sessions(limit=10)
    if not sessions:
        return ""

    rows = []
    for s in sessions:
        rows.append(html.Tr([
            html.Td(html.A(s.session_name, href=f"/analysis/{s.session_id}")),
            html.Td(f"{s.n_points:,} rows"),
            html.Td(s.embedding_model or _DEFAULT_EMBEDDING_MODEL),
            html.Td(_phase_label(s.phase)),
            html.Td(s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else ""),
            html.Td(
                dbc.Button(
                    [html.I(className="bi bi-trash me-1"), "Delete"],
                    id={"type": "delete-session-btn", "index": s.session_id},
                    color="outline-danger",
                    size="sm",
                ),
                className="text-end",
            ),
        ]))

    return dbc.Card([
        dbc.CardHeader(html.H6("Recent sessions", className="mb-0")),
        dbc.CardBody(dbc.Table(
            [html.Thead(html.Tr([html.Th("Name"), html.Th("Size"),
                                 html.Th("Embedding"), html.Th("Phase"), html.Th("Created"), html.Th("")])),
             html.Tbody(rows)],
            striped=True, hover=True, size="sm", className="mb-0",
        )),
    ], className="shadow-sm")


@callback(
    Output("delete-session-modal", "is_open"),
    Output("delete-session-modal-body", "children"),
    Output("delete-session-target-store", "data"),
    Output("delete-session-click-store", "data"),
    Input({"type": "delete-session-btn", "index": ALL}, "n_clicks_timestamp"),
    Input("delete-session-cancel", "n_clicks"),
    State({"type": "delete-session-btn", "index": ALL}, "id"),
    State("delete-session-click-store", "data"),
    prevent_initial_call=True,
)
def toggle_delete_session_modal(delete_click_timestamps, cancel_clicks, delete_ids, last_handled_ts):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update

    triggered = ctx.triggered_id
    if triggered == "delete-session-cancel":
        return False, "", {}, no_update

    if not isinstance(triggered, dict) or triggered.get("type") != "delete-session-btn":
        return no_update, no_update, no_update, no_update

    session_id = triggered.get("index")
    if not session_id or not delete_ids or not delete_click_timestamps:
        return no_update, no_update, no_update, no_update

    timestamp_by_session = {
        id_obj.get("index"): ts
        for id_obj, ts in zip(delete_ids, delete_click_timestamps)
        if isinstance(id_obj, dict)
    }
    click_ts = timestamp_by_session.get(session_id)
    if click_ts in (None, -1) or click_ts <= (last_handled_ts or 0):
        return no_update, no_update, no_update, no_update

    session = get_session(session_id) if session_id else None
    if not session:
        return False, "Session not found.", {}, click_ts

    body = html.Div([
        html.P(f'Delete session "{session.session_name}"?'),
        html.P(
            "This removes the session, points, assignments, clusters, and edit history. This cannot be undone.",
            className="text-muted small mb-0",
        ),
    ])
    return True, body, {"session_id": session_id}, click_ts


@callback(
    Output("delete-session-modal", "is_open", allow_duplicate=True),
    Output("sessions-refresh-store", "data"),
    Input("delete-session-confirm", "n_clicks"),
    State("delete-session-target-store", "data"),
    State("sessions-refresh-store", "data"),
    prevent_initial_call=True,
)
def confirm_delete_session(n_clicks, target, refresh_count):
    if not n_clicks:
        return no_update, no_update

    session_id = (target or {}).get("session_id")
    if not session_id:
        return False, no_update

    delete_session(session_id)
    return False, (refresh_count or 0) + 1


def _phase_label(phase: int) -> str:
    return {0: "Uploaded", 1: "Embedded", 2: "Clustered", 3: "Editing"}.get(phase, "?")
