"""Dash application entry point.

Run with:
    cd dash_app
    python app.py
"""

import sys
from io import BytesIO
from pathlib import Path

# Ensure the dash_app directory is on sys.path regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State
from flask import abort, request, send_file

from db.models import init_db
from layout.upload_page import upload_layout
from layout.analysis_page import analysis_layout
from config import POLL_INTERVAL_MS

# ── Initialise database ────────────────────────────────────────────────────────
init_db()

# ── Create app ─────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        dbc.icons.BOOTSTRAP,
    ],
    suppress_callback_exceptions=True,   # dynamic layouts need this
    title="Survey Cluster Analysis",
)
server = app.server  # expose Flask server for gunicorn / deployment

# ── Root layout (persistent across pages) ─────────────────────────────────────
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),

    # Persists session_id across browser refreshes
    dcc.Store(id="session-store", storage_type="local", data={}),

    # Controls the polling interval for background tasks
    dcc.Interval(
        id="progress-interval",
        interval=POLL_INTERVAL_MS,
        n_intervals=0,
        disabled=False,   # individual pages manage enabling/disabling
    ),

    # Dynamic page content
    html.Div(id="page-content"),
])


# ── Page routing ───────────────────────────────────────────────────────────────

@app.callback(
    Output("page-content", "children"),
    Input("url",           "pathname"),
)
def display_page(pathname: str):
    if pathname and pathname.startswith("/analysis/"):
        session_id = pathname.split("/analysis/", 1)[1]
        if session_id:
            return analysis_layout(session_id)
    return upload_layout()


from callbacks.export import build_export_workbook  # noqa: E402


@server.get("/api/export/<session_id>")
def export_session_file(session_id: str):
    try:
        workbook_bytes, filename = build_export_workbook(session_id)
    except ValueError:
        abort(404)

    return send_file(
        BytesIO(workbook_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename,
        max_age=0,
    )


# ── Register all callbacks (must come AFTER app is defined) ───────────────────
import callbacks  # noqa: F401, E402  — side-effect: registers @callback decorators


# ── Dev server ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
