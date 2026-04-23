---
title: SurveyAi
emoji: "📊"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Survey Cluster Analysis

An interactive Dash app for clustering open-ended survey responses with OpenAI embeddings, UMAP, HDBSCAN, and an editable review workflow.

The app lets you upload a CSV, choose the ID and response columns, generate embeddings, cluster responses, inspect results in a 3D plot, refine the cluster structure, and export a final workbook.

## At a Glance

```mermaid
flowchart LR
	A[Upload CSV] --> B[Choose ID and Response Columns]
	B --> C[Low-info Filtering]
	C --> D[OpenAI Embeddings]
	D --> E[UMAP Projections]
	E --> F[HDBSCAN Clustering]
	F --> G[Optional Auto-Merge Suggestions]
	G --> H[LLM Cluster Titles and Descriptions]
	H --> I[Interactive Review and Edits]
	I --> J[Export-Time Centroid Assignment]
	J --> K[Excel Export]
```

## Current Flow Summary

The current clustering model is intentionally simple:

- Phase 1 filters low-information responses, generates embeddings, and builds cached UMAP projections.
- Phase 2 runs HDBSCAN on the active responses.
- The editable working state is the raw HDBSCAN result after any auto-merges. There is no soft-membership reassignment layer in the live cluster state.
- User actions such as join, split, rename, exclude, theme assignment, undo, and re-cluster are stored as edits and replayed into the current state.
- Export runs one separate centroid-based refinement pass over the final edited active clusters using the original cached embeddings.
- That export-only pass can re-check raw residual responses still sitting in `Other Themes` and add secondary themes, but it does not change the interactive cluster view.
- Clusters manually moved to `Other Themes` are treated as a final user override and are not reconsidered during export.

## App Flow

```mermaid
flowchart TD
	subgraph UI
		U1[Upload Page]
		U2[Analysis Page]
	end

	subgraph Core
		C1[Embedder]
		C2[UMAP Runner]
		C3[Clusterer]
		C4[LLM Helpers]
		C5[Cache]
		C6[State Reconstruction]
	end

	subgraph Persistence
		P1[SQLite Session DB]
		P2[Numpy Cache Files]
	end

	U1 --> P1
	U1 --> C5
	U2 --> C6
	C1 --> P2
	C2 --> P2
	C3 --> P1
	C4 --> P1
	C5 --> P2
	P1 --> U2
	P2 --> U2
```

## What The Project Does

- Uploads a CSV of survey responses and lets you select the identifier and free-text columns.
- Filters low-information responses before expensive embedding and clustering steps.
- Embeds active responses with OpenAI embedding models.
- Builds both high-dimensional and 3D UMAP projections.
- Runs HDBSCAN for cluster discovery.
- Keeps merged HDBSCAN labels as the editable working cluster state.
- Uses LLM summaries to name and describe clusters.
- Supports post-cluster editing: join, split, rename, exclude, undo, and re-cluster.
- Exports results to Excel with an export-only centroid pass for raw residual `Other Themes` responses and secondary theme assignment.

## Project Map

```text
.
├── app.py                  # Dash app entrypoint and page routing
├── config.py               # Environment variables and clustering settings
├── tasks.py                # Background task registry and progress tracking
├── callbacks/              # Upload, export, and phase control callbacks
├── core/                   # Embedding, clustering, UMAP, LLM, cache, state logic
├── db/                     # SQLite models and query helpers
├── layout/                 # Upload page, analysis page, and reusable UI components
├── cache/                  # Generated numpy artifacts for embeddings and UMAP
└── assets/                 # Static frontend assets
```

## Key Files

- `app.py`: initializes Dash, database setup, layout routing, and callback registration.
- `config.py`: loads `.env`, defines model choices, cache/database paths, and clustering/export tuning.
- `callbacks/phase_controller.py`: orchestrates phase progression and builds the live working cluster state from HDBSCAN plus auto-merges.
- `callbacks/export.py`: builds the export workbook and runs the export-only centroid assignment pass.
- `core/clusterer.py`: HDBSCAN clustering, representative extraction, and centroid threshold helpers.
- `core/export_centroid.py`: export-only centroid refinement for raw residual `Other Themes` responses and secondary theme assignment.
- `core/cache.py`: persists and reloads embeddings and UMAP arrays under `cache/`.
- `db/models.py`: SQLite schema for sessions, points, clusters, assignments, and edit history.

## Installation With uv

### 1. Install uv

If you do not already have `uv`, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On macOS with Homebrew, this also works:

```bash
brew install uv
```

### 2. Ensure Python 3.13 is available

This project declares `requires-python = ">=3.13"`.

```bash
uv python install 3.13
```

### 3. Sync dependencies

From the project root:

```bash
uv sync
```

This creates the project virtual environment and installs the dependencies from `pyproject.toml`.

### 4. Configure environment variables

Create a `.env` file in the project root:

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-large
EOF
```

Notes:

- `OPENAI_API_KEY` is required unless you provide a key in the app UI.
- `EMBEDDING_MODEL` is optional. Valid values are `text-embedding-3-small` and `text-embedding-3-large`.

### 5. Run the app

```bash
uv run python app.py
```

Then open:

```text
http://127.0.0.1:8050
```

## Typical Development Workflow

```bash
uv sync
uv run python app.py
```

If you change dependencies:

```bash
uv add <package>
uv sync
```

## Running Tests

The project uses the Python standard library `unittest` test runner, so no extra test dependency is required.

Run the full core test suite from the project root:

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

Run a single test module:

```bash
uv run python -m unittest tests.test_core_clusterer
```

Run one specific test class:

```bash
uv run python -m unittest tests.test_core_splitter.CoreSplitterTests
```

## Runtime Artifacts

The app creates local runtime data while you work:

- `cache/`: embeddings and UMAP projections.
- `db/survey_clusters.db`: session metadata, points, cluster assignments, and edit history.

These files are local artifacts and should not be committed.

## User Journey

```mermaid
sequenceDiagram
	participant User
	participant DashApp
	participant Cache
	participant DB
	participant OpenAI

	User->>DashApp: Upload CSV and choose columns
	DashApp->>DB: Create session and store points
	User->>DashApp: Start analysis
	DashApp->>OpenAI: Filter and embed responses
	DashApp->>Cache: Save embeddings and UMAP arrays
	DashApp->>DashApp: Run HDBSCAN and optional auto-merges
	DashApp->>DB: Save working clusters and assignments
	User->>DashApp: Review, rename, split, join, exclude
	DashApp->>DB: Persist edit log
	User->>DashApp: Export results
	DashApp->>DashApp: Run centroid refinement for export only on raw residual Other Themes responses
```

## Notes

- The UI is built with Dash and Dash Bootstrap Components.
- The app uses background tasks and interval polling so long-running phases do not block the interface.
- Export produces an Excel workbook with a response sheet and a cluster summary sheet.
- The interactive UI and the export workbook intentionally use different assignment layers: live editing uses HDBSCAN plus edits, while export adds a final centroid-based refinement pass.
- Manual `Other Themes` assignment is terminal until the user changes it again. Export refinement only revisits raw residual responses that still sit in `Other Themes`.
