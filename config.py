import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
DB_DIR    = BASE_DIR / "db"

CACHE_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

DB_PATH = DB_DIR / "survey_clusters.db"

AVAILABLE_EMBEDDING_MODELS = (
	"text-embedding-3-small",
	"text-embedding-3-large",
)


def _get_embedding_model() -> str:
	model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large").strip()
	if model not in AVAILABLE_EMBEDDING_MODELS:
		raise ValueError(
			"EMBEDDING_MODEL must be one of: " + ", ".join(AVAILABLE_EMBEDDING_MODELS)
		)
	return model

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL      = _get_embedding_model()
LLM_MODEL            = "gpt-5.4-mini"   # used for low-info filtering
LLM_MODEL_SMART      = "gpt-5.4"        # used for merge, summarise, join
EMBEDDING_BATCH_SIZE = 100

# ── Structural low-info filter ────────────────────────────────────────────────
MIN_WORD_COUNT = 1
MIN_CHAR_COUNT = 4

# ── LLM low-info filter ───────────────────────────────────────────────────────
LLM_FILTER_CHAR_THRESHOLD  = 75   # fallback if dynamic threshold is unstable
LLM_FILTER_CHAR_PERCENTILE = 30
LLM_FILTER_CHAR_MIN        = 40
LLM_FILTER_CHAR_MAX        = 120
LLM_FILTER_MIN_SAMPLES     = 20
LLM_FILTER_BATCH_SIZE      = 75

# ── UMAP ─────────────────────────────────────────────────────────────────────
UMAP_N_COMPONENTS_HIGH = 100     # for clustering
UMAP_N_COMPONENTS_VIS  = 3       # for 3-D visualisation
UMAP_N_NEIGHBORS       = 15
UMAP_MIN_DIST          = 0.0
UMAP_N_EPOCHS          = 200

# ── HDBSCAN ───────────────────────────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES      = 5

# ── Representatives ───────────────────────────────────────────────────────────
N_REPRESENTATIVES = 5
N_OUTLIER_SAMPLE  = 2
# ── Soft / secondary cluster membership ────────────────────────────────────────
# Percentile of a cluster's primary members' soft scores used as the assignment
# threshold for that cluster (higher = stricter).
SECONDARY_MEMBERSHIP_PERCENTILE = 5
# Absolute floor so noise clusters don't get thresholds near 0.
SECONDARY_MEMBERSHIP_FLOOR      = 0.05
# ── Centroid cosine secondary assignment ─────────────────────────────────────
# Percentile of primary members' cosine similarity to cluster centroid used as
# assignment threshold. Higher = stricter (fewer secondary assignments). 75 is
# conservative — only points well within the cluster's similarity range qualify.
SECONDARY_CENTROID_PERCENTILE   = 75
# Hard cap: each point gets at most this many secondary cluster assignments
# (i.e. a point belongs to primary + at most MAX_SECONDARY_CLUSTERS others).
MAX_SECONDARY_CLUSTERS          = 3
# ── Dash ──────────────────────────────────────────────────────────────────────
POLL_INTERVAL_MS = 2000    # background-task polling interval
