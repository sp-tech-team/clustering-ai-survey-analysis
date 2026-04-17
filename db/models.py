"""
SQLAlchemy models.

Phase legend:
  0  CSV uploaded, columns selected — nothing computed yet
  1  Embeddings + UMAP computed (.npy files written)
  2  HDBSCAN run + LLM summaries generated — base cluster state stored
  3  User editing (edit log active)
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, JSON, Text, create_engine, inspect, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError

from config import DB_PATH, EMBEDDING_MODEL

Base = declarative_base()


# ── Core session ──────────────────────────────────────────────────────────────

class AnalysisSession(Base):
    __tablename__ = "sessions"

    session_id   = Column(String, primary_key=True)
    csv_hash     = Column(String, index=True)          # SHA-256 of raw CSV bytes
    id_col       = Column(String)                      # user-selected ID column name
    response_col = Column(String)                      # user-selected response column name
    session_name = Column(String)
    n_points     = Column(Integer)                     # total rows after cleaning
    phase        = Column(Integer, default=0)
    api_key      = Column(String)                      # OpenAI API key (stored for background tasks)
    embedding_model = Column(String, default=EMBEDDING_MODEL)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ── Response points ───────────────────────────────────────────────────────────

class Point(Base):
    __tablename__ = "points"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    session_id    = Column(String, index=True)
    orig_id       = Column(String)       # user's ID column value — passed through as-is
    response_text = Column(Text)
    # active | low_info_structural | low_info_llm | excluded
    status        = Column(String, default="active")


# ── Clusters (base state, written once after HDBSCAN) ─────────────────────────

class Cluster(Base):
    __tablename__ = "clusters"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    cluster_id = Column(Integer)     # HDBSCAN ID; synthetic IDs for splits start at max+1
    title      = Column(String)
    description= Column(Text)
    sentiment  = Column(String)
    theme_name = Column(String)      # user-assigned aggregated theme
    is_active  = Column(Boolean, default=True)
    n_points   = Column(Integer)


# ── Point→cluster assignments (base state) ────────────────────────────────────

class ClusterAssignment(Base):
    __tablename__ = "cluster_assignments"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    point_id   = Column(Integer)     # FK → Point.id
    cluster_id = Column(Integer)     # -1 = HDBSCAN outlier


# ── Mutable edit log — replay this to reconstruct current state ───────────────

class ClusterEdit(Base):
    __tablename__ = "cluster_edits"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    seq        = Column(Integer)     # monotonically increasing — used for replay order
    edit_type  = Column(String)      # join | split | rename | exclude | unexclude | theme
    payload    = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    undone     = Column(Boolean, default=False)


# ── Engine / factory ──────────────────────────────────────────────────────────

def _make_engine():
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
    )
    return engine


def init_db() -> None:
    """Create all tables. Safe to call multiple times."""
    e = _make_engine()
    try:
        Base.metadata.create_all(e)
    except OperationalError as exc:
        # Under multi-worker startup, a second process can race and attempt the
        # same CREATE TABLE right after the first one succeeds.
        if "already exists" not in str(exc):
            raise

    inspector = inspect(e)
    session_columns = {col["name"] for col in inspector.get_columns("sessions")}
    if "embedding_model" not in session_columns:
        default_model = EMBEDDING_MODEL.replace("'", "''")
        try:
            with e.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE sessions "
                        f"ADD COLUMN embedding_model VARCHAR DEFAULT '{default_model}'"
                    )
                )
        except OperationalError as exc:
            if "duplicate column name" not in str(exc):
                raise


engine         = _make_engine()
SessionFactory = sessionmaker(bind=engine)
