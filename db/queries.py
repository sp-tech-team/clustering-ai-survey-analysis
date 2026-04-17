"""
CRUD helpers — all DB access goes through these functions.
Each function opens its own session so callers stay stateless.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Tuple

from config import EMBEDDING_MODEL
from db.models import (
    SessionFactory, AnalysisSession, Point, Cluster,
    ClusterAssignment, ClusterEdit,
)


# ── Context manager ───────────────────────────────────────────────────────────

@contextmanager
def db_session():
    s = SessionFactory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ── AnalysisSession ───────────────────────────────────────────────────────────

def get_session(session_id: str) -> Optional[AnalysisSession]:
    with db_session() as s:
        row = s.query(AnalysisSession).filter_by(session_id=session_id).first()
        if row:
            s.expunge(row)
        return row


def find_existing_session(
    csv_hash: str, id_col: str, response_col: str, embedding_model: str = EMBEDDING_MODEL
) -> Optional[AnalysisSession]:
    """Return the most-recent session for the same file + column selection."""
    with db_session() as s:
        row = (
            s.query(AnalysisSession)
             .filter_by(
                 csv_hash=csv_hash,
                 id_col=id_col,
                 response_col=response_col,
                 embedding_model=embedding_model,
             )
             .order_by(AnalysisSession.created_at.desc())
             .first()
        )
        if row:
            s.expunge(row)
        return row


def create_session(
    session_id: str, csv_hash: str, id_col: str, response_col: str,
    session_name: str, n_points: int, api_key: str = "", embedding_model: str = EMBEDDING_MODEL,
) -> None:
    with db_session() as s:
        s.add(AnalysisSession(
            session_id=session_id, csv_hash=csv_hash,
            id_col=id_col, response_col=response_col,
            session_name=session_name, n_points=n_points,
            phase=0, api_key=api_key, embedding_model=embedding_model,
        ))


def delete_session(session_id: str) -> None:
    with db_session() as s:
        s.query(ClusterEdit).filter_by(session_id=session_id).delete(synchronize_session=False)
        s.query(ClusterAssignment).filter_by(session_id=session_id).delete(synchronize_session=False)
        s.query(Cluster).filter_by(session_id=session_id).delete(synchronize_session=False)
        s.query(Point).filter_by(session_id=session_id).delete(synchronize_session=False)
        s.query(AnalysisSession).filter_by(session_id=session_id).delete(synchronize_session=False)


def advance_phase(session_id: str, phase: int) -> None:
    with db_session() as s:
        row = s.query(AnalysisSession).filter_by(session_id=session_id).first()
        if row:
            row.phase = phase
            row.updated_at = datetime.utcnow()


def list_sessions(limit: int = 20):
    """Return sessions ordered newest first."""
    with db_session() as s:
        rows = (
            s.query(AnalysisSession)
             .order_by(AnalysisSession.created_at.desc())
             .limit(limit)
             .all()
        )
        for r in rows:
            s.expunge(r)
        return rows


# ── Points ────────────────────────────────────────────────────────────────────

def bulk_insert_points(session_id: str, rows: List[Tuple[str, str]]) -> None:
    """rows: list of (orig_id, response_text)."""
    with db_session() as s:
        for orig_id, text in rows:
            s.add(Point(
                session_id=session_id,
                orig_id=str(orig_id),
                response_text=text,
                status="active",
            ))


def get_points(session_id: str) -> List[Point]:
    with db_session() as s:
        rows = s.query(Point).filter_by(session_id=session_id).order_by(Point.id).all()
        for r in rows:
            s.expunge(r)
        return rows


def mark_points_status(session_id: str, point_ids: List[int], status: str) -> None:
    with db_session() as s:
        s.query(Point).filter(
            Point.session_id == session_id,
            Point.id.in_(point_ids),
        ).update({"status": status}, synchronize_session=False)


# ── Clusters ──────────────────────────────────────────────────────────────────

def save_clusters(session_id: str, cluster_list: list) -> None:
    """cluster_list: list of dicts with cluster_id, title, description, sentiment, n_points."""
    with db_session() as s:
        for c in cluster_list:
            s.add(Cluster(
                session_id=session_id,
                cluster_id=c["cluster_id"],
                title=c["title"],
                description=c.get("description", ""),
                sentiment=c.get("sentiment", "unknown"),
                n_points=c["n_points"],
                theme_name=None,
                is_active=True,
            ))


def get_clusters(session_id: str) -> List[Cluster]:
    with db_session() as s:
        rows = s.query(Cluster).filter_by(session_id=session_id).all()
        for r in rows:
            s.expunge(r)
        return rows


# ── ClusterAssignments ────────────────────────────────────────────────────────

def save_cluster_assignments(session_id: str, assignments: List[Tuple[int, int]]) -> None:
    """assignments: list of (point_id, cluster_id)."""
    with db_session() as s:
        for point_id, cluster_id in assignments:
            s.add(ClusterAssignment(
                session_id=session_id,
                point_id=point_id,
                cluster_id=cluster_id,
            ))


def get_cluster_assignments(session_id: str) -> List[ClusterAssignment]:
    with db_session() as s:
        rows = (s.query(ClusterAssignment)
                  .filter_by(session_id=session_id)
                  .order_by(ClusterAssignment.point_id)
                  .all())
        for r in rows:
            s.expunge(r)
        return rows


# ── Edit log ──────────────────────────────────────────────────────────────────

def log_edit(session_id: str, edit_type: str, payload: dict) -> None:
    with db_session() as s:
        seq = s.query(ClusterEdit).filter_by(session_id=session_id).count() + 1
        s.add(ClusterEdit(
            session_id=session_id,
            seq=seq,
            edit_type=edit_type,
            payload=payload,
        ))


def get_all_edits(session_id: str) -> List[ClusterEdit]:
    with db_session() as s:
        rows = (
            s.query(ClusterEdit)
             .filter_by(session_id=session_id, undone=False)
             .order_by(ClusterEdit.seq)
             .all()
        )
        for r in rows:
            s.expunge(r)
        return rows


def undo_last_edit(session_id: str) -> Optional[str]:
    """Mark the most-recent non-undone edit as undone. Returns edit_type or None."""
    with db_session() as s:
        last = (
            s.query(ClusterEdit)
             .filter_by(session_id=session_id, undone=False)
             .order_by(ClusterEdit.seq.desc())
             .first()
        )
        if last:
            last.undone = True
            return last.edit_type
    return None


def count_edits(session_id: str) -> int:
    with db_session() as s:
        return s.query(ClusterEdit).filter_by(session_id=session_id, undone=False).count()


def wipe_cluster_state(session_id: str) -> None:
    """Delete all clusters, assignments, and edits for a session (used before re-clustering)."""
    with db_session() as s:
        s.query(ClusterEdit).filter_by(session_id=session_id).delete(synchronize_session=False)
        s.query(ClusterAssignment).filter_by(session_id=session_id).delete(synchronize_session=False)
        s.query(Cluster).filter_by(session_id=session_id).delete(synchronize_session=False)
