import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from callbacks.export import build_export_preview, build_export_workbook, update_export_href

class ExportCallbackTests(unittest.TestCase):
    def test_update_export_href_returns_fixed_export_url(self):
        self.assertEqual(update_export_href(None), "#")
        self.assertEqual(update_export_href("session-1"), "/api/export/session-1")

    def test_build_export_workbook_uses_export_centroid_secondaries(self):
        session = SimpleNamespace(
            session_id="session-1",
            session_name="Demo",
            id_col="id",
            response_col="response",
            n_points=3,
            csv_hash="hash123",
            embedding_model="model",
        )
        points = [
            SimpleNamespace(id=1, orig_id="1", response_text="row one", status="active"),
            SimpleNamespace(id=2, orig_id="2", response_text="row two", status="active"),
            SimpleNamespace(id=3, orig_id="3", response_text="row three", status="active"),
        ]
        clusters = [
            SimpleNamespace(cluster_id=10, title="Primary", description="", sentiment="neutral", theme_name=None, n_points=1, is_active=True),
            SimpleNamespace(cluster_id=20, title="Second A", description="", sentiment="neutral", theme_name=None, n_points=1, is_active=True),
            SimpleNamespace(cluster_id=30, title="Second B", description="", sentiment="neutral", theme_name=None, n_points=1, is_active=True),
        ]
        assignments = [
            SimpleNamespace(point_id=1, cluster_id=10),
            SimpleNamespace(point_id=2, cluster_id=20),
            SimpleNamespace(point_id=3, cluster_id=30),
        ]
        edits = [
            SimpleNamespace(
                edit_type="join",
                payload={
                    "from_ids": [20, 30],
                    "to_id": 20,
                    "title": "Merged Secondary",
                    "description": "",
                    "sentiment": "neutral",
                },
            )
        ]
        cache_data = {
            "point_ids": np.array([1, 2, 3], dtype=np.int32),
            "embeddings": np.array(
                [
                    [1.0, 0.0],
                    [0.95, 0.05],
                    [0.8, 0.2],
                ],
                dtype=np.float32,
            ),
            "umap_3d": np.zeros((3, 3), dtype=np.float32),
        }

        with mock.patch("callbacks.export.get_session", return_value=session), \
             mock.patch("callbacks.export.get_points", return_value=points), \
             mock.patch("callbacks.export.get_clusters", return_value=clusters), \
             mock.patch("callbacks.export.get_cluster_assignments", return_value=assignments), \
             mock.patch("callbacks.export.get_all_edits", return_value=edits), \
             mock.patch("callbacks.export.cache_load", return_value=cache_data):
            workbook_bytes, _ = build_export_workbook("session-1")

        responses = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name="Responses")
        self.assertEqual(responses.loc[0, "theme"], "Primary, Merged Secondary")

    def test_build_export_workbook_absorbs_outlier_at_export(self):
        session = SimpleNamespace(
            session_id="session-2",
            session_name="DemoOutlier",
            id_col="id",
            response_col="response",
            n_points=3,
            csv_hash="hash234",
            embedding_model="model",
        )
        points = [
            SimpleNamespace(id=1, orig_id="1", response_text="anchor one", status="active"),
            SimpleNamespace(id=2, orig_id="2", response_text="anchor two", status="active"),
            SimpleNamespace(id=3, orig_id="3", response_text="nearby outlier", status="active"),
        ]
        clusters = [
            SimpleNamespace(cluster_id=10, title="Theme A", description="", sentiment="neutral", theme_name=None, n_points=2, is_active=True),
            SimpleNamespace(cluster_id=-1, title="Outliers", description="", sentiment="neutral", theme_name=None, n_points=1, is_active=True),
        ]
        assignments = [
            SimpleNamespace(point_id=1, cluster_id=10),
            SimpleNamespace(point_id=2, cluster_id=10),
            SimpleNamespace(point_id=3, cluster_id=-1),
        ]
        cache_data = {
            "point_ids": np.array([1, 2, 3], dtype=np.int32),
            "embeddings": np.array(
                [
                    [1.0, 0.0],
                    [0.98, 0.02],
                    [0.99, 0.01],
                ],
                dtype=np.float32,
            ),
            "umap_3d": np.zeros((3, 3), dtype=np.float32),
        }

        with mock.patch("callbacks.export.get_session", return_value=session), \
             mock.patch("callbacks.export.get_points", return_value=points), \
             mock.patch("callbacks.export.get_clusters", return_value=clusters), \
             mock.patch("callbacks.export.get_cluster_assignments", return_value=assignments), \
             mock.patch("callbacks.export.get_all_edits", return_value=[]), \
             mock.patch("callbacks.export.cache_load", return_value=cache_data):
            workbook_bytes, _ = build_export_workbook("session-2")

        responses = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name="Responses")
        self.assertEqual(responses.loc[2, "theme"], "Theme A")

        summary = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name="Cluster Summary")
        self.assertEqual(int(summary.loc[summary["theme"] == "Theme A", "count"].iloc[0]), 3)

    def test_build_export_preview_reports_centroid_export_stats(self):
        session = SimpleNamespace(
            session_id="session-4",
            session_name="Preview",
            id_col="id",
            response_col="response",
            n_points=3,
            csv_hash="hash456",
            embedding_model="model",
        )
        points = [
            SimpleNamespace(id=1, orig_id="1", response_text="a", status="active"),
            SimpleNamespace(id=2, orig_id="2", response_text="b", status="active"),
            SimpleNamespace(id=3, orig_id="3", response_text="c", status="active"),
        ]
        clusters = [
            SimpleNamespace(cluster_id=10, title="Theme A", description="", sentiment="neutral", theme_name=None, n_points=1, is_active=True),
            SimpleNamespace(cluster_id=20, title="Theme B", description="", sentiment="neutral", theme_name=None, n_points=1, is_active=True),
        ]
        assignments = [
            SimpleNamespace(point_id=1, cluster_id=10),
            SimpleNamespace(point_id=2, cluster_id=20),
            SimpleNamespace(point_id=3, cluster_id=-1),
        ]
        cache_data = {
            "point_ids": np.array([1, 2, 3], dtype=np.int32),
            "embeddings": np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.9, 0.1],
                ],
                dtype=np.float32,
            ),
            "umap_3d": np.zeros((3, 3), dtype=np.float32),
        }

        with mock.patch("callbacks.export.get_session", return_value=session), \
             mock.patch("callbacks.export.get_points", return_value=points), \
             mock.patch("callbacks.export.get_clusters", return_value=clusters), \
             mock.patch("callbacks.export.get_cluster_assignments", return_value=assignments), \
             mock.patch("callbacks.export.get_all_edits", return_value=[]), \
             mock.patch("callbacks.export.cache_load", return_value=cache_data):
            preview = build_export_preview("session-4")

        self.assertTrue(preview["available"])
        self.assertEqual(preview["cluster_count"], 2)
        self.assertEqual(preview["outliers_absorbed"], 1)

    def test_build_export_workbook_keeps_outlier_when_no_cluster_clears_threshold(self):
        session = SimpleNamespace(
            session_id="session-3",
            session_name="DemoOutlier",
            id_col="id",
            response_col="response",
            n_points=1,
            csv_hash="hash345",
            embedding_model="model",
        )
        points = [SimpleNamespace(id=1, orig_id="1", response_text="orphan", status="active")]
        clusters = [SimpleNamespace(cluster_id=-1, title="Outliers", description="", sentiment="neutral", theme_name=None, n_points=1, is_active=True)]
        assignments = [SimpleNamespace(point_id=1, cluster_id=-1)]
        cache_data = {
            "point_ids": np.array([1], dtype=np.int32),
            "embeddings": np.array([[0.2, 0.1]], dtype=np.float32),
            "umap_3d": np.zeros((1, 3), dtype=np.float32),
        }

        with mock.patch("callbacks.export.get_session", return_value=session), \
             mock.patch("callbacks.export.get_points", return_value=points), \
             mock.patch("callbacks.export.get_clusters", return_value=clusters), \
             mock.patch("callbacks.export.get_cluster_assignments", return_value=assignments), \
             mock.patch("callbacks.export.get_all_edits", return_value=[]), \
             mock.patch("callbacks.export.cache_load", return_value=cache_data):
            workbook_bytes, _ = build_export_workbook("session-3")

        responses = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name="Responses")
        self.assertEqual(responses.loc[0, "theme"], "Outliers")