import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from callbacks.export import build_export_workbook, update_export_href


class ExportCallbackTests(unittest.TestCase):
    def test_update_export_href_returns_fixed_export_url(self):
        self.assertEqual(update_export_href(None), "#")
        self.assertEqual(update_export_href("session-1"), "/api/export/session-1")

    def test_build_export_workbook_replays_joined_secondary_memberships(self):
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
        membership = np.array(
            [
                [0.9, 0.8, 0.7],
                [0.2, 0.95, 0.6],
                [0.1, 0.4, 0.96],
            ],
            dtype=np.float32,
        )
        mem_data = {
            "membership": membership,
            "cids": np.array([10, 20, 30], dtype=np.int32),
            "thresholds": np.array([0.5, 0.5, 0.5], dtype=np.float64),
        }
        cache_data = {"umap_3d": np.zeros((3, 3), dtype=np.float32)}

        with mock.patch("callbacks.export.get_session", return_value=session), \
             mock.patch("callbacks.export.get_points", return_value=points), \
             mock.patch("callbacks.export.get_clusters", return_value=clusters), \
             mock.patch("callbacks.export.get_cluster_assignments", return_value=assignments), \
             mock.patch("callbacks.export.get_all_edits", return_value=edits), \
             mock.patch("callbacks.export.load_membership", return_value=mem_data), \
             mock.patch("callbacks.export.cache_load", return_value=cache_data):
            workbook_bytes, _ = build_export_workbook("session-1")

        responses = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name="Responses")
        self.assertEqual(responses.loc[0, "theme"], "Primary, Merged Secondary")

    def test_build_export_workbook_replays_split_local_qualifying_scores(self):
        session = SimpleNamespace(
            session_id="session-2",
            session_name="DemoSplit",
            id_col="id",
            response_col="response",
            n_points=2,
            csv_hash="hash234",
            embedding_model="model",
        )
        points = [
            SimpleNamespace(id=1, orig_id="1", response_text="left text", status="active"),
            SimpleNamespace(id=2, orig_id="2", response_text="right text", status="active"),
        ]
        clusters = [
            SimpleNamespace(cluster_id=10, title="Original", description="", sentiment="neutral", theme_name=None, n_points=2, is_active=True),
        ]
        assignments = [
            SimpleNamespace(point_id=1, cluster_id=10),
            SimpleNamespace(point_id=2, cluster_id=10),
        ]
        edits = [
            SimpleNamespace(
                edit_type="split",
                payload={
                    "from_id": 10,
                    "new_assignments": [[0, 100], [1, 101]],
                    "new_cluster_info": {
                        "100": {"title": "Left", "description": "", "sentiment": "neutral"},
                        "101": {"title": "Right", "description": "", "sentiment": "neutral"},
                    },
                    "local_qualifying": {
                        "0": [[100, 0.95], [101, 0.72]],
                        "1": [[101, 0.97]],
                    },
                    "local_thresholds": {"100": 0.6, "101": 0.7},
                },
            )
        ]
        mem_data = {
            "membership": np.array([[0.9], [0.8]], dtype=np.float32),
            "cids": np.array([10], dtype=np.int32),
            "thresholds": np.array([0.5], dtype=np.float64),
        }
        cache_data = {"umap_3d": np.zeros((2, 3), dtype=np.float32)}

        with mock.patch("callbacks.export.get_session", return_value=session), \
             mock.patch("callbacks.export.get_points", return_value=points), \
             mock.patch("callbacks.export.get_clusters", return_value=clusters), \
             mock.patch("callbacks.export.get_cluster_assignments", return_value=assignments), \
             mock.patch("callbacks.export.get_all_edits", return_value=edits), \
             mock.patch("callbacks.export.load_membership", return_value=mem_data), \
             mock.patch("callbacks.export.cache_load", return_value=cache_data):
            workbook_bytes, _ = build_export_workbook("session-2")

        responses = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name="Responses")
        self.assertEqual(responses.loc[0, "theme"], "Left, Right")
        self.assertEqual(responses.loc[1, "theme"], "Right")

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
        mem_data = {
            "membership": np.array([[0.2, 0.1]], dtype=np.float32),
            "cids": np.array([10, 20], dtype=np.int32),
            "thresholds": np.array([0.5, 0.5], dtype=np.float64),
        }
        cache_data = {"umap_3d": np.zeros((1, 3), dtype=np.float32)}

        with mock.patch("callbacks.export.get_session", return_value=session), \
             mock.patch("callbacks.export.get_points", return_value=points), \
             mock.patch("callbacks.export.get_clusters", return_value=clusters), \
             mock.patch("callbacks.export.get_cluster_assignments", return_value=assignments), \
             mock.patch("callbacks.export.get_all_edits", return_value=[]), \
             mock.patch("callbacks.export.load_membership", return_value=mem_data), \
             mock.patch("callbacks.export.cache_load", return_value=cache_data):
            workbook_bytes, _ = build_export_workbook("session-3")

        responses = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name="Responses")
        self.assertEqual(responses.loc[0, "theme"], "Outliers")