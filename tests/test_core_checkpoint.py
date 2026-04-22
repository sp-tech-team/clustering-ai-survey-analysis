import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from core import checkpoint


class CoreCheckpointTests(unittest.TestCase):
    def test_arrays_ready_delegates_to_cache(self):
        with mock.patch("core.checkpoint.cache.exists", return_value=True) as mocked:
            self.assertTrue(checkpoint.arrays_ready("hash123", "model"))
        mocked.assert_called_once_with("hash123", "model")

    def test_load_session_state_returns_none_for_missing_session(self):
        with mock.patch("core.checkpoint.get_session", return_value=None):
            self.assertIsNone(checkpoint.load_session_state("missing"))

    def test_load_session_state_phase_two_reconstructs_cluster_state(self):
        session = SimpleNamespace(phase=2, csv_hash="hash123", embedding_model="model")
        points = [SimpleNamespace(id=10), SimpleNamespace(id=11)]
        assignments = [SimpleNamespace(point_id=10, cluster_id=0), SimpleNamespace(point_id=11, cluster_id=1)]
        clusters = [
            SimpleNamespace(cluster_id=0, title="A", description="", sentiment="neutral", n_points=1, theme_name=None, is_active=True),
            SimpleNamespace(cluster_id=1, title="B", description="", sentiment="neutral", n_points=1, theme_name=None, is_active=True),
        ]
        arrays = {"embeddings": np.zeros((2, 2), dtype=np.float32)}

        with mock.patch.multiple(
            checkpoint,
            get_session=mock.DEFAULT,
            get_points=mock.DEFAULT,
            get_clusters=mock.DEFAULT,
            get_cluster_assignments=mock.DEFAULT,
            get_all_edits=mock.DEFAULT,
        ) as mocked, mock.patch("core.checkpoint.cache.load", return_value=arrays), mock.patch("db.queries.count_edits", return_value=0):
            mocked["get_session"].return_value = session
            mocked["get_points"].return_value = points
            mocked["get_clusters"].return_value = clusters
            mocked["get_cluster_assignments"].return_value = assignments
            mocked["get_all_edits"].return_value = []
            result = checkpoint.load_session_state("session-1")

        self.assertEqual(result["phase"], 2)
        self.assertEqual(result["cluster_state"].labels.tolist(), [0, 1])
        self.assertEqual(result["n_edits"], 0)