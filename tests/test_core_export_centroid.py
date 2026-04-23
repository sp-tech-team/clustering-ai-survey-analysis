import unittest
from types import SimpleNamespace

import numpy as np

from core.export_centroid import compute_export_centroid_assignments


class ExportCentroidTests(unittest.TestCase):
    def test_outlier_is_absorbed_by_best_qualifying_cluster(self):
        labels = np.array([10, 10, -1], dtype=np.int32)
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.98, 0.02],
                [0.99, 0.01],
            ],
            dtype=np.float32,
        )
        state = SimpleNamespace(
            active_ids=[10],
            info={10: SimpleNamespace(is_active=True)},
        )

        export_labels, secondary_map, diagnostics = compute_export_centroid_assignments(embeddings, labels, state)

        self.assertEqual(export_labels.tolist(), [10, 10, 10])
        self.assertEqual(secondary_map, {})
        self.assertEqual(diagnostics["outliers_absorbed"], 1)

    def test_primary_stays_fixed_while_secondary_is_added(self):
        labels = np.array([10, 20, 20], dtype=np.int32)
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.98, 0.2],
            ],
            dtype=np.float32,
        )
        state = SimpleNamespace(
            active_ids=[10, 20],
            info={10: SimpleNamespace(is_active=True), 20: SimpleNamespace(is_active=True)},
        )

        export_labels, secondary_map, _diagnostics = compute_export_centroid_assignments(
            embeddings,
            labels,
            state,
            max_secondary_clusters=2,
            percentile=50,
        )

        self.assertEqual(export_labels.tolist(), [10, 20, 20])
        self.assertEqual(secondary_map[2], [10])

    def test_inactive_cluster_points_are_not_reassigned(self):
        labels = np.array([10, 20], dtype=np.int32)
        embeddings = np.array([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32)
        state = SimpleNamespace(
            active_ids=[10],
            info={10: SimpleNamespace(is_active=True), 20: SimpleNamespace(is_active=False)},
        )

        export_labels, secondary_map, diagnostics = compute_export_centroid_assignments(embeddings, labels, state)

        self.assertEqual(export_labels.tolist(), [10, 20])
        self.assertEqual(secondary_map, {})
        self.assertEqual(diagnostics["outliers_absorbed"], 0)