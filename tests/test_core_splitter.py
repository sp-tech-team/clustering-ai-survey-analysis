import unittest

import numpy as np

from core.splitter import split_cluster


class CoreSplitterTests(unittest.TestCase):
    def test_split_cluster_returns_local_assignments_and_thresholds(self):
        cluster_a = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.08]])
        cluster_b = np.array([[5.0, 5.0], [5.1, 5.0], [5.0, 5.1], [5.1, 5.1], [5.05, 5.08]])
        points = np.vstack([cluster_a, cluster_b]).astype(np.float32)

        assignments, new_ids, qualifying, thresholds = split_cluster(7, list(range(len(points))), points, 100)

        self.assertEqual(len(new_ids), 2)
        self.assertEqual(sorted(set(assignments.values())), sorted(new_ids))
        self.assertEqual(sorted(thresholds), sorted(new_ids))
        self.assertTrue(0 in qualifying)

    def test_split_cluster_rejects_tiny_clusters(self):
        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            split_cluster(3, [0, 1, 2], points, 10)