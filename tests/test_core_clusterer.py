import unittest

import numpy as np

from core.clusterer import (
    build_base_cluster_list,
    choose_hdbscan_params,
    compute_centroid_thresholds,
)


class CoreClustererTests(unittest.TestCase):
    def test_choose_hdbscan_params_uses_sqrt_rule_and_low_min_samples(self):
        self.assertEqual(choose_hdbscan_params(100), (10, 2))
        self.assertEqual(choose_hdbscan_params(400), (20, 4))
        self.assertEqual(choose_hdbscan_params(900), (30, 5))

    def test_choose_hdbscan_params_rejects_non_positive_counts(self):
        with self.assertRaises(ValueError):
            choose_hdbscan_params(0)

    def test_build_base_cluster_list_counts_points(self):
        labels = np.array([0, 0, 1, -1], dtype=np.int32)
        result = build_base_cluster_list(labels, [0, 1, -1])
        counts = {row["cluster_id"]: row["n_points"] for row in result}
        self.assertEqual(counts[0], 2)
        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[-1], 1)

    def test_compute_centroid_thresholds_returns_per_cluster_outputs(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        rep_indices = {10: [0, 1], 12: [2, 3]}
        labels = np.array([10, 10, 12, 12], dtype=np.int32)
        centroids, cids, thresholds = compute_centroid_thresholds(
            embeddings,
            rep_indices,
            labels,
            named_merged_clusters=[10, 12],
            unique_raw_clusters=[10, 12],
            merge_map={},
            percentile=50,
        )

        self.assertEqual(centroids.shape, (2, 2))
        self.assertEqual(cids.tolist(), [10, 12])
        self.assertEqual(thresholds.shape, (2,))