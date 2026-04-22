import unittest
from unittest import mock

import numpy as np

from core.clusterer import (
    aggregate_membership_by_cluster,
    assign_clusters_from_scores,
    build_base_cluster_list,
    compute_centroid_thresholds,
    compute_cluster_thresholds,
    compute_soft_membership,
)


class CoreClustererTests(unittest.TestCase):
    def test_build_base_cluster_list_counts_points(self):
        labels = np.array([0, 0, 1, -1], dtype=np.int32)
        result = build_base_cluster_list(labels, [0, 1, -1])
        counts = {row["cluster_id"]: row["n_points"] for row in result}
        self.assertEqual(counts[0], 2)
        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[-1], 1)

    def test_aggregate_thresholds_and_assignment_unify_membership(self):
        membership = np.array(
            [
                [0.9, 0.2, 0.1],
                [0.6, 0.7, 0.2],
                [0.1, 0.2, 0.8],
                [0.2, 0.1, 0.1],
            ],
            dtype=np.float32,
        )
        raw_cids = np.array([10, 11, 12], dtype=np.int32)
        merged, merged_cids = aggregate_membership_by_cluster(membership, raw_cids, {11: 10})
        thresholds = compute_cluster_thresholds(
            merged,
            np.array([10, 10, 12, -1], dtype=np.int32),
            merged_cids,
            percentile=10,
            floor=0.05,
        )
        labels, qualifying = assign_clusters_from_scores(merged, merged_cids, thresholds)

        self.assertEqual(merged_cids.tolist(), [10, 12])
        self.assertEqual(labels.tolist(), [10, -1, 12, -1])
        self.assertEqual(qualifying[0][0][0], 10)

    def test_assign_clusters_keeps_multiple_qualifiers_sorted(self):
        scores = np.array([[0.91, 0.83, 0.2]], dtype=np.float32)
        cids = np.array([10, 20, 30], dtype=np.int32)
        thresholds = np.array([0.5, 0.7, 0.5], dtype=np.float64)

        labels, qualifying = assign_clusters_from_scores(scores, cids, thresholds)

        self.assertEqual(labels.tolist(), [10])
        self.assertEqual(qualifying[0], [(10, 0.9100000262260437), (20, 0.8299999833106995)])

    def test_compute_soft_membership_uses_all_primary_members(self):
        labels = np.array([0, 0, 1], dtype=np.int32)
        memberships = np.array(
            [
                [0.8, 0.2],
                [0.6, 0.1],
                [0.1, 0.7],
            ],
            dtype=np.float32,
        )
        with mock.patch("core.clusterer.hdbscan.all_points_membership_vectors", return_value=memberships):
            result_membership, cids, thresholds = compute_soft_membership(object(), labels, [0, 1], percentile=10, floor=0.05)

        np.testing.assert_array_equal(result_membership, memberships)
        self.assertEqual(cids.tolist(), [0, 1])
        self.assertGreaterEqual(thresholds[0], 0.05)
        self.assertGreaterEqual(thresholds[1], 0.05)

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