import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from core import cache


class CoreCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        cache._mem.clear()

    def tearDown(self):
        cache._mem.clear()
        self.temp_dir.cleanup()

    def test_save_and_load_arrays_round_trip(self):
        arrays = {
            "embeddings": np.array([[1.0, 2.0]], dtype=np.float32),
            "umap_high": np.array([[3.0, 4.0]], dtype=np.float32),
            "umap_3d": np.array([[5.0, 6.0, 7.0]], dtype=np.float32),
            "point_ids": np.array([11], dtype=np.int32),
        }
        with mock.patch.object(cache, "_dir", return_value=self.cache_dir):
            cache.save("abc", arrays, embedding_model="model")
            loaded = cache.load("abc", embedding_model="model")

        self.assertIsNotNone(loaded)
        for key, expected in arrays.items():
            np.testing.assert_array_equal(loaded[key], expected)

    def test_save_and_load_membership_round_trip(self):
        membership = np.array([[0.8, 0.2], [0.4, 0.7]], dtype=np.float32)
        cids = np.array([10, 20], dtype=np.int32)
        thresholds = np.array([0.5, 0.6], dtype=np.float64)

        with mock.patch.object(cache, "_dir", return_value=self.cache_dir):
            cache.save_membership("abc", membership, cids, thresholds, embedding_model="model")
            loaded = cache.load_membership("abc", embedding_model="model")

        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded["membership"], membership)
        np.testing.assert_array_equal(loaded["cids"], cids)
        np.testing.assert_array_equal(loaded["thresholds"], thresholds)

    def test_evict_removes_memory_entry(self):
        arrays = {
            "embeddings": np.array([[1.0]], dtype=np.float32),
            "umap_high": np.array([[2.0]], dtype=np.float32),
            "umap_3d": np.array([[3.0, 4.0, 5.0]], dtype=np.float32),
            "point_ids": np.array([1], dtype=np.int32),
        }
        with mock.patch.object(cache, "_dir", return_value=self.cache_dir):
            cache.save("abc", arrays, embedding_model="model")
            cache.evict("abc", embedding_model="model")

        self.assertEqual(cache._mem, {})