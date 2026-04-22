import unittest
from unittest import mock

import numpy as np

from core.umap_runner import run_umap


class _FakeReducer:
    def __init__(self, n_components, **kwargs):
        self.n_components = n_components

    def fit_transform(self, embeddings):
        rows = embeddings.shape[0]
        return np.full((rows, self.n_components), float(self.n_components), dtype=np.float32)


class CoreUmapRunnerTests(unittest.TestCase):
    def test_run_umap_returns_two_float32_projections(self):
        embeddings = np.ones((4, 6), dtype=np.float32)
        progress = []

        with mock.patch("core.umap_runner.umap.UMAP", side_effect=lambda **kwargs: _FakeReducer(**kwargs)):
            umap_high, umap_3d = run_umap(
                embeddings,
                progress_cb=lambda step, total, message: progress.append((step, total, message)),
            )

        self.assertEqual(umap_high.dtype, np.float32)
        self.assertEqual(umap_3d.dtype, np.float32)
        self.assertEqual(umap_high.shape[0], 4)
        self.assertEqual(umap_3d.shape, (4, 3))
        self.assertEqual(progress[-1][0], 2)