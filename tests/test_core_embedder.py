import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from core.embedder import get_embeddings


class CoreEmbedderTests(unittest.TestCase):
    def test_get_embeddings_batches_and_reports_progress(self):
        responses = [
            SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]), SimpleNamespace(embedding=[0.0, 1.0])]),
            SimpleNamespace(data=[SimpleNamespace(embedding=[0.5, 0.5])]),
        ]
        client = SimpleNamespace(embeddings=SimpleNamespace(create=mock.Mock(side_effect=responses)))
        progress = []

        with mock.patch("core.embedder.time.sleep") as sleep_mock:
            result = get_embeddings(
                client,
                ["a", "b", "c"],
                progress_cb=lambda current, total, message: progress.append((current, total, message)),
                batch_size=2,
            )

        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(len(progress), 2)
        sleep_mock.assert_called_once()
