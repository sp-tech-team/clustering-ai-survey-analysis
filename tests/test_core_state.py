import unittest
from types import SimpleNamespace

import numpy as np

from core.state import reconstruct


class CoreStateTests(unittest.TestCase):
    def test_reconstruct_replays_join_rename_exclude_theme(self):
        base_labels = np.array([0, 0, 1, 1], dtype=np.int32)
        base_info = {
            0: {"title": "A", "description": "", "sentiment": "neutral", "n_points": 2, "is_active": True},
            1: {"title": "B", "description": "", "sentiment": "neutral", "n_points": 2, "is_active": True},
        }
        edits = [
            SimpleNamespace(edit_type="join", payload={"from_ids": [0, 1], "to_id": 0, "title": "Merged", "description": "desc", "sentiment": "positive"}),
            SimpleNamespace(edit_type="rename", payload={"cluster_id": 0, "title": "Renamed", "description": "new desc"}),
            SimpleNamespace(edit_type="theme", payload={"cluster_ids": [0], "theme_name": "Theme X"}),
            SimpleNamespace(edit_type="exclude", payload={"cluster_id": 0, "reason": "low_info"}),
        ]

        state = reconstruct(base_labels, base_info, edits)

        self.assertTrue((state.labels == 0).all())
        self.assertEqual(state.info[0].title, "Renamed")
        self.assertEqual(state.info[0].theme_name, "Theme X")
        self.assertFalse(state.info[0].is_active)

    def test_reconstruct_replays_split(self):
        base_labels = np.array([5, 5, 5, 5], dtype=np.int32)
        base_info = {
            5: {"title": "Original", "description": "", "sentiment": "neutral", "n_points": 4, "is_active": True},
        }
        edits = [
            SimpleNamespace(
                edit_type="split",
                payload={
                    "from_id": 5,
                    "new_assignments": [[0, 10], [1, 10], [2, 11], [3, 11]],
                    "new_cluster_info": {
                        "10": {"title": "Left", "description": "", "sentiment": "neutral", "theme_name": "Split from Original"},
                        "11": {"title": "Right", "description": "", "sentiment": "neutral", "theme_name": "Split from Original"},
                    },
                },
            )
        ]

        state = reconstruct(base_labels, base_info, edits)

        self.assertNotIn(5, state.info)
        self.assertEqual(state.labels.tolist(), [10, 10, 11, 11])
        self.assertEqual(state.info[10].n_points, 2)
        self.assertEqual(state.info[11].n_points, 2)
        self.assertEqual(state.info[10].theme_name, "Split from Original")
        self.assertEqual(state.info[11].theme_name, "Split from Original")