import unittest
from types import SimpleNamespace

import numpy as np

from core.secondary_membership import replay_secondary_memberships


class SecondaryMembershipReplayTests(unittest.TestCase):
    def test_replay_secondary_memberships_applies_join_and_split_rules(self):
        mem_data = {
            "membership": np.array(
                [
                    [0.9, 0.8, 0.7],
                    [0.2, 0.95, 0.6],
                    [0.1, 0.4, 0.96],
                ],
                dtype=np.float32,
            ),
            "cids": np.array([10, 20, 30], dtype=np.int32),
            "thresholds": np.array([0.5, 0.5, 0.5], dtype=np.float64),
        }
        edits = [
            SimpleNamespace(edit_type="join", payload={"from_ids": [20, 30], "to_id": 20}),
            SimpleNamespace(
                edit_type="split",
                payload={
                    "new_assignments": [[2, 100]],
                    "local_qualifying": {"2": [[100, 0.99], [101, 0.71]]},
                },
            ),
        ]
        state = SimpleNamespace(
            labels=np.array([10, 20, 100], dtype=np.int32),
            info={
                10: SimpleNamespace(is_active=True),
                20: SimpleNamespace(is_active=True),
                100: SimpleNamespace(is_active=True),
                101: SimpleNamespace(is_active=True),
            },
        )

        secondary_map, diagnostics = replay_secondary_memberships(mem_data, edits, state, max_secondary_clusters=3)

        self.assertEqual(secondary_map[0], [20])
        self.assertEqual(secondary_map[2], [101])
        self.assertEqual(diagnostics["points_with_secondaries"], 2)

    def test_replay_secondary_memberships_handles_missing_cache(self):
        state = SimpleNamespace(labels=np.array([10, -1], dtype=np.int32), info={10: SimpleNamespace(is_active=True)})
        secondary_map, diagnostics = replay_secondary_memberships(None, [], state)

        self.assertEqual(secondary_map, {})
        self.assertFalse(diagnostics["membership_available"])