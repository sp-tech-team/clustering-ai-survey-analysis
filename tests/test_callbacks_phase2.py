import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from callbacks.phase2 import assign_other_themes, do_split, render_cluster_list


class Phase2CallbackTests(unittest.TestCase):
    def test_do_split_logs_basic_split_payload(self):
        session = SimpleNamespace(csv_hash="hash123", embedding_model="model", api_key="key")
        arrays = {
            "point_ids": np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
            "umap_high": np.arange(12, dtype=np.float32).reshape(6, 2),
        }
        points = [
            SimpleNamespace(id=i, orig_id=str(i), response_text=f"text {i}", status="active")
            for i in range(1, 7)
        ]
        assignments = [SimpleNamespace(point_id=i, cluster_id=7) for i in range(1, 7)]
        clusters = [SimpleNamespace(cluster_id=7, title="Cluster 7", description="", sentiment="neutral", theme_name=None, n_points=6, is_active=True)]
        fake_state = SimpleNamespace(
            labels=np.array([7, 7, 7, 7, 7, 7], dtype=np.int32),
            info={7: SimpleNamespace(title="Cluster 7")},
        )
        split_result = (
            {0: 100, 1: 100, 2: 100, 3: 101, 4: 101, 5: 101},
            [100, 101],
        )

        with mock.patch("callbacks.phase2.get_session", return_value=session), \
             mock.patch("callbacks.phase2.cache_get", return_value=arrays), \
             mock.patch("callbacks.phase2.get_cluster_assignments", return_value=assignments), \
             mock.patch("callbacks.phase2.get_all_edits", return_value=[]), \
             mock.patch("callbacks.phase2.get_clusters", return_value=clusters), \
             mock.patch("callbacks.phase2.get_points", return_value=points), \
             mock.patch("callbacks.phase2.reconstruct", return_value=fake_state), \
             mock.patch("core.splitter.split_cluster", return_value=split_result), \
             mock.patch("openai.OpenAI", return_value=object()), \
             mock.patch("core.llm.summarise_cluster", return_value={"title": "Sub", "description": "Desc"}), \
             mock.patch("callbacks.phase2.log_edit") as log_edit:
            refresh, title, message, is_open = do_split(1, "session-1", {"selected_cluster_ids": [7]}, 0)

        self.assertEqual(refresh, 1)
        self.assertEqual(title, "Split done")
        self.assertTrue(is_open)
        self.assertIn("split into 2 sub-clusters", message)

        payload = log_edit.call_args.args[2]
        self.assertEqual(payload["new_assignments"][0], [0, 100])
        self.assertNotIn("local_qualifying", payload)
        self.assertEqual(payload["new_cluster_info"]["100"]["theme_name"], "Split from Cluster 7")

    def test_do_split_requires_exactly_one_cluster(self):
        refresh, title, message, is_open = do_split(1, "session-1", {"selected_cluster_ids": [1, 2]}, 0)
        self.assertEqual(title, "Split")
        self.assertEqual(message, "Select exactly one cluster to split.")
        self.assertTrue(is_open)

    def test_assign_other_themes_logs_theme_edit_for_selected_clusters(self):
        with mock.patch("callbacks.phase2.log_edit") as log_edit:
            refresh = assign_other_themes(1, "session-1", {"selected_cluster_ids": [7, 8]}, 2)

        self.assertEqual(refresh, 3)
        log_edit.assert_called_once_with(
            "session-1",
            "theme",
            {"cluster_ids": [7, 8], "theme_name": "Other Themes"},
        )

    def test_render_cluster_list_includes_other_themes_footer_count(self):
        clusters = [
            SimpleNamespace(cluster_id=7, title="Cluster 7", description="", sentiment="neutral", theme_name="Other Themes", n_points=2, is_active=True),
            SimpleNamespace(cluster_id=8, title="Cluster 8", description="", sentiment="neutral", theme_name=None, n_points=1, is_active=True),
        ]
        assignments = [
            SimpleNamespace(point_id=1, cluster_id=7),
            SimpleNamespace(point_id=2, cluster_id=7),
            SimpleNamespace(point_id=3, cluster_id=-1),
        ]
        points = [
            SimpleNamespace(id=1, status="active"),
            SimpleNamespace(id=2, status="active"),
            SimpleNamespace(id=3, status="active"),
        ]
        fake_state = SimpleNamespace(
            labels=np.array([7, 7, -1], dtype=np.int32),
            info={
                7: SimpleNamespace(theme_name="Other Themes", is_active=True, title="Cluster 7", description=""),
                8: SimpleNamespace(theme_name=None, is_active=True, title="Cluster 8", description=""),
                -1: SimpleNamespace(theme_name=None, is_active=True, title="Other Themes", description=""),
            },
            active_ids=[7, 8],
        )
        session = SimpleNamespace(n_points=4, csv_hash="hash123", embedding_model="model")

        with mock.patch("callbacks.phase2.get_clusters", return_value=clusters), \
             mock.patch("callbacks.phase2.get_all_edits", return_value=[]), \
             mock.patch("callbacks.phase2.get_cluster_assignments", return_value=assignments), \
             mock.patch("callbacks.phase2.reconstruct", return_value=fake_state), \
             mock.patch("callbacks.phase2.get_session", return_value=session), \
             mock.patch("callbacks.phase2.get_points", return_value=points):
            list_group, count_badge, _ = render_cluster_list(1, "session-1", {"selected_cluster_ids": []})

        footer_labels = []
        for item in list_group.children[1:]:
            label_span = item.children[0]
            footer_labels.append(label_span.children)

        self.assertEqual(count_badge, "1")
        self.assertIn("◌ Other Themes", footer_labels)