import unittest
from types import SimpleNamespace

from core.llm import classify_batch, summarise_all_clusters, summarise_cluster, suggest_merges


def _fake_client_with_content(content):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])
    completions = SimpleNamespace(create=lambda **kwargs: response)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


class CoreLlmTests(unittest.TestCase):
    def test_classify_batch_parses_json_labels(self):
        client = _fake_client_with_content('[{"id": 0, "label": "low_info"}, {"id": 1, "label": "substantive"}]')
        labels = classify_batch(client, ["n/a", "pricing is high"], question="What do you think?")
        self.assertEqual(labels, ["low_info", "substantive"])

    def test_summarise_all_clusters_returns_mapping(self):
        client = _fake_client_with_content('[{"cluster_id": 10, "title": "Pricing", "description": "Cost concerns."}]')
        result = summarise_all_clusters(
            client,
            [{"cluster_id": 10, "n_points": 3, "rep_texts": ["too expensive"]}],
            total=3,
            question="Feedback?",
        )
        self.assertEqual(result[10]["title"], "Pricing")

    def test_summarise_cluster_and_suggest_merges_parse_json(self):
        summary_client = _fake_client_with_content('{"title": "Support", "description": "Service quality feedback."}')
        merge_client = _fake_client_with_content('{"merges": [{"cluster_ids": [1, 2]}], "no_merge_reason": ""}')

        summary = summarise_cluster(summary_client, ["great service"], size=1, total=10, question="Feedback?")
        merges = suggest_merges(merge_client, "Feedback?", {1: ["a"], 2: ["b"]}, {1: 5, 2: 1}, total=10)

        self.assertEqual(summary["title"], "Support")
        self.assertEqual(merges["merges"][0]["cluster_ids"], [1, 2])