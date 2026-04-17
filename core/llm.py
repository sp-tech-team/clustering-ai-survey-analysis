"""LLM prompt helpers — ported from the notebook pipeline."""

import json
import time
from typing import Callable, Optional

from openai import OpenAI

from config import LLM_MODEL, LLM_MODEL_SMART

# ── Classify responses as low_info / substantive ──────────────────────────────

_CLASSIFY_PROMPT = """\
You are filtering open-ended survey responses to remove ones that carry no useful information.

Survey question: "{question}"

A response is "low_info" if — regardless of how it is phrased or how many words it uses —
it ultimately conveys no real content in relation to the question. This includes any response
whose actual meaning reduces to: nothing, none, not applicable, I don't know, I'm not sure,
no answer, or any equivalent sentiment expressed in any words or sentence structure.

A response is "substantive" if it communicates at least one genuine thought, observation,
concern, intention, or piece of feedback that is meaningful in the context of the question —
even if brief.

When in doubt, prefer "low_info". A politely worded non-answer is still a non-answer.

Respond ONLY with a valid JSON array of objects, one per response, preserving the original order:
[{{"id": 0, "label": "low_info"}}, {{"id": 1, "label": "substantive"}}, ...]

Responses to classify (id, text):
{responses}"""


def classify_batch(
    client: OpenAI, texts: list[str], question: str = ""
) -> list[str]:
    """Return list of 'substantive' | 'low_info' in input order."""
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(texts))
    prompt   = _CLASSIFY_PROMPT.format(
        question=question or "Please share your reflections.",
        responses=numbered,
    )
    max_tok  = max(512, len(texts) * 50)

    raw = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=max_tok,
    ).choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()

    results = json.loads(raw)
    labels  = ["substantive"] * len(texts)
    for item in results:
        labels[item["id"]] = item["label"]
    return labels


# ── Summarise all clusters in one call ───────────────────────────────────────

_SUMMARISE_ALL_PROMPT = """\
You are analysing open-ended survey responses grouped into clusters.

Survey question: "{question}"
Total substantive responses: {total}

Below are the clusters, each with representative responses.
Generate a title (3-5 words) and description (2-3 sentences) for EVERY cluster.
The titles and descriptions must be heterogeneous — each should highlight what is
distinct and specific about that cluster relative to all the others.
Avoid generic language that could apply to multiple clusters.

{clusters_block}

Respond ONLY with valid JSON, no preamble — a list in the same order as the clusters above:
[{{"cluster_id": 0, "title": "...", "description": "..."}}, ...]"""


def summarise_all_clusters(
    client: OpenAI,
    clusters: list[dict],   # [{cluster_id, n_points, rep_texts: [str]}]
    total: int,
    question: str = "",
) -> dict:                  # returns {cluster_id: {title, description}}
    blocks = []
    for c in clusters:
        cid  = c["cluster_id"]
        size = c["n_points"]
        pct  = round(size / max(total, 1) * 100, 1)
        reps = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(c["rep_texts"]))
        blocks.append(f"Cluster {cid} (n={size}, {pct}% of total):\n{reps}")
    clusters_block = "\n\n".join(blocks)
    prompt = _SUMMARISE_ALL_PROMPT.format(
        question=question or "Please share your reflections.",
        total=total,
        clusters_block=clusters_block,
    )
    max_tokens = max(600, len(clusters) * 200)
    results = _call_json(client, prompt, max_tokens=max_tokens, temperature=0.3)
    return {item["cluster_id"]: {"title": item["title"], "description": item["description"]}
            for item in results}


# ── Summarise a single cluster (used for interactive split) ───────────────────

_SUMMARISE_PROMPT = """\
You are analysing open-ended survey responses from {group}.

Survey question: "{question}"
Cluster size: {size} responses ({pct}% of {total})

Representative responses:
{formatted}

Provide: 1) title (3-5 words), 2) description (2-3 sentences)
Respond ONLY with valid JSON, no preamble:
{{"title": "...", "description": "..."}}"""


def summarise_cluster(
    client: OpenAI,
    reps: list[str],
    size: int = 0,
    total: int = 1,
    question: str = "",
    is_outlier: bool = False,
) -> dict:
    group     = "ungrouped/miscellaneous responses" if is_outlier else "a cluster of thematically similar responses"
    formatted = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reps))
    size      = size or len(reps)
    pct       = round(size / max(total, 1) * 100, 1)
    prompt    = _SUMMARISE_PROMPT.format(
        group=group,
        question=question or "Please share your reflections.",
        size=size, pct=pct, total=total,
        formatted=formatted,
    )
    return _call_json(client, prompt, max_tokens=350)


# ── Merge suggestion ──────────────────────────────────────────────────────────

_MERGE_PROMPT = """\
You are reviewing clusters from an open-ended survey analysis.

Survey question: "{question}"

{clusters_block}

Identify clusters that cover the same or closely related theme and should be merged.
Merge clusters whenever they address the same underlying idea — even if phrased or approached differently.
Treat cluster size as important context.
If a cluster is less than 1% of the total responses, prefer merging it into a larger thematically compatible cluster rather than keeping it standalone.
Only leave a <1% cluster unmerged if it is clearly distinct and would lose meaning if absorbed.
Suggesting zero merges is fine if the clusters are genuinely distinct.

Respond ONLY with valid JSON, no preamble:
{{"merges": [{{"cluster_ids": [0, 3]}}], "no_merge_reason": ""}}"""


def suggest_merges(
    client: OpenAI,
    question: str,
    reps_texts: dict,
    counts: dict,
    total: int,
) -> dict:
    """reps_texts: {cid: [sample_str, ...]}, counts: {cid: n_points}"""
    descs = []
    for cid in sorted(reps_texts.keys()):
        if cid == -1:
            continue
        samples = reps_texts.get(cid, [])[:5]
        count = counts.get(cid, 0)
        pct = round(count / max(total, 1) * 100, 1)
        numbered = "\n".join(f'  {i+1}. "{t}"' for i, t in enumerate(samples))
        descs.append(f"Cluster {cid} (n={count}, {pct}% of total):\n{numbered}")
    clusters_block = "\n\n".join(descs)
    prompt = _MERGE_PROMPT.format(question=question, clusters_block=clusters_block)
    return _call_json(client, prompt, max_tokens=5000, temperature=0.2)


# ── Manual join summary ───────────────────────────────────────────────────────

_JOIN_PROMPT = """\
You are merging two or more survey response clusters into one.

Survey question: "{question}"

Clusters being merged:
{clusters_block}

Representative responses sampled from the merged pool:
{sampled_block}

Generate a new title (3-5 words) and description (2-3 sentences) for the merged cluster.
Respond ONLY with valid JSON, no preamble:
{{"title": "...", "description": "..."}}"""


def summarise_join(
    client: OpenAI,
    clusters: list[dict],  # [{title, count}]
    sampled_reps: list[str],
    question: str = "",
) -> dict:
    descs = []
    for cluster in clusters:
        title = cluster.get("title", "")
        count = int(cluster.get("count", 0))
        descs.append(f"• {title} (n={count})")
    clusters_block = "\n\n".join(descs)
    sampled_block = "\n".join(f"  {i+1}. {text}" for i, text in enumerate(sampled_reps))
    prompt = _JOIN_PROMPT.format(
        question=question or "Please share your reflections.",
        clusters_block=clusters_block,
        sampled_block=sampled_block,
    )
    return _call_json(client, prompt, max_tokens=300)


# ── Split sub-cluster summaries ───────────────────────────────────────────────

def summarise_split_clusters(
    client: OpenAI,
    sub_clusters: dict,  # {new_cluster_id: [rep_texts]}
    total: int,
    question: str = "",
) -> dict:
    """Returns {new_cluster_id: {title, description, sentiment}}."""
    results = {}
    for cid, reps in sub_clusters.items():
        size  = len(reps)
        try:
            s = summarise_cluster(client, reps, size=size, total=total, question=question)
            results[cid] = s
        except Exception:
            results[cid] = {"title": f"Sub-cluster {cid}", "description": "", "sentiment": "unknown"}
        time.sleep(0.5)
    return results


# ── Internal helper ───────────────────────────────────────────────────────────

def _call_json(
    client: OpenAI,
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.3,
) -> dict:
    resp = client.chat.completions.create(
        model=LLM_MODEL_SMART,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)
