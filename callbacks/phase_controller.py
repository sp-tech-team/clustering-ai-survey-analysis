"""
Unified phase controller.

A SINGLE @callback owns ALL phase-related layout outputs so Dash never
sees duplicate output registrations — no `allow_duplicate` needed.

Handles triggers via callback_context:
  session-id-store   → page load / navigation (sets up initial panel visibility)
  progress-interval  → polls background task registry every POLL_INTERVAL_MS
  run-clustering-btn → starts Phase 2 background worker
  recluster-confirm  → confirms re-clustering with excluded clusters removed
"""

from dash import Input, Output, State, callback, html, no_update, callback_context
import dash_bootstrap_components as dbc

import tasks
from db.queries import (
    get_session, advance_phase, get_points,
    mark_points_status, save_clusters, save_cluster_assignments,
    wipe_cluster_state, get_clusters, get_cluster_assignments, get_all_edits,
)
from core.cache import exists as cache_exists, save as cache_save, load as cache_get
from config import EMBEDDING_MODEL
from layout.components import progress_card, phase_badge

_N = no_update
_NOOP = (_N,) * 13   # len must match output count below


def _compute_llm_filter_threshold(lengths: list[int], fallback: int, percentile: int,
                                  min_chars: int, max_chars: int,
                                  min_samples: int) -> int:
    if len(lengths) < min_samples:
        return fallback

    if min(lengths) == max(lengths):
        return fallback

    import numpy as np

    percentile_value = int(round(float(np.percentile(lengths, percentile))))
    return max(min_chars, min(max_chars, percentile_value))


@callback(
    Output("session-name-display",        "children"),        # 0
    Output("phase-badge-display",         "children"),        # 1
    Output("phase-store",                 "data"),            # 2
    Output("phase1-overlay",              "is_open"),         # 3
    Output("phase2-trigger-panel",        "is_open"),         # 4
    Output("main-edit-panel",             "is_open"),         # 5
    Output("task-store",                  "data"),            # 6
    Output("phase1-progress-container",   "children"),        # 7
    Output("phase2-progress-container",   "children"),        # 8
    Output("cluster-refresh-store",       "data"),            # 9
    Output("recluster-btn-container",     "style"),           # 10
    Output("recluster-progress-collapse", "is_open"),         # 11
    Output("recluster-progress-container","children"),        # 12
    Input("session-id-store",             "data"),
    Input("progress-interval",            "n_intervals"),
    Input("run-clustering-btn",           "n_clicks"),
    Input("recluster-confirm",            "n_clicks"),
    State("task-store",                   "data"),
    State("cluster-refresh-store",        "data"),
    State("modal-recluster",              "is_open"),
)
def phase_controller(session_id, n_intervals, run_btn, recluster_btn,
                     task_data, refresh_ctr, recluster_modal_open):
    ctx = callback_context
    # On initial render ctx.triggered may be empty — default to page-load logic
    triggered_id = ctx.triggered_id if ctx.triggered else "session-id-store"
    task_data = task_data or {}

    # ── Page load / navigation ─────────────────────────────────────────────────
    if triggered_id == "session-id-store":
        if not session_id:
            return _NOOP

        sess = get_session(session_id)
        if not sess:
            return "Session not found", "", {"phase": -1}, False, False, False, {}, _N, _N, _N, _N, _N, _N

        badge      = phase_badge(sess.phase)
        name       = sess.session_name
        phase_data = {"phase": sess.phase}
        recluster_style = {"display": "block"} if sess.phase >= 2 else {"display": "none"}
        embedding_model = sess.embedding_model or EMBEDDING_MODEL

        if sess.phase == 0:
            _start_phase1_bg(session_id, sess)
            return name, badge, phase_data, True, False, False, \
                   {"status": "running", "task_type": "phase1"}, _N, _N, _N, {"display": "none"}, _N, _N

        if sess.phase == 1:
            if cache_exists(sess.csv_hash, embedding_model):
                return name, badge, phase_data, False, True, False, {"status": "idle"}, _N, _N, _N, {"display": "none"}, _N, _N
            # Cache missing — re-embed
            _start_phase1_bg(session_id, sess)
            return name, badge, phase_data, True, False, False, \
                   {"status": "running", "task_type": "phase1"}, _N, _N, _N, {"display": "none"}, _N, _N

        if sess.phase >= 2:
            return name, badge, phase_data, False, False, True, {"status": "idle"}, _N, _N, 1, recluster_style, _N, _N

        return name, badge, phase_data, False, False, False, {}, _N, _N, _N, {"display": "none"}, _N, _N

    # ── Start clustering button ────────────────────────────────────────────────
    if triggered_id == "run-clustering-btn":
        if not run_btn or not session_id:
            return _NOOP
        sess = get_session(session_id)
        if not sess:
            return _NOOP
        _start_phase2_bg(session_id, sess)
        msg = html.P("Clustering started…", className="text-muted small")
        return _N, _N, _N, _N, True, _N, {"status": "running", "task_type": "phase2"}, _N, msg, _N, _N, _N, _N

    # ── Re-cluster confirmed ───────────────────────────────────────────────────
    if triggered_id == "recluster-confirm":
        if not recluster_btn or not session_id:
            return _NOOP
        sess = get_session(session_id)
        if not sess:
            return _NOOP

        # Find which cluster IDs are currently excluded via the edit log
        import numpy as np
        from core.state import reconstruct
        assignments = get_cluster_assignments(session_id)
        base_labels  = np.array([a.cluster_id for a in assignments], dtype=int)
        clusters_db  = get_clusters(session_id)
        base_info    = {
            c.cluster_id: {
                "title": c.title, "description": c.description,
                "sentiment": c.sentiment, "theme_name": c.theme_name,
                "n_points": c.n_points, "is_active": c.is_active,
            }
            for c in clusters_db
        }
        edits = get_all_edits(session_id)
        state = reconstruct(base_labels, base_info, edits)

        # Collect point IDs whose cluster is inactive → mark as low_info_user.
        # Use assignment order here because it is the same order used to build
        # base_labels/state.labels above, so labels stay aligned to point_ids.
        excluded_cids = {cid for cid, ci in state.info.items() if not ci.is_active and cid != -1}
        if excluded_cids:
            excluded_pt_ids = [
                assignment.point_id
                for assignment, lbl in zip(assignments, state.labels)
                if int(lbl) in excluded_cids
            ]
            if excluded_pt_ids:
                mark_points_status(session_id, excluded_pt_ids, "low_info_user")

        # Wipe all cluster state and reset phase to 1
        wipe_cluster_state(session_id)
        advance_phase(session_id, 1)

        # Kick off phase 2 again
        _start_phase2_bg(session_id, get_session(session_id))
        msg = html.P("Re-clustering started…", className="text-muted small")
        return (_N, phase_badge(1), {"phase": 1},
                _N, _N, False,
                {"status": "running", "task_type": "phase2", "recluster": True},
                _N, _N, _N,
                {"display": "none"}, True, msg)

    # ── Polling (interval) ─────────────────────────────────────────────────────
    if triggered_id == "progress-interval":
        if not session_id:
            return _NOOP

        task_type = task_data.get("task_type")
        t = tasks.get_task(session_id)

        if task_type == "phase1":
            if t is None or t.status in ("idle", ""):
                return _NOOP
            card = progress_card(t.message, t.progress, t.steps)
            if t.status == "done":
                return _N, phase_badge(1), {"phase": 1}, False, True, _N, \
                       {"status": "idle"}, card, _N, _N, {"display": "none"}, _N, _N
            if t.status == "error":
                err = dbc.Alert(f"Phase 1 failed: {t.error}", color="danger")
                return _N, _N, _N, True, _N, _N, {"status": "error"}, err, _N, _N, _N, _N, _N
            # still running
            return _N, _N, _N, True, _N, _N, _N, card, _N, _N, _N, _N, _N

        if task_type == "phase2":
            if t is None or t.status in ("idle", ""):
                return _NOOP
            card = progress_card(t.message, t.progress, t.steps)
            if t.status == "done":
                return (_N, phase_badge(2), {"phase": 2},
                        _N, False, True,
                        {"status": "idle"},
                        _N, card, (refresh_ctr or 0) + 1,
                        {"display": "block"}, False, _N)
            if t.status == "error":
                err = dbc.Alert(f"Phase 2 failed: {t.error}", color="danger")
                return (_N, _N, _N,
                        _N, True, _N,
                        {"status": "error"},
                        _N, err, _N,
                        _N, False, err)
            # still running — show progress in the active container
            recluster_running = task_data.get("recluster", False)
            if recluster_running:
                return _N, _N, _N, _N, _N, _N, _N, _N, _N, _N, _N, True, card
            return _N, _N, _N, _N, True, _N, _N, _N, card, _N, _N, _N, _N

    return _NOOP


# ── Background task launchers ─────────────────────────────────────────────────

def _start_phase1_bg(session_id: str, sess) -> None:
    existing = tasks.get_task(session_id)
    if existing and existing.status == "running":
        return
    tasks.run_in_background(session_id, _phase1_worker, session_id, sess.api_key)


def _start_phase2_bg(session_id: str, sess) -> None:
    existing = tasks.get_task(session_id)
    if existing and existing.status == "running":
        return
    tasks.run_in_background(session_id, _phase2_worker, session_id, sess.api_key)


# ── Phase 1 worker: filter → embed → UMAP ─────────────────────────────────────

def _phase1_worker(session_id: str, api_key: str) -> None:
    try:
        from openai import OpenAI
        from core.embedder import get_embeddings
        from core.umap_runner import run_umap
        from core.llm import classify_batch
        from config import (
            MIN_WORD_COUNT, MIN_CHAR_COUNT,
            LLM_FILTER_BATCH_SIZE, LLM_FILTER_CHAR_THRESHOLD,
            LLM_FILTER_CHAR_PERCENTILE, LLM_FILTER_CHAR_MIN,
            LLM_FILTER_CHAR_MAX, LLM_FILTER_MIN_SAMPLES,
        )

        client = OpenAI(api_key=api_key)
        sess   = get_session(session_id)
        embedding_model = sess.embedding_model or EMBEDDING_MODEL

        # Structural filter
        tasks.update_progress(session_id, 5, "Structural filter…")
        points = get_points(session_id)
        low_ids = [
            p.id for p in points
            if len((p.response_text or "").split()) < MIN_WORD_COUNT
            or len(p.response_text or "") < MIN_CHAR_COUNT
        ]
        if low_ids:
            mark_points_status(session_id, low_ids, "low_info_structural")
        tasks.update_progress(session_id, 12, f"Structural filter: removed {len(low_ids)} responses")
        tasks.mark_step_done(session_id, "Structural filter", f"{len(low_ids)} removed")

        # LLM filter — re-query so structural exclusions are reflected in status
        # Only send short/ambiguous responses below the dynamic threshold to LLM;
        # anything longer is assumed substantive.
        tasks.update_progress(session_id, 15, "LLM quality filter…")
        active = [p for p in get_points(session_id) if p.status == "active"]
        non_empty_lengths = [
            len((p.response_text or "").strip())
            for p in active
            if (p.response_text or "").strip()
        ]
        llm_char_threshold = _compute_llm_filter_threshold(
            lengths=non_empty_lengths,
            fallback=LLM_FILTER_CHAR_THRESHOLD,
            percentile=LLM_FILTER_CHAR_PERCENTILE,
            min_chars=LLM_FILTER_CHAR_MIN,
            max_chars=LLM_FILTER_CHAR_MAX,
            min_samples=LLM_FILTER_MIN_SAMPLES,
        )
        llm_candidates = [
            p for p in active
            if len((p.response_text or "").strip()) < llm_char_threshold
        ]
        llm_candidate_count = len(llm_candidates)
        texts  = [p.response_text for p in llm_candidates]
        try:
            llm_low = []
            for i in range(0, len(texts), LLM_FILTER_BATCH_SIZE):
                bt = texts[i: i + LLM_FILTER_BATCH_SIZE]
                bp = llm_candidates[i: i + LLM_FILTER_BATCH_SIZE]
                pct = 15 + int(15 * i / max(len(texts), 1))
                tasks.update_progress(session_id, pct,
                                      f"LLM filter batch {i // LLM_FILTER_BATCH_SIZE + 1}…")
                for pt, lbl in zip(bp, classify_batch(client, bt, question=sess.response_col or "")):
                    if lbl == "low_info":
                        llm_low.append(pt.id)
            if llm_low:
                mark_points_status(session_id, llm_low, "low_info_llm")
            tasks.update_progress(
                session_id,
                32,
                f"LLM filter: threshold {llm_char_threshold} chars, sent {llm_candidate_count} responses, removed {len(llm_low)} responses",
            )
            tasks.mark_step_done(
                session_id,
                "LLM filter",
                f"threshold {llm_char_threshold} chars; sent {llm_candidate_count}; {len(llm_low)} removed",
            )
        except Exception as exc:
            import traceback
            tasks.update_progress(session_id, 32,
                                  f"LLM filter skipped ({type(exc).__name__}: {exc})\n{traceback.format_exc()}")
            tasks.mark_step_done(session_id, "LLM filter", f"skipped ({type(exc).__name__})")

        # Embed
        embed_pts   = [p for p in get_points(session_id) if p.status == "active"]
        embed_texts = [p.response_text for p in embed_pts]
        tasks.update_progress(session_id, 35, "Generating embeddings…")

        def _embed_prog(done, total, message=""):
            tasks.update_progress(session_id, 35 + int(25 * done / max(total, 1)),
                                  message or f"Embedding {done}/{total}…")

        embeddings = get_embeddings(
            client,
            embed_texts,
            progress_cb=_embed_prog,
            model=embedding_model,
        )
        tasks.update_progress(session_id, 62, f"Embeddings done ({len(embeddings):,} vectors, {embedding_model})")
        tasks.mark_step_done(session_id, "Embeddings", f"{len(embeddings):,} vectors via {embedding_model}")

        # UMAP
        tasks.update_progress(session_id, 65, "Running UMAP…")

        def _umap_prog(step, total, message=""):
            pct = 65 + int(25 * step / max(total, 1))
            tasks.update_progress(session_id, pct, message or f"UMAP step {step}/{total}…")

        umap_high, umap_3d = run_umap(embeddings, progress_cb=_umap_prog)
        tasks.update_progress(session_id, 90, "UMAP complete")
        tasks.mark_step_done(session_id, "UMAP", f"{len(embed_pts)} responses ready for clustering")

        # Save cache
        tasks.update_progress(session_id, 92, "Saving cache…")
        sess = get_session(session_id)
        cache_save(sess.csv_hash, {
            "embeddings": embeddings,
            "umap_high":  umap_high,
            "umap_3d":    umap_3d,
            "point_ids":  [p.id for p in embed_pts],
        }, embedding_model=embedding_model)
        advance_phase(session_id, 1)
        tasks.mark_done(session_id, {"phase": 1})

    except Exception as exc:
        tasks.mark_error(session_id, str(exc))


# ── Phase 2 worker: HDBSCAN → LLM label ──────────────────────────────────────

def _phase2_worker(session_id: str, api_key: str) -> None:
    try:
        import numpy as np
        from openai import OpenAI
        from core.clusterer import (
            choose_hdbscan_params, run_hdbscan, extract_representatives, build_base_cluster_list,
        )
        from core.llm import summarise_all_clusters, summarise_cluster, suggest_merges

        client = OpenAI(api_key=api_key)
        sess   = get_session(session_id)
        embedding_model = sess.embedding_model or EMBEDDING_MODEL
        arrays = cache_get(sess.csv_hash, embedding_model)
        if arrays is None:
            raise RuntimeError("Cache not found — re-run Phase 1 first")

        # Filter cache arrays to only currently-active points.
        # On first run all points are active so the mask is all-True.
        # After recluster some are low_info_user, so we must exclude them.
        all_points    = get_points(session_id)
        active_pts    = [p for p in all_points if p.status == "active"]
        active_id_set = {p.id for p in active_pts}
        point_ids_all = list(arrays["point_ids"])
        active_mask   = np.array([pid in active_id_set for pid in point_ids_all], dtype=bool)
        # Keep active_pts_ordered aligned with the filtered cache positions
        pid_to_point      = {p.id: p for p in active_pts}
        active_pts_ordered = [pid_to_point[pid] for pid in point_ids_all if pid in active_id_set]
        umap_high  = arrays["umap_high"][active_mask]
        embeddings = arrays["embeddings"][active_mask]

        idx_to_text = {i: p.response_text for i, p in enumerate(active_pts_ordered)}
        total       = len(active_pts_ordered)
        min_cluster_size, min_samples = choose_hdbscan_params(total)

        tasks.update_progress(
            session_id,
            10,
            f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})…",
        )
        clusterer, labels = run_hdbscan(
            umap_high,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        unique_clusters    = [c for c in sorted(set(labels)) if c != -1]
        n_hdbscan_clusters = len(unique_clusters)
        n_raw_outliers     = int(np.sum(labels == -1))
        tasks.mark_step_done(
            session_id,
            "HDBSCAN",
            (
                f"min_cluster_size={min_cluster_size}, min_samples={min_samples}; "
                f"{n_hdbscan_clusters} clusters, {n_raw_outliers} other-themes points"
            ),
        )

        tasks.update_progress(session_id, 30, "Extracting representatives…")
        reps = extract_representatives(clusterer, labels, umap_high,
                                       embeddings, unique_clusters)

        # ── LLM-driven auto-merge (before summarisation) ──────────────────────
        tasks.update_progress(session_id, 38, "LLM merge analysis…")
        reps_texts = {
            cid: [idx_to_text[j] for j in reps.get(cid, [])[:5] if j in idx_to_text]
            for cid in unique_clusters if cid != -1
        }
        counts = {int(cid): int((labels == cid).sum()) for cid in unique_clusters}
        try:
            merge_result = suggest_merges(
                client,
                sess.response_col or "",
                reps_texts,
                counts,
                total,
            )
            suggested    = merge_result.get("merges", [])
        except Exception:
            suggested = []

        n_before_merge  = len(unique_clusters)
        final_merge_map = {}   # raw_cid → canonical_cid (empty if no merges)
        if suggested:
            # Use union-find to consolidate overlapping merge groups correctly.
            parent: dict[int, int] = {}

            def _find(x: int) -> int:
                while parent.get(x, x) != x:
                    x = parent[x]
                return x

            for m in suggested:
                ids = [int(x) for x in m["cluster_ids"] if int(x) != -1]
                if len(ids) < 2:
                    continue
                for x in ids:
                    parent.setdefault(x, x)
                root = _find(ids[0])
                for x in ids[1:]:
                    rx = _find(x)
                    if rx != root:
                        if root < rx:
                            parent[rx] = root
                        else:
                            parent[root] = rx
                            root = rx

            final_candidate = {x: _find(x) for x in parent}
            if any(x != c for x, c in final_candidate.items()):
                final_merge_map = final_candidate

        canonical_labels = np.array(
            [final_merge_map.get(int(l), int(l)) if int(l) != -1 else -1 for l in labels],
            dtype=np.int32,
        )
        canonical_cluster_ids = [int(cid) for cid in sorted(set(canonical_labels)) if int(cid) != -1]
        labels = canonical_labels
        base_list = build_base_cluster_list(labels, canonical_cluster_ids)

        n_absorbed = n_before_merge - len(base_list)
        if n_absorbed > 0:
            tasks.mark_step_done(session_id, "Auto-merge",
                                 f"{n_absorbed} clusters merged, {len(base_list)} remain")
        else:
            tasks.mark_step_done(session_id, "Auto-merge", "no merges needed")

        n_assigned = int(np.sum(labels != -1))
        n_outliers_final = int(np.sum(labels == -1))
        tasks.mark_step_done(
            session_id,
            "Working state",
            f"{n_assigned} assigned, {n_outliers_final} other-themes points, {len(base_list)} active clusters",
        )

        # ── LLM labelling of (merged) clusters — single batched call ─────────
        import random as _random
        tasks.update_progress(session_id, 42, "LLM labelling clusters…")
        cluster_inputs = []
        for info in base_list:
            cid = info["cluster_id"]
            is_merged = final_merge_map and any(
                final_merge_map.get(oc, oc) == cid and oc != cid
                for oc in unique_clusters
            )
            if is_merged:
                # Pool raw-cluster representatives, then sample 10 from that pool.
                # This keeps the merged summary grounded in the source clusters
                # while avoiding an oversized prompt.
                source_rep_idxs = []
                for oc in unique_clusters:
                    if final_merge_map.get(oc, oc) == cid:
                        source_rep_idxs.extend(reps.get(oc, []))
                pooled_rep_idxs = [j for j in source_rep_idxs if j in idx_to_text]
                sampled_rep_idxs = _random.sample(
                    pooled_rep_idxs,
                    min(10, len(pooled_rep_idxs)),
                )
                rep_texts = [idx_to_text[j] for j in sampled_rep_idxs]
            else:
                rep_texts = [idx_to_text[j] for j in reps.get(cid, []) if j in idx_to_text]
            cluster_inputs.append({"cluster_id": cid, "n_points": info["n_points"], "rep_texts": rep_texts})

        tasks.update_progress(session_id, 70, f"Labelling {len(cluster_inputs)} clusters in one call…")
        summaries_map = summarise_all_clusters(
            client, cluster_inputs, total=total, question=sess.response_col or ""
        )
        labelled = [
            {**info, **summaries_map.get(info["cluster_id"], {"title": f"Cluster {info['cluster_id']}", "description": ""})}
            for info in base_list
        ]

        tasks.update_progress(session_id, 92, "Saving to database…")
        save_clusters(session_id, labelled)
        assignments = [(pt.id, int(lbl)) for pt, lbl in zip(active_pts_ordered, labels)]
        save_cluster_assignments(session_id, assignments)

        tasks.mark_step_done(session_id, "Clusters",
                             f"{len(labelled)} clusters, {n_outliers_final} other-themes points")
        advance_phase(session_id, 2)
        tasks.mark_done(session_id, {"phase": 2})

    except Exception as exc:
        import traceback
        tasks.mark_error(session_id, f"{exc}\n{traceback.format_exc()}")
