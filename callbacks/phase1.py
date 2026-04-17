"""Phase 1 — worker only.  Callbacks moved to phase_controller.py."""
# This file is intentionally left minimal.  The @callback decorators that
# previously lived here caused Dash duplicate-output errors.  They have been
# consolidated into callbacks/phase_controller.py.


# ── Phase 1 background worker ──────────────────────────────────────────────────
# (Kept here for reference only; the live copy used by the app lives in
#  phase_controller.py → _phase1_worker.)

def _phase1_worker(session_id: str, api_key: str):
    """Runs in a daemon thread. Updates task state; saves .npy cache."""
    try:
        from openai import OpenAI
        from db.queries import get_points as _gp, mark_points_status, advance_phase as _ap
        from core.embedder import get_embeddings
        from core.umap_runner import run_umap
        from core.cache import save as cache_save
        from config import MIN_WORD_COUNT, MIN_CHAR_COUNT

        client = OpenAI(api_key=api_key)
        tasks.mark_step_done(session_id, "init", "Initialising…")

        # ── Structural filter ──────────────────────────────────────────────
        tasks.update_progress(session_id, 5, "Applying structural filter…")
        points = _gp(session_id)
        low_info_ids = []
        for pt in points:
            txt = (pt.response_text or "").strip()
            words = txt.split()
            if len(words) < MIN_WORD_COUNT or len(txt) < MIN_CHAR_COUNT:
                low_info_ids.append(pt.id)

        if low_info_ids:
            mark_points_status(session_id, low_info_ids, "low_info_structural")

        tasks.mark_step_done(session_id, "structural_filter",
                             f"Structural filter: removed {len(low_info_ids)} responses")

        # ── LLM filter (classify_batch) ────────────────────────────────────
        tasks.update_progress(session_id, 15, "LLM quality filter…")
        active_points = [p for p in points if p.id not in set(low_info_ids)
                         and p.status == "active"]
        texts = [p.response_text for p in active_points]

        try:
            from core.llm import classify_batch
            from config import LLM_FILTER_BATCH_SIZE

            llm_low_ids = []
            for i in range(0, len(texts), LLM_FILTER_BATCH_SIZE):
                batch_texts = texts[i: i + LLM_FILTER_BATCH_SIZE]
                batch_pts   = active_points[i: i + LLM_FILTER_BATCH_SIZE]
                pct = 15 + int(15 * (i / max(len(texts), 1)))
                tasks.update_progress(session_id, pct,
                                      f"LLM filter batch {i // LLM_FILTER_BATCH_SIZE + 1}…")
                labels = classify_batch(client, batch_texts)
                for pt, lbl in zip(batch_pts, labels):
                    if lbl == "low_info":
                        llm_low_ids.append(pt.id)
            if llm_low_ids:
                mark_points_status(session_id, llm_low_ids, "low_info_llm")
            tasks.mark_step_done(session_id, "llm_filter",
                                 f"LLM filter: removed {len(llm_low_ids)} responses")
        except Exception as exc:
            tasks.mark_step_done(session_id, "llm_filter",
                                 f"LLM filter skipped ({exc})")

        # ── Reload points after filtering ──────────────────────────────────
        embed_points = [p for p in _gp(session_id) if p.status == "active"]
        embed_texts  = [p.response_text for p in embed_points]

        # ── Embed ──────────────────────────────────────────────────────────
        tasks.update_progress(session_id, 35, "Generating embeddings…")

        def _embed_prog(done, total):
            pct = 35 + int(25 * (done / max(total, 1)))
            tasks.update_progress(session_id, pct, f"Embedding batch {done}/{total}…")

        embeddings = get_embeddings(client, embed_texts, progress_cb=_embed_prog)
        tasks.mark_step_done(session_id, "embed",
                             f"Embeddings done ({len(embeddings):,} vectors)")

        # ── UMAP ───────────────────────────────────────────────────────────
        tasks.update_progress(session_id, 65, "Running UMAP (100-d + 3-d)…")

        def _umap_prog(stage: str):
            tasks.update_progress(session_id, 75 if stage == "3d" else 68,
                                  f"UMAP {stage}…")

        umap_high, umap_3d = run_umap(embeddings, progress_cb=_umap_prog)
        tasks.mark_step_done(session_id, "umap", "UMAP complete")

        # ── Save cache ─────────────────────────────────────────────────────
        tasks.update_progress(session_id, 92, "Saving cache…")
        sess = get_session(session_id)
        cache_save(sess.csv_hash, {
            "embeddings": embeddings,
            "umap_high":  umap_high,
            "umap_3d":    umap_3d,
            "point_ids":  [p.id for p in embed_points],
        })

        _ap(session_id, 1)
        tasks.mark_done(session_id, {"phase": 1})

    except Exception as exc:
        tasks.mark_error(session_id, str(exc))


# poll_phase1 callback removed — consolidated into phase_controller.py
