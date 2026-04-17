"""
OpenAI embedding with progress callback.
Cache is handled at a higher level (core/cache.py).
"""

import time
import numpy as np
from typing import Callable, Optional

from openai import OpenAI
from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


def get_embeddings(
    client: OpenAI,
    texts: list[str],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """
    Embed texts in batches.

    progress_cb(current_batch, total_batches, message) — called after each batch.
    Returns float32 array of shape (n_texts, embedding_dim).
    """
    all_embeddings = []
    total_batches  = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        b_num = i // batch_size + 1

        resp = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend(item.embedding for item in resp.data)

        if progress_cb:
            progress_cb(b_num, total_batches, f"Embedding batch {b_num}/{total_batches}")

        if i + batch_size < len(texts):
            time.sleep(0.3)

    return np.array(all_embeddings, dtype=np.float32)
