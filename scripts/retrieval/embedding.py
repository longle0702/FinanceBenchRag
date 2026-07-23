"""Shared embedding model wrapper, used at both index-build time and query time
so the same model always embeds both sides of a similarity search."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_loaded_models: dict[str, SentenceTransformer] = {}


def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    if model_name not in _loaded_models:
        _loaded_models[model_name] = SentenceTransformer(model_name)
    return _loaded_models[model_name]


def encode(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    model = load_model(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # so cosine similarity == inner product (needed for FAISS IndexFlatIP)
        convert_to_numpy=True,
    )
