"""Run-wide retrieval parameters."""
from __future__ import annotations

from dataclasses import dataclass

from .embedding import DEFAULT_MODEL


@dataclass
class RetrievalConfig:
    embedding_model: str = DEFAULT_MODEL
    top_k: int = 5
    neighbor_window: int = 1
    batch_size: int = 64
