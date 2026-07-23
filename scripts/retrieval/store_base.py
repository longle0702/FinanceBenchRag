"""Common interface for vector-store backends.

Kept intentionally "dumb": add just enough metadata to support `where`
filtering (doc_name, chunk_type, ...), never full chunk content — that's always
resolved afterward through ChunkStore, keyed by chunk_id. This is what lets
FAISS (no native metadata storage) and Chroma (has it) share one interface
without FAISS needing its own content store.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Hit:
    chunk_id: str
    doc_name: str
    score: float


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        """metadatas[i] must include at least 'doc_name'."""

    @abstractmethod
    def query(self, embedding: np.ndarray, top_k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...
