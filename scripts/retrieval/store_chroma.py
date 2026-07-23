"""ChromaDB-backed VectorStore — stores vectors and their filter metadata
together, so it's the simpler backend to stand up (contrast with FAISS, which
needs a separate id sidecar; see store_faiss.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from .store_base import Hit, VectorStore

_MAX_BATCH = 4000  # stay safely under Chroma's internal max-batch-size limit


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: Path, collection_name: str = "financebench_chunks"):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        # Explicit cosine space so "distance" lines up with the FAISS backend's
        # cosine-via-inner-product score (score = 1 - distance either way).
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        for start in range(0, len(ids), _MAX_BATCH):
            end = start + _MAX_BATCH
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end],
            )

    def query(self, embedding: np.ndarray, top_k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        result = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            where=where,
        )
        hits = []
        for chunk_id, distance, meta in zip(result["ids"][0], result["distances"][0], result["metadatas"][0]):
            hits.append(Hit(chunk_id=chunk_id, doc_name=meta["doc_name"], score=1.0 - distance))
        return hits

    def count(self) -> int:
        return self.collection.count()
