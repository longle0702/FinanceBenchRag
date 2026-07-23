"""FAISS-backed VectorStore — IndexFlatIP over L2-normalized vectors (cosine
similarity via inner product). FAISS has no native metadata storage, so this
backend keeps only a row-position -> chunk_id/doc_name sidecar (`ids.json`);
full content is resolved the same way as the Chroma backend, through
ChunkStore. A per-doc row index supports `where={"doc_name": ...}` by
pre-filtering the search space (searching only that doc's vectors) rather than
post-filtering top_k results, so `top_k` stays meaningful when a filter is active.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .store_base import Hit, VectorStore


class FaissVectorStore(VectorStore):
    def __init__(self, persist_dir: Path, dim: int | None = None):
        self.persist_dir = persist_dir
        self.index_path = persist_dir / "index.faiss"
        self.ids_path = persist_dir / "ids.json"
        self.ids: list[str] = []
        self.doc_names: list[str] = []
        self._doc_row_index: dict[str, list[int]] = {}

        if self.index_path.exists() and self.ids_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            sidecar = json.loads(self.ids_path.read_text(encoding="utf-8"))
            self.ids = sidecar["ids"]
            self.doc_names = sidecar["doc_names"]
            self._rebuild_doc_index()
        else:
            if dim is None:
                raise ValueError("dim is required when creating a new FAISS index")
            self.index = faiss.IndexFlatIP(dim)

    def _rebuild_doc_index(self) -> None:
        self._doc_row_index = {}
        for i, doc_name in enumerate(self.doc_names):
            self._doc_row_index.setdefault(doc_name, []).append(i)

    def add(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        self.index.add(embeddings.astype("float32"))
        self.ids.extend(ids)
        self.doc_names.extend(m["doc_name"] for m in metadatas)
        self._rebuild_doc_index()

    def query(self, embedding: np.ndarray, top_k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        query_vec = embedding.astype("float32").reshape(1, -1)

        if where and "doc_name" in where:
            candidate_rows = self._doc_row_index.get(where["doc_name"], [])
            if not candidate_rows:
                return []
            sub_index = faiss.IndexFlatIP(self.index.d)
            sub_index.add(np.vstack([self.index.reconstruct(i) for i in candidate_rows]))
            scores, local_idx = sub_index.search(query_vec, min(top_k, len(candidate_rows)))
            hits = []
            for score, local_i in zip(scores[0], local_idx[0]):
                if local_i == -1:
                    continue
                row = candidate_rows[local_i]
                hits.append(Hit(chunk_id=self.ids[row], doc_name=self.doc_names[row], score=float(score)))
            return hits

        scores, indices = self.index.search(query_vec, top_k)
        hits = []
        for score, row in zip(scores[0], indices[0]):
            if row == -1:
                continue
            hits.append(Hit(chunk_id=self.ids[row], doc_name=self.doc_names[row], score=float(score)))
        return hits

    def count(self) -> int:
        return self.index.ntotal

    def save(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.ids_path.write_text(
            json.dumps({"ids": self.ids, "doc_names": self.doc_names}), encoding="utf-8"
        )
