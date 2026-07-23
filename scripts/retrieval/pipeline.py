"""Ties embedding + vector search + neighbor expansion + glossary lookup +
table-image attachment together into one retrieval call."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import embedding
from .chunk_store import ChunkStore
from .config import RetrievalConfig
from .glossary_lookup import GlossaryLookup
from .store_base import VectorStore


@dataclass
class RetrievedHit:
    chunk: dict[str, Any]
    score: float
    neighbors: list[dict[str, Any]] = field(default_factory=list)
    glossary_matches: list[dict[str, Any]] = field(default_factory=list)
    table_image: str | None = None


@dataclass
class RetrievalResult:
    query: str
    hits: list[RetrievedHit]


class RetrievalPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        chunk_store: ChunkStore,
        glossary_lookup: GlossaryLookup,
        config: RetrievalConfig,
    ):
        self.vector_store = vector_store
        self.chunk_store = chunk_store
        self.glossary_lookup = glossary_lookup
        self.config = config

    def search(self, query: str, top_k: int | None = None, doc_name: str | None = None) -> RetrievalResult:
        top_k = top_k or self.config.top_k
        query_embedding = embedding.encode(
            [query], model_name=self.config.embedding_model, show_progress=False,
        )[0]
        where = {"doc_name": doc_name} if doc_name else None
        vector_hits = self.vector_store.query(query_embedding, top_k, where=where)

        hits = []
        for vh in vector_hits:
            chunk = self.chunk_store.get(vh.doc_name, vh.chunk_id)
            if chunk is None:
                continue
            neighbors = self.chunk_store.get_neighbors(vh.doc_name, vh.chunk_id, self.config.neighbor_window)
            glossary_text = chunk["text"] + "\n" + "\n".join(n["text"] for n in neighbors)
            glossary_matches = self.glossary_lookup.expand_terms(glossary_text, vh.doc_name)
            table_image = None
            if chunk.get("chunk_type") == "table" and chunk.get("structured"):
                table_image = chunk["structured"].get("table_image")
            hits.append(RetrievedHit(
                chunk=chunk, score=vh.score, neighbors=neighbors,
                glossary_matches=glossary_matches, table_image=table_image,
            ))
        return RetrievalResult(query=query, hits=hits)
