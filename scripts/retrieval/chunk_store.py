"""Lazy per-document loader/cache for extracted chunk records.

Vector stores are kept "dumb" (id + vector + minimal filter metadata) — this is
where full chunk content (text, section, structured table data) actually gets
resolved from just a chunk_id, and where neighbor-chunk expansion happens, via
the prev_chunk_id/next_chunk_id pointers already recorded at extraction time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ChunkStore:
    def __init__(self, chunks_dir: Path):
        self.chunks_dir = chunks_dir
        self._doc_cache: dict[str, dict[str, dict[str, Any]]] = {}

    def _load_doc(self, doc_name: str) -> dict[str, dict[str, Any]]:
        if doc_name not in self._doc_cache:
            path = self.chunks_dir / f"{doc_name}.jsonl"
            chunk_map: dict[str, dict[str, Any]] = {}
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            chunk_map[record["chunk_id"]] = record
            self._doc_cache[doc_name] = chunk_map
        return self._doc_cache[doc_name]

    def get(self, doc_name: str, chunk_id: str) -> dict[str, Any] | None:
        return self._load_doc(doc_name).get(chunk_id)

    def get_neighbors(self, doc_name: str, chunk_id: str, window: int) -> list[dict[str, Any]]:
        """Walk prev_chunk_id/next_chunk_id up to `window` hops each direction.
        Returned in document reading order (preceding chunks first, then
        following chunks) — stops cleanly at a doc boundary rather than erroring."""
        doc = self._load_doc(doc_name)
        chunk = doc.get(chunk_id)
        if chunk is None:
            return []

        before: list[dict[str, Any]] = []
        cur = chunk
        for _ in range(window):
            prev_id = cur.get("prev_chunk_id")
            if not prev_id or prev_id not in doc:
                break
            cur = doc[prev_id]
            before.insert(0, cur)

        after: list[dict[str, Any]] = []
        cur = chunk
        for _ in range(window):
            next_id = cur.get("next_chunk_id")
            if not next_id or next_id not in doc:
                break
            cur = doc[next_id]
            after.append(cur)

        return before + after
