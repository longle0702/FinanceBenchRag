"""Shared data model for extracted chunks and glossary entries, plus JSONL I/O."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

ChunkType = Literal["narrative", "table"]


@dataclass
class TableData:
    table_id: str
    row_range: tuple[int, int]
    table_title: str | None
    unit_hint: str | None
    rows: list[dict[str, Any]]
    table_image: str | None = None  # path (repo-root-relative) to a cropped image of the source table


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    doc_type: str
    company: str
    doc_period: str
    chunk_type: ChunkType
    section: str | None
    section_id: str | None
    page_start: int
    page_end: int
    chunk_index: int
    n_tokens: int
    text: str
    structured: TableData | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    source: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        if self.structured is not None:
            d["structured"] = asdict(self.structured)
        return d


@dataclass
class GlossaryEntry:
    term: str
    definition: str
    doc_name: str
    page_num: int
    section: str | None
    pattern_type: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def link_neighbors(chunks: list[Chunk]) -> None:
    """Set prev_chunk_id/next_chunk_id in place, in doc order (by chunk_index)."""
    chunks.sort(key=lambda c: c.chunk_index)
    for i, chunk in enumerate(chunks):
        chunk.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
        chunk.next_chunk_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
