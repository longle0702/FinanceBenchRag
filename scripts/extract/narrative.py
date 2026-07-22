"""Narrative-text cleaning and section-aware chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from .config import ExtractionConfig
from .pdf_io import Block, Line

_ENCODING = tiktoken.get_encoding("cl100k_base")

HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
WHITESPACE_RE = re.compile(r"[ \t]+")
KERNED_LETTER_RE = re.compile(r"(?<=\w) (?=\w\b)")  # best-effort fix for "Shee t" style CID-font gaps


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def clean_line_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def join_block_paragraphs(block: Block, boilerplate: set[str]) -> list[str]:
    """Split a block's lines into paragraphs via vertical-gap heuristics, de-hyphenating
    line-wrap breaks within a paragraph."""
    from .pdf_io import is_boilerplate

    kept_lines = [line for line in block.lines if not is_boilerplate(line, boilerplate)]
    if not kept_lines:
        return []

    paragraphs: list[list[Line]] = [[kept_lines[0]]]
    line_heights = [max(1.0, line.bbox[3] - line.bbox[1]) for line in kept_lines]
    for i in range(1, len(kept_lines)):
        prev, cur = kept_lines[i - 1], kept_lines[i]
        gap = cur.bbox[1] - prev.bbox[3]
        if gap > 1.5 * line_heights[i]:
            paragraphs.append([])
        paragraphs[-1].append(cur)

    result = []
    for para_lines in paragraphs:
        raw = "\n".join(line.text for line in para_lines)
        text = HYPHEN_BREAK_RE.sub(r"\1\2", raw)
        text = clean_line_text(text.replace("\n", " "))
        if text:
            result.append(text)
    return result


@dataclass
class ParagraphUnit:
    text: str
    page_num: int
    section_id: str | None
    section_title: str | None
    tokens: int


def pack_paragraphs(paragraphs: list[ParagraphUnit], config: ExtractionConfig) -> list[list[ParagraphUnit]]:
    """Greedy sliding-window packer: fills chunks up to chunk_size tokens, carrying
    forward the trailing ~chunk_overlap tokens of context into the next chunk."""
    chunks: list[list[ParagraphUnit]] = []
    current: list[ParagraphUnit] = []
    tokens_so_far = 0

    for para in paragraphs:
        if current and tokens_so_far + para.tokens > config.chunk_size:
            chunks.append(current)
            overlap_paras: list[ParagraphUnit] = []
            overlap_tokens = 0
            for p in reversed(current):
                if overlap_tokens + p.tokens > config.chunk_overlap:
                    break
                overlap_paras.insert(0, p)
                overlap_tokens += p.tokens
            current = list(overlap_paras)
            tokens_so_far = overlap_tokens
        current.append(para)
        tokens_so_far += para.tokens

    if current:
        chunks.append(current)
    return chunks
