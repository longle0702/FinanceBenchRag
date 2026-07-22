"""Low-level PDF layout access shared by narrative/glossary/table extraction:
page block/span reading, repeated header/footer detection, section/heading
tracking, and table-region bboxes to exclude from the narrative stream.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from .config import ExtractionConfig

ITEM_HEADING_RE = re.compile(r"^item\s+(\d+[a-z]?)\.?\s*[-–—]?\s*(.*)$", re.IGNORECASE)
TRAILING_PAGE_NUM_RE = re.compile(r"\s*\d+\s*$")


@dataclass
class Span:
    text: str
    size: float
    bold: bool
    bbox: tuple[float, float, float, float]


@dataclass
class Line:
    text: str
    spans: list[Span]
    bbox: tuple[float, float, float, float]

    @property
    def size(self) -> float:
        return max((s.size for s in self.spans), default=0.0)

    @property
    def bold(self) -> bool:
        return any(s.bold for s in self.spans)


@dataclass
class Block:
    lines: list[Line]
    bbox: tuple[float, float, float, float]
    page_num: int

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text.strip())


def open_document(path: Path) -> fitz.Document:
    return fitz.open(path)


def get_page_blocks(page: fitz.Page) -> list[Block]:
    raw = page.get_text("dict")
    blocks: list[Block] = []
    for b in raw.get("blocks", []):
        if b.get("type") != 0:  # skip image blocks
            continue
        lines: list[Line] = []
        for ln in b.get("lines", []):
            spans = [
                Span(
                    text=s["text"],
                    size=s["size"],
                    bold=bool(s["flags"] & 2 ** 4),
                    bbox=tuple(s["bbox"]),
                )
                for s in ln.get("spans", [])
                if s["text"].strip()
            ]
            if not spans:
                continue
            line_text = "".join(s.text for s in spans).strip()
            if line_text:
                lines.append(Line(text=line_text, spans=spans, bbox=tuple(ln["bbox"])))
        if lines:
            blocks.append(Block(lines=lines, bbox=tuple(b["bbox"]), page_num=page.number))
    return blocks


def normalize_boilerplate_candidate(text: str) -> str:
    text = TRAILING_PAGE_NUM_RE.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip().lower()


def compute_boilerplate_lines(doc: fitz.Document, config: ExtractionConfig) -> set[str]:
    """Lines in the top/bottom margin band that recur across most pages."""
    counts: dict[str, int] = {}
    n_pages = doc.page_count
    for page in doc:
        height = page.rect.height
        for block in get_page_blocks(page):
            for line in block.lines:
                y0 = line.bbox[1]
                in_margin = y0 <= config.boilerplate_margin_pt or y0 >= height - config.boilerplate_margin_pt
                if not in_margin:
                    continue
                key = normalize_boilerplate_candidate(line.text)
                if key:
                    counts[key] = counts.get(key, 0) + 1
    threshold = max(2, int(n_pages * config.boilerplate_recurrence_ratio))
    return {key for key, count in counts.items() if count >= threshold}


def is_boilerplate(line: Line, boilerplate: set[str]) -> bool:
    return normalize_boilerplate_candidate(line.text) in boilerplate


def compute_median_body_font_size(doc: fitz.Document, sample_pages: int = 20) -> float:
    sizes: list[float] = []
    for page in list(doc)[:sample_pages]:
        for block in get_page_blocks(page):
            for line in block.lines:
                sizes.append(line.size)
    return statistics.median(sizes) if sizes else 10.0


def normalize_section_id(title: str) -> str:
    m = ITEM_HEADING_RE.match(title.strip())
    if m:
        return f"item_{m.group(1).lower()}"
    slug = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")
    return slug[:40] or "section"


def build_toc_section_map(doc: fitz.Document) -> dict[int, tuple[str, str]] | None:
    """Map page_num -> (section_id, section_title) using embedded bookmarks, if present."""
    toc = doc.get_toc(simple=True)
    if not toc:
        return None
    # toc entries: [level, title, page_1_indexed]; keep top-level entries only
    # for section granularity, sorted by page.
    entries = sorted(
        ((page - 1, title) for level, title, page in toc if level == 1 and page >= 1),
        key=lambda e: e[0],
    )
    if not entries:
        return None
    page_map: dict[int, tuple[str, str]] = {}
    for i, (start_page, title) in enumerate(entries):
        end_page = entries[i + 1][0] - 1 if i + 1 < len(entries) else doc.page_count - 1
        section_id = normalize_section_id(title)
        for p in range(start_page, end_page + 1):
            page_map[p] = (section_id, title)
    return page_map


def detect_heading_line(line: Line, median_size: float, config: ExtractionConfig) -> tuple[str, str] | None:
    """Regex + font-size/bold heuristic fallback when no TOC is available."""
    text = line.text.strip()
    m = ITEM_HEADING_RE.match(text)
    if m:
        return normalize_section_id(text), text
    is_heading_style = line.size >= median_size * config.heading_font_size_ratio or line.bold
    if is_heading_style and 3 <= len(text) <= 90 and not text.endswith((".", ",", ";")):
        return normalize_section_id(text), text
    return None


@dataclass
class TableRegion:
    bbox: tuple[float, float, float, float]
    rows: list[list[str | None]]
    page_num: int


def get_page_tables(page: fitz.Page, backend: str, pdfplumber_page: Any = None) -> list[TableRegion]:
    if backend == "pdfplumber":
        if pdfplumber_page is None:
            return []
        regions = []
        for table in pdfplumber_page.find_tables():
            rows = table.extract()
            if rows:
                regions.append(TableRegion(bbox=tuple(table.bbox), rows=rows, page_num=page.number))
        return regions

    # SEC financial tables are almost never ruled (no grid lines), so the "lines"
    # strategy (pymupdf's default) misses them; "text" clusters whitespace-aligned
    # columns instead, which matches these filings' layout.
    finder = page.find_tables(vertical_strategy="text", horizontal_strategy="text")
    regions = []
    for table in finder.tables:
        rows = table.extract()
        if rows:
            regions.append(TableRegion(bbox=tuple(table.bbox), rows=rows, page_num=page.number))
    return regions


def bbox_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    return inter / area_a


def block_in_tables(block: Block, tables: list[TableRegion], config: ExtractionConfig) -> bool:
    return any(bbox_overlap_ratio(block.bbox, t.bbox) > config.table_overlap_ratio for t in tables)


def extractable_char_count(page: fitz.Page) -> int:
    return len(page.get_text("text").strip())


def save_table_crop(
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    out_path: Path,
    dpi: int = 300,
    image_format: str = "png",
    padding_pt: float = 4.0,
) -> None:
    """Render the table's bbox region as a high-resolution image, so a table
    chunk's markdown/HTML text can link back to what the source table actually
    looked like (useful when the kerning-split-cell artifacts described above
    make a row label hard to read)."""
    rect = fitz.Rect(bbox) + (-padding_pt, -padding_pt, padding_pt, padding_pt)
    rect &= page.rect  # clamp to the page so padding can't push past its edges
    pixmap = page.get_pixmap(clip=rect, dpi=dpi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(out_path, output=image_format)
