"""Run-wide extraction parameters, snapshotted per run for reproducibility."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExtractionConfig:
    chunk_size: int = 450
    chunk_overlap: int = 60
    min_chunk_tokens: int = 40  # narrative chunks smaller than this get merged into the previous chunk
    table_backend: str = "pymupdf"  # "pymupdf" | "pdfplumber"
    table_format: str = "markdown"  # "markdown" | "html" | "sentences"
    table_row_group_size: int = 20
    table_images_enabled: bool = True
    table_image_dpi: int = 300
    table_image_format: str = "png"  # "png" | "jpg"
    table_image_padding_pt: float = 4.0
    heading_font_size_ratio: float = 1.15  # heading span size vs. doc's median body font size
    boilerplate_margin_pt: float = 60.0
    boilerplate_recurrence_ratio: float = 0.6
    table_overlap_ratio: float = 0.5
    min_chars_per_page: int = 20  # below this, a page is flagged likely_scanned_or_image

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
