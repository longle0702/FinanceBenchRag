"""Table extraction, cleanup, and "full meaning representation" serialization.

Raw PDF table extraction gives a grid of strings; naive flattening (as plain
text-layer extraction does) collapses row/column structure into a bare number
stream. This module reconstructs each data cell into a self-contained
"row-label, column-header: value" sentence so a table chunk reads meaningfully
on its own, while keeping the raw grid available in `structured` for exact
lookups.

These SEC filings use embedded fonts with irregular inter-character spacing
("kerning gaps"), which causes the whitespace-based column detector to
sometimes split a single word into two grid cells (e.g. "Cash and cash e" /
"quivalents"). Column-header-by-index alignment is therefore unreliable; the
robust part of a filing's tables is the *trailing run of numeric-looking
cells* on each row (the actual reported values), so rows are parsed by
peeling that run off the right-hand side rather than by trusting raw column
positions.
"""
from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass

from .config import ExtractionConfig
from .pdf_io import TableRegion
from .schema import TableData

UNIT_HINT_RE = re.compile(r"in (millions|thousands|billions)(?: of dollars)?", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^\(?-?\$?\s*[\d,]+(?:\.\d+)?\)?%?$")
BARE_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
PREFIX_MERGE_CHARS = {"$", "(", "-$", "-"}


def _is_real_value(cell: str) -> bool:
    """NUMERIC_RE alone is too permissive: kerning gaps in a title/date line like
    'December 31,' can split off a lone digit-plus-comma fragment (e.g. '1,')
    that matches it but isn't an actual reported figure. Require a currency
    sign, a decimal, parenthesized-negative formatting, or >=2 digits."""
    if not NUMERIC_RE.match(cell):
        return False
    if "$" in cell or "." in cell or cell.startswith("("):
        return True
    return len(re.sub(r"\D", "", cell)) >= 2


def _is_header_row(values: list[str]) -> bool:
    """A row whose entire value-run is bare 4-digit years (e.g. "2018", "2017")
    is a column-header row, not a data row — used to (re)set the current column
    headers as we scan down the table, regardless of table size or how many
    boilerplate/title lines precede it."""
    return bool(values) and all(BARE_YEAR_RE.match(v) for v in values)


def find_table_title_and_units(preceding_text: str | None) -> tuple[str | None, str | None]:
    if not preceding_text:
        return None, None
    candidate = preceding_text.strip().splitlines()[-1].strip() if preceding_text.strip() else None
    title = candidate if candidate and len(candidate) <= 100 else None
    unit_match = UNIT_HINT_RE.search(preceding_text)
    unit_hint = f"in {unit_match.group(1).lower()}" if unit_match else None
    return title, unit_hint


def parse_numeric(raw: str) -> float | None:
    text = raw.strip()
    if not text or text in {"-", "—", "–"}:
        return None
    # A trailing ")" is sometimes dropped by the text extractor (e.g. "(16,048"
    # with no closing paren), so a leading "(" alone is treated as the negative
    # marker rather than requiring both.
    negative = text.startswith("(")
    stripped = text.strip("()").replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        value = float(stripped)
    except ValueError:
        return None
    return -value if negative else value


def _merge_prefix_cells(cells: list[str]) -> list[str]:
    """Join a lone '$'/'(' cell with the value cell that follows it (currency and
    negative-parenthesis markers sometimes land in their own grid cell)."""
    merged: list[str] = []
    i = 0
    while i < len(cells):
        cell = cells[i]
        if cell in PREFIX_MERGE_CHARS and i + 1 < len(cells) and cells[i + 1]:
            merged.append(cell + cells[i + 1])
            i += 2
        else:
            merged.append(cell)
            i += 1
    return merged


def split_label_and_values(row: list[str | None]) -> tuple[str, list[str]]:
    """Peel a trailing run of numeric-looking cells off the row; everything to
    the left (however many grid cells it was split into) is the row label."""
    cells = [c.strip() for c in row if isinstance(c, str) and c.strip()]
    cells = _merge_prefix_cells(cells)

    values_rev: list[str] = []
    cut = len(cells)
    for cell in reversed(cells):
        if _is_real_value(cell):
            values_rev.append(cell)
            cut -= 1
        else:
            break
    values = list(reversed(values_rev))
    label = " ".join(cells[:cut]).strip()
    return label, values


def is_plausible_table(region: TableRegion, min_numeric_ratio: float = 0.2) -> bool:
    """The "text" strategy needed to detect whitespace-aligned financial tables
    (see module docstring) also fires on ordinary prose — any page of narrative
    text has whitespace-separated "cells" too. Require a meaningful fraction of
    the region's non-blank rows to actually carry a numeric value before
    treating it as a real table; a real financial table is numeric-dense, a
    paragraph of prose misdetected as a "table" is not."""
    if len(region.rows) < 2:
        return False
    parsed = [split_label_and_values(row) for row in region.rows]
    non_blank_rows = sum(1 for label, values in parsed if label or values)
    if non_blank_rows == 0:
        return False
    rows_with_values = sum(1 for _, values in parsed if values)
    return (rows_with_values / non_blank_rows) >= min_numeric_ratio


def _canonical_columns(cell_records: list[dict]) -> list[str]:
    """Most rows share one column-header tuple (the table's real header); a
    handful of rows can end up with a mismatched tuple due to parsing edge
    cases (see split_label_and_values docstring) — use the majority tuple as
    the grid's column order rather than whatever the first row happened to
    have."""
    by_row: dict[str, tuple[str, ...]] = {}
    for r in cell_records:
        by_row.setdefault(r["row_label"], tuple())
    row_headers: dict[str, list[str]] = {}
    for r in cell_records:
        row_headers.setdefault(r["row_label"], []).append(r["column_header"])
    tuples = [tuple(v) for v in row_headers.values()]
    return list(Counter(tuples).most_common(1)[0][0]) if tuples else []


def _rows_for_grid(cell_records: list[dict], columns: list[str]) -> list[tuple[str, list[str]]]:
    by_row: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for r in cell_records:
        if r["row_label"] not in by_row:
            by_row[r["row_label"]] = {}
            order.append(r["row_label"])
        by_row[r["row_label"]][r["column_header"]] = r["raw_value"]
    return [(label, [by_row[label].get(col, "") for col in columns]) for label in order]


def _to_markdown_table(columns: list[str], rows: list[tuple[str, list[str]]], header_prefix: str, context: str) -> str:
    def esc(cell: str) -> str:
        return cell.replace("|", r"\|")

    header = "| Line Item | " + " | ".join(columns) + " |"
    sep = "| --- | " + " | ".join("---" for _ in columns) + " |"
    body = [f"| {esc(label)} | " + " | ".join(esc(v) for v in values) + " |" for label, values in rows]
    return f"{header_prefix} {context}\n\n" + "\n".join([header, sep, *body])


def _to_html_table(columns: list[str], rows: list[tuple[str, list[str]]], header_prefix: str, context: str) -> str:
    head_cells = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body_rows = "".join(
        "<tr><td>" + html.escape(label) + "</td>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>"
        for label, values in rows
    )
    return (
        f"<p>{html.escape(header_prefix)} {html.escape(context)}</p>\n"
        f"<table>\n<thead><tr><th>Line Item</th>{head_cells}</tr></thead>\n"
        f"<tbody>{body_rows}</tbody>\n</table>"
    )


def _to_sentences(rows: list[tuple[str, list[str]]], columns: list[str], header_prefix: str, context: str) -> str:
    lines = [f"{label}, {col}: {value}" for label, values in rows for col, value in zip(columns, values) if value]
    return f"{header_prefix} {context}\n" + "\n".join(lines)


@dataclass
class TableChunkDraft:
    text: str
    structured: TableData
    page_num: int


def build_table_chunks(
    region: TableRegion,
    preceding_text: str | None,
    table_id: str,
    company: str,
    doc_period: str,
    config: ExtractionConfig,
    image_path: str | None = None,
) -> list[TableChunkDraft]:
    title, unit_hint = find_table_title_and_units(preceding_text)
    header_prefix = title or "Table"
    context = f"({company}, {doc_period}{', ' + unit_hint if unit_hint else ''})"

    # Single top-to-bottom scan: a row whose values are all bare years (re)sets
    # the running column headers; a row with a real label and values is data,
    # emitted against whichever header was most recently seen (never index-based,
    # so this works the same for a 3-row note table and a 60-row statement).
    current_header: list[str] = []
    cell_records = []
    for row in region.rows:
        label, values = split_label_and_values(row)
        if _is_header_row(values):
            current_header = values
            continue
        if not label or not values:
            continue  # blank row or a subheading with no values (e.g. "Assets")
        columns = current_header if len(current_header) == len(values) else [f"Col{i + 1}" for i in range(len(values))]
        for column_header, raw_value in zip(columns, values):
            cell_records.append({
                "row_label": label,
                "column_header": column_header,
                "raw_value": raw_value,
                "value": parse_numeric(raw_value),
            })

    if not cell_records:
        return []

    columns = _canonical_columns(cell_records)
    serializer = {
        "markdown": _to_markdown_table,
        "html": _to_html_table,
        "sentences": lambda cols, rows, hp, ctx: _to_sentences(rows, cols, hp, ctx),
    }[config.table_format]

    chunks: list[TableChunkDraft] = []
    group_size = config.table_row_group_size
    unique_row_labels = list(dict.fromkeys(r["row_label"] for r in cell_records))
    for group_start in range(0, len(unique_row_labels), group_size):
        group_labels = set(unique_row_labels[group_start:group_start + group_size])
        group_cells = [r for r in cell_records if r["row_label"] in group_labels]
        if not group_cells:
            continue
        grid_rows = _rows_for_grid(group_cells, columns)
        text = serializer(columns, grid_rows, header_prefix, context)
        chunks.append(TableChunkDraft(
            text=text,
            structured=TableData(
                table_id=table_id,
                row_range=(group_start, group_start + len(group_labels) - 1),
                table_title=title,
                unit_hint=unit_hint,
                rows=group_cells,
                table_image=image_path,
            ),
            page_num=region.page_num,
        ))
    return chunks
