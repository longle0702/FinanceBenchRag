# PDF Text Extraction Pipeline (Step 2: Corpus Processing)

## Configuration reference

Every parameter below is a field on `ExtractionConfig` ([scripts/extract/config.py](scripts/extract/config.py))
unless marked "CLI-only" or "constant". A full snapshot of the config actually used is
written to `data/processed/extraction_config.json` on every run, for reproducibility.

**Run control (CLI-only, not part of `ExtractionConfig`)**

| Flag | Default | Meaning |
|---|---|---|
| `--catalog` | `data/financebench_document_information.jsonl` | Source catalog (doc_name, doc_link, company, doc_type, doc_period) |
| `--pdf-dir` | `data/raw_pdfs/` | Where downloaded PDFs are read from |
| `--output-dir` | `data/processed/` | Where chunks/glossary/images/reports are written |
| `--workers` | `4` | Parallel worker **processes** (CPU-bound parsing → `ProcessPoolExecutor`, not threads) |
| `--limit` | none | Only process the first N catalog docs (testing) |
| `--doc-name` | none | Only process this one doc (testing) |
| `--overwrite` | off | Reprocess docs that already have output chunks |
| `--report-path` | `<output-dir>/extraction_report.csv` | Where the CSV status report is written |

**Chunking (narrative text)**

| Parameter | CLI flag | Default | Meaning |
|---|---|---|---|
| `chunk_size` | `--chunk-size` | `450` (tokens, `tiktoken` cl100k_base) | Target size of a narrative chunk before a new one starts |
| `chunk_overlap` | `--chunk-overlap` | `60` (tokens) | Trailing context carried into the next chunk (sliding window) |

**Table extraction & serialization**

| Parameter | CLI flag | Default | Meaning |
|---|---|---|---|
| `table_backend` | `--table-backend` | `pymupdf` (or `pdfplumber`) | Which library detects table grids |
| `table_format` | `--table-format` | `markdown` (or `html`, `sentences`) | How a table's cells are serialized into chunk `text` |
| `table_row_group_size` | — | `20` (unique row labels) | Large tables split into multiple chunks of this many rows each |
| `table_images_enabled` | `--no-table-images` (disables) | `True` | Whether to crop+save an image of each detected table region |
| `table_image_format` | `--table-image-format` | `png` (or `jpg`) | Cropped table image file format |
| `table_image_dpi` | `--table-image-dpi` | `300` | Resolution of cropped table images |
| `table_image_padding_pt` | — | `4.0` (PDF points) | Padding added around a table's bbox before cropping |
| `min_numeric_ratio` (constant, in `tables.is_plausible_table`, not yet CLI-exposed) | — | `0.2` | Fraction of a candidate table region's rows that must carry a numeric value before it's treated as a real table rather than prose misdetected by the "text" strategy |

**Detection thresholds (boilerplate / headings / table regions / scanned pages)**

| Parameter | CLI flag | Default | Meaning |
|---|---|---|---|
| `boilerplate_margin_pt` | — | `60.0` (PDF points) | Top/bottom page-margin band scanned for repeated header/footer lines |
| `boilerplate_recurrence_ratio` | — | `0.6` | A margin line recurring on ≥60% of pages is treated as boilerplate and stripped |
| `heading_font_size_ratio` | — | `1.15` | A line's font size must be ≥1.15× the doc's median body size (or bold) to be treated as a heading, when no PDF bookmark/TOC is available |
| `table_overlap_ratio` | — | `0.5` | A text block overlapping a detected table's bbox by more than this fraction is excluded from the narrative stream |
| `min_chars_per_page` | — | `20` | Below this average extractable-character count (sampled over first 5 pages), a doc is flagged `likely_scanned_or_image` and skipped |

None of the "—" (no CLI flag) parameters are exposed on the command line yet; change them by
editing the `ExtractionConfig` defaults in `config.py` if an experiment needs it.

## Context

This is Step 2 of the FinanceBenchRag pipeline (see [README.md](README.md)). Step 1
(`scripts/download_pdfs.py`) already pulled 287 SEC filing PDFs (10-K/10-Q/8-K/earnings
releases, 35 companies) into `data/raw_pdfs/`; 73 links failed to download (dead/blocked
IR domains) and must be tolerated, not fixed here.

The assignment (`Graded_project_instructions_A_RAG.pdf`) requires this stage to do
"text extraction... data cleaning... document segmentation (chunking)... metadata
management," explicitly calling out **tables and financial layouts** as something the
parsing/chunking strategy must handle, and requires chunk size/overlap/structure-
preservation to be experimentally justified later. Sampling `3M_2018_10K.pdf` confirmed
these are text-native EDGAR/IR filings (no OCR needed) with three distinct content
shapes that need different treatment:

1. **Narrative prose** (Business, Risk Factors, MD&A, legal boilerplate) — standard
   semantic chunking is fine.
2. **Defined terms / abbreviations** ("PP&E", "GAAP", quoted-term definitions) —
   flattening these into prose loses the ability to later expand an abbreviation to its
   full meaning on demand, so they need their own structured lookup store in addition to
   being retrievable as text.
3. **Financial tables** (balance sheet, cash flow, segment/note schedules) — raw text
   extraction of these collapses row/column structure into a flat number stream
   (verified directly against `evidence_text` in `financebench_open_source.jsonl`, e.g.
   a label followed by a vertical stream of bare numbers). These need structure-aware
   extraction and a serialized "row-label + column-header + value" representation so
   each table chunk is meaningful in isolation.

The goal of this plan is **only the extraction/chunking/metadata stage** — not
embeddings, vector store, retrieval, or generation. But the metadata schema is designed
so the next phase can do two things without touching this stage's code: (a) fetch
neighboring chunks around a match for extra context, and (b) look up full definitions
for abbreviations found in a retrieved chunk or the user's query.

## Library choices

- **PyMuPDF (`pymupdf`, import `fitz`)** — primary extractor. Pure-C wheels (no
  Ghostscript/poppler/tesseract needed on Windows), gives layout info
  (`page.get_text("dict")` → blocks/lines/spans with bbox + font size/bold), reads
  embedded bookmarks (`doc.get_toc()`) which many of these filings have for Item
  boundaries, and has native table detection (`page.find_tables()`).
- **pdfplumber** — secondary/optional table backend (`--table-backend` flag), pure
  Python, mature whitespace-aligned table detection (most SEC tables aren't
  ruled/gridded). Cheap to include and gives a second backend to compare in the report's
  methodology section, which the assignment rewards.
- **tiktoken** — token counting for chunk-size budgeting, independent of whatever
  embedding model gets picked later.
- Explicitly **not** using `camelot` (needs a Ghostscript system install, targets ruled
  tables which these mostly aren't) or `unstructured` (drags in a heavy layout-model
  dependency chain meant for scanned documents; these are text-native, so it buys
  nothing).

`requirements.txt` gets: `pymupdf>=1.24`, `pdfplumber>=0.11`, `tiktoken>=0.7`.

## Architecture: one pass per document, three interleaved output streams

For each PDF, walk pages top-to-bottom once:

1. **Boilerplate detection** (pre-pass over all pages): normalize every text line in the
   top/bottom ~60pt margin band; any line recurring on >60% of pages (e.g. "Table of
   Contents" footer, running company-name header) goes into a per-doc boilerplate set
   and gets stripped from narrative text.
2. **Section/heading tracking**: prefer `doc.get_toc()` mapped to page ranges when the
   PDF has real bookmarks; otherwise fall back to regex `^Item\s+\d+[A-Z]?\.` combined
   with font-size/bold heuristics, updating a `current_section` state as blocks are
   walked in order.
3. **Table-region exclusion**: any text block whose bbox overlaps a detected table's
   bbox by >50% is routed to the table stream, not narrative.
4. **Remaining blocks** = narrative stream for that page, tagged with `current_section`.
5. **Glossary pass** runs as a regex sweep over the cleaned narrative stream text
   (dual-emitted — see below).

All three streams share one monotonically increasing `chunk_index` per document, in
document reading order — this is what makes "give me the neighboring chunks" a simple
`chunk_index ± k` lookup later, regardless of whether the neighbor is narrative or
table.

## Per-type handling

### Narrative text
- Clean: strip boilerplate lines, de-hyphenate line-wrap breaks, collapse whitespace,
  join spans into paragraphs via vertical-gap heuristics.
- Chunk: **section-aware** — an Item/heading boundary is a hard split (a chunk never
  spans two Items); within a section, pack paragraphs up to a configurable token budget
  (`tiktoken`) with configurable overlap. Both are CLI parameters (defaults
  `chunk_size=450`, `overlap=60` tokens) so the report can sweep them per the
  assignment's required ablations.

### Glossary / defined terms
- Regex patterns over narrative sentences:
  - `"X" means/refers to/is defined as/shall mean ...` (quoted defined term)
  - `Long form name ("ABBR")` (parenthetical abbreviation introduction — e.g.
    "Property, plant and equipment (“PP&E”)")
  - A dedicated `Definitions|Glossary|Abbreviations` section, if present, parsed
    line-by-line instead of by sentence regex.
- **Lookup-only storage** (not embedded as a retrievable chunk):
  1. A structured lookup record `{term, definition, doc_name, page_num, section,
     pattern_type}` written to a per-doc glossary file — this is the mechanism
     for "get the full term for a shortened term" at generation time (direct dict
     lookup, no retrieval needed).
  2. A corpus-level rollup (`glossary_global.jsonl`) dedupes identical terms across all
     287 docs (e.g. "GAAP" appears in nearly every filing) for a fast global lookup,
     while the per-doc file stays authoritative (the same abbreviation can mean
     different things in different filings).

  (An earlier version also duplicated each entry into the main chunk stream as a
  `chunk_type="glossary"` chunk, so it would be independently retrievable by vector
  search if a user asked about a term directly. That was dropped by request — chunks
  now contain only `narrative`/`table`; glossary lookups are lookup-file-only.)

### Tables / financial layouts
- Extract via `page.find_tables(vertical_strategy="text", horizontal_strategy="text")`
  (or the pdfplumber backend) → rows + bbox. The "text" strategy is required because
  these tables are whitespace-aligned with no ruling lines — PyMuPDF's default
  "lines" strategy misses nearly all of them (verified: 7 tables found across a whole
  160-page 10-K under "lines" vs. correctly finding the Balance Sheet/Cash Flow/Equity
  statements once switched to "text").
- These filings' embedded fonts have kerning gaps that sometimes split a single word
  into two grid cells (e.g. "Cash and cash e" / "quivalents"), so column-by-index
  alignment is unreliable. Each row is instead parsed by peeling the trailing run of
  numeric-looking cells off the right (the actual reported values) and joining
  whatever remains as the row label — robust regardless of how many stray cells a
  label got split into.
- Column headers are resolved with a single top-to-bottom scan per table: a row whose
  entire value-run is bare 4-digit years (e.g. "2018", "2017") (re)sets the current
  column headers; a row with a real label and values is data, emitted against
  whichever header was most recently seen. This works the same for a 3-row note
  table and a 60-row statement, since it isn't tied to a fixed row index.
- Capture table title and units from the text block immediately preceding the table
  (e.g. "Consolidated Statement of Cash Flows", "in millions").
- **Serialize a "full meaning representation"**: for each row × numeric column, emit a
  reconstructive sentence, e.g. *"Consolidated Statement of Cash Flows (3M, FY2018, in
  millions) — Purchases of property, plant and equipment (PP&E): $(1,577)"*. All rows
  for one table (or one row-group, for large tables) join into a single `text` field
  tagged `chunk_type="table"` — self-contained and embeddable on its own.
- Large tables (segment/quarterly grids, note schedules) split into multiple chunks by
  row-group, each carrying a repeated `table_title` prefix plus a shared `table_id` and
  `row_range`, so the next phase can specifically expand "other rows of this same
  table" in addition to generic doc-order neighbors.
- Raw row/col grid + parsed numeric values are retained in a `structured` field (not
  embedded) for potential exact-value lookups later.

## Metadata schema (shared across all chunk types)

```jsonc
{
  "chunk_id": "3M_2018_10K__p0057__c0042",
  "doc_name": "3M_2018_10K",
  "doc_type": "10k",
  "company": "3M",
  "doc_period": 2018,
  "chunk_type": "narrative" | "table",
  "section": "Item 8. Financial Statements and Supplementary Data",
  "section_id": "item_8",
  "page_start": 57,        // 0-based, matches FinanceBench's evidence_page_num convention (verified)
  "page_end": 57,
  "chunk_index": 42,        // monotonic across the WHOLE doc, all types interleaved in doc order
  "prev_chunk_id": "3M_2018_10K__p0056__c0041",
  "next_chunk_id": "3M_2018_10K__p0058__c0000",
  "n_tokens": 187,
  "text": "<embeddable text>",
  "structured": null,        // table chunks only: {table_id, row_range, table_title, unit_hint, rows: [...], table_image}
  "source": {"extractor": "pymupdf.find_tables", "confidence": null}
}
```

`page_start`/`page_end` uses `page.number` (0-based) directly, confirmed to match
FinanceBench's `evidence_page_num` semantics — important for later citation/evaluation
cross-checks even though evaluation itself is out of scope here.

## Output layout

```
data/processed/
  chunks/<doc_name>.jsonl                       # narrative + table chunks, one file per doc
  table_images/<doc_name>/<table_id>.png        # high-res crop of each detected table region
  glossary/<doc_name>.glossary.jsonl            # per-doc term -> definition lookup
  glossary_global.jsonl                         # corpus-level rollup of normalized terms
  extraction_report.csv                         # doc_name, status, n_pages, n_chunks_*, timing, error (mirrors download_report.csv)
  extraction_config.json                        # snapshot of run params (chunk_size, overlap, table_backend...) for reproducibility
```
Per-doc files (not one giant file) keep the step resumable and let the next phase
simply glob `data/processed/chunks/*.jsonl`.

## Code layout

Mirrors `scripts/download_pdfs.py` conventions: argparse CLI, per-item try/except, CSV
report, tqdm progress.

```
scripts/
  download_pdfs.py            # existing (Step 1)
  extract_corpus.py           # Step 2 CLI entrypoint
  extract/
    __init__.py
    config.py                 # ExtractionConfig dataclass: chunk_size, chunk_overlap, table_backend, heading thresholds
    pdf_io.py                 # per-page blocks/spans, boilerplate detection, section/heading tracking
    narrative.py               # cleaning + section-aware chunking
    glossary.py                 # regex-based term/definition extraction + dedup/rollup
    tables.py                   # table detection (pymupdf/pdfplumber backends), value-run parsing, markdown/html/sentence serialization, image cropping hook
    schema.py                   # Chunk/GlossaryEntry dataclasses, chunk_id generation, JSONL writers
```

CLI, matching `download_pdfs.py` style:
```
python scripts/extract_corpus.py
python scripts/extract_corpus.py --chunk-size 500 --chunk-overlap 75
python scripts/extract_corpus.py --limit 5
python scripts/extract_corpus.py --doc-name 3M_2018_10K --overwrite
python scripts/extract_corpus.py --table-backend pymupdf|pdfplumber
python scripts/extract_corpus.py --table-format markdown|html|sentences
python scripts/extract_corpus.py --no-table-images
python scripts/extract_corpus.py --workers 4
```
See "Configuration reference" at the top of this document for the full parameter list.
This stage is CPU-bound (unlike the I/O-bound download step), so use
`ProcessPoolExecutor` rather than `ThreadPoolExecutor` for `--workers`.

## Edge cases

- **Missing PDFs (73 failed downloads)**: glob `data/raw_pdfs/*.pdf` that actually exist
  rather than iterating the catalog directly; catalog entries with no file logged as
  `status="source_missing"` in the report.
- **Corrupt/non-PDF files** (e.g. a saved WAF-block HTML page): wrap `fitz.open()` and
  per-page parsing in try/except, log `status="failed"` with the error, continue the
  batch — same isolation pattern as `download_pdf()`.
- **Scanned-page guard**: sample a few pages per doc; if extractable character count is
  ~0, flag `status="likely_scanned_or_image"` and skip (OCR is out of scope).
- **Huge documents** (up to 37MB/160+ pages): stream page-by-page, flush chunks
  incrementally to the per-doc JSONL rather than holding the whole doc in memory;
  record per-doc timing in the report.
- **Font-kerning artifacts** (observed glitches like "Shee t", "Flow s" from CID-font
  spacing in these filings): light regex cleanup pass, documented as a known limitation
  rather than perfectly reconstructed.
- **Orphaned raw files not in the catalog** (e.g. an `_annualreport.pdf` alongside a
  `_10K.pdf` for the same company): log a warning for manual review; extraction keys
  strictly off catalog `doc_name` for the graded corpus.

## Verification

1. `pip install -r requirements.txt` after adding the new deps.
2. `python scripts/extract_corpus.py --doc-name 3M_2018_10K --overwrite` — run on the
   one sampled/understood document first.
3. Manually inspect `data/processed/chunks/3M_2018_10K.jsonl`: confirm Item boundaries
   are respected, confirm at least one `chunk_type="table"` chunk for the Balance
   Sheet/Cash Flow Statement reads as a coherent sentence set, confirm `page_start`
   values align with the page numbers cited in `financebench_open_source.jsonl`
   evidence for that doc.
4. Inspect `data/processed/glossary/3M_2018_10K.glossary.jsonl` for at least a few
   correctly captured term/definition pairs (e.g. PP&E-style parenthetical
   abbreviations).
5. Run `python scripts/extract_corpus.py --limit 10` across a small multi-company
   sample to confirm the pipeline tolerates variety (10-K vs 10-Q vs 8-K vs earnings
   release layouts) before running the full 287-doc batch.
6. Run the full batch (`python scripts/extract_corpus.py`) and check
   `extraction_report.csv` for failure/skip counts consistent with the known 73 missing
   PDFs.
