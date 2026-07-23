# FinanceBenchRag

A Retrieval-Augmented Generation (RAG) system that answers questions about company
financial filings (10-K, 10-Q, 8-K, earnings releases), using the
[FinanceBench](https://github.com/patronus-ai/financebench) dataset. This is the EPITA
NLP graded project A: *Building and Evaluating a Retrieval-Augmented Generation (RAG)
System*. Full assignment brief: `Graded_project_instructions_A_RAG.pdf`.

**Goal**: answer financial questions using only what's in the provided documents (no
outside knowledge), using free, locally-runnable models for embedding and generation.

This README walks through the project in the order you'd actually do it: setup, get the
data, process it, then (once built) index/retrieve/generate/evaluate.

## 1. Setup

Requires Python 3.12+.

```
python -m venv .venv          # or use a conda environment
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## 2. The data

Two small JSONL files ship in [data/](data/), linked by `doc_name`:

| File | What it is | Key fields |
|---|---|---|
| [financebench_document_information.jsonl](data/financebench_document_information.jsonl) | Catalog of filings | `doc_name`, `doc_link` (URL to the PDF), `company`, `doc_type`, `doc_period` |
| [financebench_open_source.jsonl](data/financebench_open_source.jsonl) | Test questions with known-correct answers | `doc_name`, `question`, `answer`, `evidence_text` |

The actual PDF filings are **not** checked into the repo — they're downloaded in the
next step.

## 3. Download the PDFs

```
python scripts/download_pdfs.py               # downloads everything into data/raw_pdfs/
python scripts/download_pdfs.py --limit 5      # just test on a few files first
python scripts/download_pdfs.py --retry-failed data/raw_pdfs/download_report.csv
```

What it does:
- Downloads every filing listed in the catalog, 4 at a time by default (polite towards
  SEC EDGAR's rate limit), with progress bars.
- Handles slow/hanging servers with timeouts, and unwraps a couple of known "viewer
  link" URL formats down to the real PDF.
- Writes `data/raw_pdfs/download_report.csv` (one row per file: downloaded / skipped /
  failed) so you can see what's missing and why.
- **Known failures you can't fix**: a handful of company investor-relations sites block
  non-browser downloads or have dead links (e.g. Johnson & Johnson's IR domain no longer
  resolves at all). Re-running `--retry-failed` won't help those; 73 of the 360 filings
  listed in the catalog end up permanently missing this way. The rest of the pipeline is
  built to tolerate that gracefully rather than fail on it.

## 4. Extract text, tables, and glossary terms from the PDFs

```
python scripts/extract_corpus.py                          # process every downloaded PDF
python scripts/extract_corpus.py --doc-name 3M_2018_10K    # just one file, for testing
python scripts/extract_corpus.py --limit 10                # a small batch, for testing
```

This is the step that turns raw PDFs into the pieces a retrieval system can search over.
Each filing gets read once, page by page, and split into three kinds of content:

- **Narrative text** (Business description, Risk Factors, MD&A, etc.) — cleaned up and
  cut into overlapping chunks of a few hundred words each, so nearby sentences don't get
  awkwardly split apart. Never splits across a section (e.g. "Item 7") boundary.
- **Financial tables** (Balance Sheet, Cash Flow Statement, Notes) — detected
  separately from narrative text, and written out as a proper Markdown (or HTML) table,
  since flattening a table into plain text scrambles which number belongs to which row
  and year. Each table is also **cropped out of the PDF page as a high-resolution PNG
  image**, so you can visually double-check a number if the text version looks off.
- **Glossary terms** (abbreviations like "PP&E", "GAAP", defined terms) — pulled out
  into a separate per-document lookup file, so an abbreviation found anywhere later can
  be expanded to its full meaning by simple lookup instead of another search.

What you get afterwards, all under `data/processed/`:

```
data/processed/
  chunks/<doc_name>.jsonl                 # narrative + table chunks, ready to embed
  table_images/<doc_name>/<table_id>.png  # cropped image of each table
  glossary/<doc_name>.glossary.jsonl      # this document's term -> definition lookup
  glossary_global.jsonl                   # same, merged across every document
  extraction_report.csv                   # one row per doc: succeeded / failed / skipped, and why
  extraction_config.json                  # exact settings used for this run (for reproducibility)
```

Every chunk also carries metadata — which document/page/section it came from, and
which chunk comes right before/after it — so a later retrieval step can pull in
neighboring chunks for extra context, not just the single best match.

A page that has no extractable text at all (i.e. it's a scanned image, not real text) is
detected and skipped rather than producing garbage — this turned out to apply to all 8
KraftHeinz filings in the catalog, which are scanned images with no text layer. OCR is
not implemented, so these are logged as `likely_scanned_or_image` and left out.

Useful options (chunk size, table format, image DPI, etc.) are all listed with defaults
in **[EXTRACTION_PLAN.md](EXTRACTION_PLAN.md)** under "Configuration reference" — that
file also explains the *why* behind each design choice (font-kerning quirks in these
PDFs, why tables need a different detection strategy than plain text, etc.).

## Data cleaning

Raw text pulled straight out of a PDF is messy — it comes out line-by-line, not as
proper paragraphs, and repeats things like page headers on every single page. Before
anything gets chunked or embedded, each document goes through these cleaning steps:

1. **Repeated header/footer removal** — any line (e.g. a "Table of Contents" running
   footer) that shows up on more than 60% of a document's pages gets detected once and
   stripped everywhere, so it never pollutes a chunk's text (`pdf_io.compute_boilerplate_lines`).
2. **De-hyphenation** — PDFs wrap words across lines with a hyphen (`"cash-"` /
   `"equivalents"`); these get rejoined into `"cash equivalents"` (`narrative.HYPHEN_BREAK_RE`).
3. **Whitespace normalization** — repeated spaces/tabs collapsed to one space.
4. **Paragraph reconstruction** — the PDF gives text one line at a time; lines are
   regrouped into real paragraphs based on the vertical spacing between them
   (`narrative.join_block_paragraphs`).
5. **Table cell cleaning** — currency/negative-sign symbols that land in their own grid
   cell (e.g. `"$"` next to `"2,853"`) are merged back together, and every value is
   parsed into a proper positive/negative number, filtering out fragments that aren't
   really values (`tables._merge_prefix_cells`, `tables.parse_numeric`).
6. **Corrupt-file isolation** — if one PDF is broken or unreadable, only that document
   fails (logged with its error); the rest of the batch keeps going.

(Scanned/image pages with no real text layer — see the KraftHeinz case above — are also
caught here and skipped rather than cleaned, since there's no text to clean.)

What's deliberately **not** done: no lowercasing, stemming, or stopword removal — those
would hurt embedding quality for semantic search, not help it. The font-kerning glitch
that occasionally splits a word in two (e.g. `"Cash and cash e quivalents"`) is left as
a documented known limitation rather than patched with a fragile heuristic — see
`EXTRACTION_PLAN.md`'s Edge Cases section for why.

## 5. What's next (not built yet)

5. **Indexing** — turn each chunk's text into a vector embedding, store in a vector
   database (e.g. FAISS or Chroma).
6. **Retrieval** — given a question, find the most similar chunk(s) by vector search,
   then expand with neighboring chunks for context.
7. **Generation** — pass the retrieved passages to a local, free LLM (e.g. Llama,
   Mistral, Qwen, Phi) to write an answer that cites its sources.
8. **Evaluation** — run every question in `financebench_open_source.jsonl` through the
   pipeline and score the answers against the known-correct ones (e.g. with RAGAS or an
   LLM-as-judge).

## Status

| Step | Status |
|---|---|
| 1. Download PDFs | Done |
| 2. Extract text/tables/glossary | Done |
| 3. Index into a vector store | Not started |
| 4. Retrieval | Not started |
| 5. Generation | Not started |
| 6. Evaluation | Not started |

Latest full-corpus extraction run, over all 360 documents in the catalog:

| | Count |
|---|---|
| Processed successfully | 279 |
| Missing (download failed — see step 3 above) | 73 |
| Skipped (scanned image, no text layer — all 8 KraftHeinz filings) | 8 |
| **Narrative chunks produced** | ~125,400 |
| **Table chunks produced** | ~12,000 |
| **Glossary entries extracted** | ~9,800 |

This README is updated as each step lands.
