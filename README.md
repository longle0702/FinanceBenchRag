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

The most commonly-changed settings, and their defaults — all of these are plain CLI
flags, so it's safe to experiment without touching any code:

| Setting | Flag | Default |
|---|---|---|
| Narrative chunk size | `--chunk-size` | 450 tokens |
| Narrative chunk overlap | `--chunk-overlap` | 60 tokens |
| Minimum narrative chunk size | `--min-chunk-tokens` | 40 tokens |
| Table backend | `--table-backend` | pymupdf |
| Table text format | `--table-format` | markdown |
| Rows per table chunk | `--table-row-group-size` | 5 (was 20 — changed after testing showed smaller groups substantially improve retrieval precision for specific numeric facts; see "Problems" below) |
| Table images | `--no-table-images` (disables) | on, 300 DPI PNG |

The full parameter list (boilerplate detection, heading detection, table-region
thresholds, etc.) — plus the reasoning behind each default — is in
**[EXTRACTION_PLAN.md](EXTRACTION_PLAN.md)** under "Configuration reference".

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
7. **Tiny-fragment merging** — a section boundary, heading, or table can leave a
   near-empty leftover (e.g. a lone page number between two tables, or a couple of
   words after a heading) that would otherwise become its own near-content-free chunk.
   Any packed chunk under `--min-chunk-tokens` (default 40 tokens) gets merged into the
   previous narrative chunk, or carried forward across tables/headings until there's
   real content to attach to (`extract_corpus.flush_narrative`).

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

## Problems / known issues

**Table chunks are too coarse for precise numeric-fact retrieval.** Tested by
building a vector index (`scripts/build_index.py`, both Chroma and FAISS backends,
`sentence-transformers/all-MiniLM-L6-v2`) for `3M_2018_10K` and querying
`"What was 3M's total assets in 2018?"` (`scripts/retrieve.py`). The Balance Sheet
chunk containing the actual answer ("Total assets, 2018: $36,500") **is** in the
index, but ranked **#106 out of 396 chunks** (score 0.365) — nowhere near a
realistic top-k. Both backends agreed on the ranking, and the rest of the pipeline
(filtering, neighbor expansion, glossary lookup) worked correctly, so this isn't a
retrieval-pipeline bug — it's a chunking-granularity problem: `table_row_group_size`
(default 20) packs ~20 unrelated line items — cash, receivables, inventories, PP&E,
goodwill, total assets, etc. — into one chunk, and a sentence-embedding model pools
over the whole chunk, so the resulting vector represents "a generic mix of balance
sheet items" rather than "total assets" specifically. A natural-language query about
one line item can't cut through that dilution.

Candidate fixes (not yet decided between): shrink `table_row_group_size` so each
table chunk covers far fewer rows (currently being tested — see below); try a
larger/different embedding model; or add exact/keyword matching alongside semantic
search for numeric-fact queries. Worth keeping as a concrete case study for the
report's Error Analysis section either way.

**Update after testing `--table-row-group-size 3` on the same document**: confirmed
the fix. Re-extracted `3M_2018_10K` with `--table-row-group-size 3` (was 20),
rebuilt the index (471 chunks, up from 396), and re-ran the same query. The
"Total assets" chunk jumped from **rank #106/396 (score 0.365) to rank #5/471
(score 0.590)** — now inside a realistic top-5. One residual nuance: rank #1 for
this query is actually a *different* line item, "Total other current assets"
(page 80), which scores even higher (0.676) purely because "Total" + "assets"
overlap lexically/semantically with the query despite being the wrong fact — both
the right and wrong "total" show up together in the top-5, which a downstream
generation step would need to disambiguate using the full retrieved text, not just
the ranking.

**Default changed to `table_row_group_size = 5`** (was 20). Note this specific
value (5) was **not** independently re-tested — only `3` was empirically validated
above; `5` was chosen as a middle ground between that result and keeping table
chunks from becoming too numerous/granular, on the reasoning that fewer, slightly
larger table chunks reduce total chunk count (embedding/index cost) while still
being far more targeted than the original 20. Worth re-validating with the same
rank-check method once the full corpus is re-processed, and worth treating as a
genuine ablation value (3 vs 5 vs 8...) for the report rather than assuming 5 is
optimal. The full 279-doc corpus has **not** been re-extracted or re-indexed with
this new default yet — that's a separate, larger re-run still to be done.
