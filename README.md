# FinanceBenchRag

RAG system for answering questions over company financial filings (10-K, 10-Q, 8-K)
using the [FinanceBench](https://github.com/patronus-ai/financebench) dataset.
EPITA NLP graded project A — full brief: `Graded_project_instructions_A_RAG.pdf`.

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

### 1. Download PDFs

```bash
python scripts/download_pdfs.py               # all filings → data/raw_pdfs/
python scripts/download_pdfs.py --limit 5     # quick test
python scripts/download_pdfs.py --retry-failed data/raw_pdfs/download_report.csv
```

73 of 360 filings fail permanently (dead IR links); the pipeline handles missing files gracefully.

### 2. Extract text, tables, and glossary

```bash
python scripts/extract_corpus.py                        # all docs
python scripts/extract_corpus.py --doc-name 3M_2018_10K # one doc
python scripts/extract_corpus.py --limit 10             # small batch
```

Key chunking defaults (all overridable via CLI flags):

| Setting | Flag | Default |
|---|---|---|
| Narrative chunk size | `--chunk-size` | 450 tokens |
| Chunk overlap | `--chunk-overlap` | 60 tokens |
| Min chunk size | `--min-chunk-tokens` | 40 tokens |
| Table rows per chunk | `--table-row-group-size` | 5 |
| Table format | `--table-format` | markdown |

Output lands in `data/processed/`: `chunks/`, `glossary/`, `table_images/`.

### 3. Build the vector index

```bash
python scripts/build_index.py                          # both backends, all chunks
python scripts/build_index.py --backend chroma --overwrite --doc-name 3M_2018_10K
python scripts/build_index.py --embedding-model sentence-transformers/all-mpnet-base-v2
```

Backends: **Chroma** (`data/processed/index/chroma`) and **FAISS** (`data/processed/index/faiss`).
Default embedding model: `sentence-transformers/all-MiniLM-L6-v2`.

> ⚠️ If you change the embedding model, rebuild the index with `--overwrite` and pass the same `--embedding-model` flag to `generate_answers.py`.

### 4. Generate answers + evaluate

**Quickstart** (runs both steps, prints a summary):

```bash
source .venv/bin/activate
bash scripts/run_task3.sh --limit 20
```

**Step-by-step:**

```bash
# Generate answers
python scripts/generate_answers.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --backend chroma --top-k 5 --limit 100 \
  --output results/answers_baseline.jsonl

# Evaluate
python scripts/evaluate.py \
  --answers results/answers_baseline.jsonl \
  --output  results/eval_baseline.json
```

#### `generate_answers.py` flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace generation model |
| `--backend` | `chroma` | `chroma` or `faiss` |
| `--embedding-model` | `all-MiniLM-L6-v2` | Must match the index |
| `--top-k` | `5` | Retrieved passages per question |
| `--neighbor-window` | `1` | Expand hits with ±N adjacent chunks |
| `--max-new-tokens` | `256` | Max tokens to generate |
| `--limit` | *(all)* | Cap at N questions |
| `--doc-name` | *(all)* | Restrict to one document |
| `--device` | `auto` | `cpu`, `cuda`, or `mps` |
| `--output` | `results/answers.jsonl` | Output path |

#### `evaluate.py` flags

| Flag | Default | Description |
|---|---|---|
| `--answers` | `results/answers.jsonl` | Output from `generate_answers.py` |
| `--output` | `results/eval_report.json` | JSON report path |
| `--no-bertscore` | *(off)* | Skip BERTScore (slow on CPU) |

#### Metrics

| Metric | What it measures |
|---|---|
| **Exact Match** | Strict string equality after normalisation |
| **Token F1** | SQuAD-style token overlap (main quality signal) |
| **Context Precision** | Fraction of retrieved passages overlapping ground-truth evidence (retrieval quality) |
| **ROUGE-1/2/L** | Lexical overlap at unigram / bigram / subsequence level |
| **BERTScore F1** | Semantic similarity via BERT embeddings |
| **Avg Retrieval (s)** | Mean vector-search time per question |
| **Avg Generation (s)** | Mean LLM generation time per question |
| **Avg Total Inference (s)** | Retrieval + generation combined |

> **Diagnostic tip:** high Context Precision + low Token F1 → generation problem. Low Context Precision + low Token F1 → retrieval problem.

#### `run_task3.sh` flags

| Flag | Default |
|---|---|
| `--limit N` | `20` |
| `--model MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `--backend chroma\|faiss` | `chroma` |
| `--top-k N` | `5` |
| `--max-new-tokens N` | `256` |
| `--no-bertscore` | *(off)* |

Results: `results/answers_<timestamp>.jsonl` + `results/eval_report_<timestamp>.json`.

### Experimental ablations

```bash
# Different generation model
bash scripts/run_task3.sh --model Qwen/Qwen2.5-7B-Instruct --limit 100

# Different k
bash scripts/run_task3.sh --top-k 10 --limit 100

# FAISS backend
bash scripts/run_task3.sh --backend faiss --limit 100

# Different embedding model (re-index first)
python scripts/build_index.py --embedding-model sentence-transformers/all-mpnet-base-v2 --overwrite
python scripts/generate_answers.py --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --limit 100 --output results/answers_mpnet.jsonl
python scripts/evaluate.py --answers results/answers_mpnet.jsonl
```

## Status

| Step | Status |
|---|---|
| 1. Download PDFs | ✅ Done |
| 2. Extract text / tables / glossary | ✅ Done |
| 3. Build vector index (Chroma + FAISS) | ✅ Done |
| 4. Retrieval pipeline | ✅ Done |
| 5. Answer generation | ✅ Done |
| 6. Evaluation | ✅ Done |

Corpus stats (279 docs processed, `table_row_group_size=5`):

| | Count |
|---|---|
| Narrative chunks | 84,382 |
| Table chunks | 32,356 |
| Glossary entries | 9,831 |
| **Total vectors indexed** | **116,738** |

## Known issues

- **73 filings permanently missing** — dead links on SEC EDGAR / company IR sites; the pipeline skips them gracefully.
- **8 KraftHeinz filings are scanned images** — no text layer, OCR not implemented, logged as `likely_scanned_or_image`.
- **Semantic search can miss specific numeric facts** in large table chunks — `table_row_group_size=5` (down from 20) substantially improves retrieval precision for line-item queries. Worth investigating as an ablation.
