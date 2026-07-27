# FinanceBenchRag

RAG system for answering questions over company financial filings (10-K, 10-Q, 8-K)
using the [FinanceBench](https://github.com/patronus-ai/financebench) dataset.

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
python scripts/build_index.py                          
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
| `--backend` | `faiss` | `chroma` or `faiss` |
| `--embedding-model` | `all-MiniLM-L6-v2` | Must match the index |
| `--top-k` | `10` | Retrieved passages per question |
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
| `--no-bertscore` | `False` | Set this to skip BERTScore |

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


### Experimental ablations

We provide automated scripts to run ablation experiments across models, top-K values, and vector backends. These are located in the `experiments/` directory. Each script runs the generation and evaluation steps and outputs a summary of key metrics.

#### 1. Model Comparison
Compares generation models (e.g., `Qwen2.5-1.5B`, `SmolLM2-1.7B`, `Phi-1.5`).
```bash
bash experiments/models/run_model_comparison.sh
```
Results are saved in `experiments/models/<model_name>/`.

#### 2. Top-K Comparison
Compares retrieval configurations for `k=3`, `5`, `7`, and `10`.
```bash
bash experiments/num_k/run_k_experiment.sh
```
Results are saved in `experiments/num_k/k_<value>/`.

#### 3. Vector Backend Comparison
Compares `chroma` and `faiss` backends.
```bash
bash experiments/embeddings/run_backend_experiment.sh
```
Results are saved in `experiments/embeddings/<backend_name>/`.

### Utilities

**Interactive Question Answering (Inference)**  
Ask arbitrary questions and get LLM-generated answers based on the retrieved corpus. You can run a single query from the command line or enter an interactive chat loop:
```bash
# Single query mode
python scripts/inference.py "What was the revenue in 2022?" --top-k 5

# Interactive loop mode (prompts for questions continuously)
python scripts/inference.py --backend faiss --model Qwen/Qwen2.5-1.5B-Instruct
```

**Manual Retrieval Querying**  
Test the retrieval pipeline interactively for a specific query without LLM generation:
```bash
python scripts/retrieve.py "What was the revenue in 2022?" --top-k 5
```

**Benchmarking Embeddings**  
Estimate full-corpus embedding time by benchmarking on a single document:
```bash
python scripts/benchmark_embedding.py --doc-name NIKE_2022_10K --batch-size 64
```