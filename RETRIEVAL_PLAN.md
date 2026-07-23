# Retrieval System (Step 3-4: Indexing + Retrieval)

## Configuration reference

| Parameter | CLI flag | Default | Meaning |
|---|---|---|---|
| `embedding_model` | `--embedding-model` | `sentence-transformers/all-MiniLM-L6-v2` | Local, free embedding model (384-dim, small, CPU-friendly) |
| `backend` (build) | `--backend` | `both` | Which vector store(s) to build: `chroma`, `faiss`, or `both` |
| `backend` (query) | `--backend` | `chroma` | Which store to query against (must pick one) |
| `top_k` | `--top-k` | `5` | Number of chunks returned per query |
| `neighbor_window` | `--neighbor-window` | `1` | How many chunks before/after a hit to pull in for context |
| `batch_size` | `--batch-size` | `64` | Embedding batch size |

A snapshot of the exact settings used is written to `data/processed/index/index_manifest.json`
on every build, same reproducibility pattern as `extraction_config.json`.

## Context

Text extraction (see [EXTRACTION_PLAN.md](EXTRACTION_PLAN.md)) is complete: 279
documents processed into **137,375 chunks** (125,415 narrative + 11,960 table) under
`data/processed/chunks/*.jsonl`, plus 9,831 glossary entries and a 2,707-entry global
rollup. This is the assignment's "Retrieval System" task — index chunks as vector
representations, retrieve via similarity search — plus three extra capabilities
layered on top: neighbor-chunk context expansion, glossary-term expansion on
retrieved text, and table-image attachment for table-type hits.

Confirmed facts going into this design:
- No embedding/retrieval code existed before this — clean slate.
- The `ml` conda env has `torch` 2.12.0+cpu (**no CUDA/GPU on this machine**),
  `faiss-cpu` 1.14.3, `numpy`. `sentence-transformers` and `chromadb` are added here.
- `Chunk`/`TableData` (`scripts/extract/schema.py`) already carries everything this
  stage needs: `chunk_id`, `doc_name`, `chunk_type`, `text`, `prev_chunk_id`/
  `next_chunk_id` (neighbor expansion), `structured.table_image` (table-image
  attachment), `page_start`/`section`, etc. — nothing had to change upstream.
- No GPU means CPU-only embedding inference for 137,375 chunks — rather than guess
  at timing, the implementation sequence benchmarks real throughput on one document
  first (see "Scale/time expectations" below) before committing to the full run.

**Two design adjustments from the original ask**: build one `VectorStore` interface
with Chroma and FAISS as swappable backends rather than two fully independent
pipelines — FAISS has no native metadata storage while Chroma does, so unifying
behind one interface avoids duplicating the embedding/loading logic twice, and get
Chroma working end-to-end first, add FAISS behind the same interface second. And
treat embedding model choice as a first-class configurable, since the assignment
explicitly requires comparing embedding models later.

## Library choices

- **`sentence-transformers`**, default model `all-MiniLM-L6-v2` — free, local,
  small (~90MB / 22M params), fast enough for CPU-only inference. Configurable via
  `--embedding-model` so it's a real ablation axis later.
- **`chromadb`** — primary backend. Stores vectors *and* metadata/documents
  together, so it's the simpler one to stand up first.
- **`faiss-cpu`** (already installed) — secondary backend, `IndexFlatIP` over
  L2-normalized vectors (cosine similarity via inner product). No native metadata
  storage, so it only needs a `row_index -> chunk_id` sidecar — real content is
  resolved the same way as Chroma, through the shared `ChunkStore` (below).
- Both backends are built from the **same embedding pass** — encode once, add to
  whichever backend(s) were requested, rather than re-embedding per backend.

## Architecture

**Key simplification**: keep the vector stores "dumb" (id + vector only) and resolve
all real content (text, metadata, neighbors) through one shared `ChunkStore` that
lazy-loads a document's chunk file (`data/processed/chunks/<doc_name>.jsonl`) into a
`chunk_id -> Chunk` map, cached per doc. This sidesteps FAISS's lack of native
metadata storage entirely and keeps Chroma's stored metadata minimal (just enough
for `where` filtering). Neighbor expansion becomes a trivial walk of
`prev_chunk_id`/`next_chunk_id` through the same cached map — no new lookup
mechanism needed.

### File layout (mirrors `scripts/extract/` package conventions)

```
scripts/
  build_index.py                 # Step 3 CLI: embed all chunks, build vector store(s)
  retrieve.py                     # Step 4 CLI: query -> print retrieval result (manual test/demo)
  benchmark_embedding.py          # small CLI: measure real embedding throughput on one document
  retrieval/
    __init__.py
    config.py                     # RetrievalConfig: embedding_model, backend, top_k, neighbor_window, batch_size
    embedding.py                   # sentence-transformers load + batch encode (shared by build_index and retrieve)
    chunk_store.py                  # per-doc chunk JSONL loader/cache; get(chunk_id), get_with_neighbors(chunk_id, k)
    glossary_lookup.py                # per-doc glossary loader/cache; expand_terms(text, doc_name) -> matched entries
    store_base.py                      # VectorStore ABC: add(ids, embeddings, metadatas), query(embedding, top_k, where) -> [(chunk_id, score)]
    store_chroma.py                     # ChromaVectorStore (chromadb.PersistentClient, one collection)
    store_faiss.py                       # FaissVectorStore (IndexFlatIP over normalized vectors + ids.json sidecar)
    pipeline.py                           # RetrievalPipeline: query -> embed -> vector search -> resolve+expand+glossary+image -> RetrievalResult
```

## Embedding

- `embedding.py` wraps `SentenceTransformer(model_name)` + a batch-encode function
  (default batch size 64) with a `tqdm` progress bar, matching this repo's existing
  script conventions.
- The *same* function is used at both index-build time and query time, so both
  sides of the similarity search are always embedded by the same model.
- Chunk `text` is embedded as-is for both narrative and table chunks (table `text`
  already has company/period/title baked in from the extraction stage, e.g.
  `"Consolidated Balance Sheet (3M, 2018)\n\n| Line Item | 2018 | 2017 |..."`).
  **Known limitation, flagged rather than silently patched**: narrative chunk
  `text` does *not* currently get a prepended context string (company/section)
  before embedding, unlike table chunks — a candidate future ablation
  ("contextual" vs. plain embedding, in the spirit of Anthropic's contextual
  retrieval technique), not implemented in this pass.

## Vector stores

- `VectorStore` ABC (`store_base.py`): `add(ids: list[str], embeddings: np.ndarray, metadatas: list[dict])`, `query(embedding, top_k, where: dict | None) -> list[tuple[str, float]]`.
- **Chroma** (`store_chroma.py`): `chromadb.PersistentClient(path="data/processed/index/chroma")`, one collection `financebench_chunks`. Metadata kept minimal (scalars only, as Chroma requires): `doc_name`, `doc_type`, `company`, `doc_period`, `chunk_type`. Batched `add()` calls (Chroma has a max-batch-size limit, ~5000ish depending on version). Supports native `where={"doc_name": ...}` filtering.
- **FAISS** (`store_faiss.py`): `faiss.IndexFlatIP`, persisted to `data/processed/index/faiss/index.faiss` + `ids.json` (row position -> `chunk_id`). `where={"doc_name": ...}` implemented as pre-filtering (search only that doc's id subset, built at index time) rather than post-filtering, so `top_k` stays meaningful when a filter is active.
- `build_index.py --backend chroma|faiss|both` builds either or both from one embedding pass.

## Retrieval pipeline (`pipeline.py`)

`RetrievalPipeline.search(query: str, top_k: int = 5, doc_name: str | None = None) -> RetrievalResult`:
1. Embed the query text (same model as indexing).
2. `vector_store.query(embedding, top_k, where={"doc_name": doc_name} if doc_name else None)` → ranked `(chunk_id, score)`.
3. For each hit: resolve the full `Chunk` via `ChunkStore`; expand neighbors via `get_with_neighbors(chunk_id, config.neighbor_window)` (walks `prev_chunk_id`/`next_chunk_id`, stops cleanly at a doc boundary); scan the hit's + neighbors' `text` for terms from **that document's own glossary file only** (not the global rollup — avoids leaking a same-abbreviation-different-meaning definition from an unrelated filing) via `glossary_lookup.expand_terms`; if `chunk_type == "table"`, attach `structured.table_image`.
4. Return `RetrievalResult(query, hits: list[RetrievedHit])` — each `RetrievedHit` has `chunk`, `score`, `neighbors`, `glossary_matches`, `table_image`.

Global search (no `doc_name`) is the default, matching the assignment's "closed
document collection" framing; per-document scoping is available as an optional
filter since `financebench_open_source.jsonl` pairs every question with a known
`doc_name` — cheap to support both, so there's no need to commit to one.

## Output layout

```
data/processed/index/
  chroma/                       # chromadb.PersistentClient storage directory
  faiss/index.faiss             # FAISS flat index
  faiss/ids.json                # row position -> chunk_id sidecar
  index_manifest.json           # embedding model, backend(s), chunk count, timestamp
```

## CLI

```
python scripts/build_index.py                                   # embed + index every chunk, both backends
python scripts/build_index.py --doc-name 3M_2018_10K --backend chroma --overwrite   # test on one doc first
python scripts/build_index.py --limit 10 --backend both
python scripts/build_index.py --embedding-model sentence-transformers/all-mpnet-base-v2

python scripts/retrieve.py "What was 3M's total assets in 2018?" --doc-name 3M_2018_10K --backend chroma --top-k 5
python scripts/retrieve.py "What does PP&E mean?" --backend faiss   # global search, no doc filter
```
`retrieve.py` prints each hit: chunk text, score, section/page, neighbor previews, glossary matches, and the table image path if present.

## `requirements.txt` additions

```
sentence-transformers>=3.0
chromadb>=0.5
faiss-cpu>=1.8
torch>=2.0
```
(`faiss-cpu`/`torch` are already present in the `ml` env; declared here so a fresh
setup pulls them too.)

## Scale/time expectations — benchmark before the full build

137,375 chunks, CPU-only embedding (no GPU available). Rather than guess at timing,
the implementation sequence is:
1. Install `sentence-transformers`.
2. Build `embedding.py` (the one real, reusable module needed for this) and a small
   `scripts/benchmark_embedding.py` that embeds one real document's chunks and
   reports chunks/sec plus an extrapolated full-corpus estimate.
3. Only after that number is known, decide batch size / model / whether to build
   both backends over the full corpus in one go or stage it.

**Measured result** (`scripts/benchmark_embedding.py --doc-name NIKE_2022_10K`, 485
chunks, batch size 64): **47.8 chunks/sec** on CPU. Extrapolated to the full
137,375-chunk corpus: **~2,877s ≈ 48 minutes (~0.8 hr)** — a one-time cost, much
lighter than the ~4-hour PDF extraction batch. (Model load took 45s on this first
run, but that was mostly downloading the ~90MB model weights from Hugging Face —
a one-time cost per machine, not per run, since weights are cached locally afterward.)

## Edge cases

- Empty query result (e.g. a `doc_name` filter on a doc with zero chunks of the
  relevant type) returns an empty `RetrievalResult`, not an error.
- First run downloads the `sentence-transformers` model weights — no internet
  needed at query time afterward (cached locally).

## Verification

1. Add new deps to `requirements.txt`, install into the `ml` env.
2. Run `scripts/benchmark_embedding.py` on one document, report real chunks/sec and
   the extrapolated full-corpus time — decide next steps based on the real number.
3. `python scripts/build_index.py --doc-name 3M_2018_10K --backend chroma --overwrite` — build a small single-doc index.
4. `python scripts/retrieve.py "What was 3M's total assets in 2018?" --doc-name 3M_2018_10K --backend chroma --top-k 3` — confirm a relevant hit, correct neighbor chunks, a glossary match on a known term, and a `table_image` path that exists on disk for a table hit.
5. Repeat with `--backend faiss` on the same doc; confirm comparable top hit.
6. `python scripts/build_index.py --limit 10 --backend both` across a small multi-doc sample; confirm both backends build cleanly and `doc_name`-scoped queries correctly filter.
7. Once validated at small scale, run the full-corpus index build in the background (same pattern as the full extraction batch), then spot-check a handful of `financebench_open_source.jsonl` questions end-to-end.
