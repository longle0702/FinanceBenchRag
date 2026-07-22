# FinanceBenchRag

Retrieval-Augmented Generation (RAG) system for answering questions over financial
10-K/10-Q filings, built on the [FinanceBench](https://github.com/patronus-ai/financebench)
dataset. This is the EPITA NLP graded project A: *Building and Evaluating a
Retrieval-Augmented Generation (RAG) System*.

## Goal

Design, implement, and evaluate an end-to-end RAG pipeline that answers financial
questions using only information retrieved from a closed corpus of SEC filings, using
free, locally executable models for both embedding and generation.

## Dataset

Two JSONL files from FinanceBench live in [data/](data/), keyed together by `doc_name`:

| File | Purpose | Key fields |
|---|---|---|
| [financebench_document_information.jsonl](data/financebench_document_information.jsonl) | Catalog of source documents | `doc_name`, `doc_link` (URL to the filing PDF) |
| [financebench_open_source.jsonl](data/financebench_open_source.jsonl) | Ground-truth evaluation set | `doc_name`, `question`, `answer`, `evidence_text` |

Raw PDFs are downloaded on demand (see below) and are not checked into the repo.

## Planned pipeline

1. **PDF retrieval** — download each filing referenced in
   `financebench_document_information.jsonl` into `data/raw_pdfs/`.
2. **Corpus processing** — extract text/tables, clean, and chunk documents
   (chunk size/overlap configurable), preserving document structure and metadata.
3. **Indexing** — embed chunks and store them in a vector database (e.g. FAISS/Chroma).
4. **Retrieval** — similarity search over the vector index for a given query.
5. **Generation** — answer synthesis from retrieved passages with a local model
   (e.g. Llama, Mistral, Qwen, Phi), citing the passages used as evidence.
6. **Evaluation** — run all questions from `financebench_open_source.jsonl` through the
   pipeline and score answers against ground truth (e.g. RAGAS, LLM-as-a-judge).

## Status

Project scaffolding — pipeline code has not been implemented yet. This README will be
updated with installation and run instructions as the source code lands.

## Reference

Full assignment instructions: `Graded_project_instructions_A_RAG.pdf`.
