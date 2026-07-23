"""
Benchmark: measure real embedding throughput on one document's chunks, and
extrapolate an estimate for the full corpus — before committing to embedding
all ~137K chunks and building the full vector-store pipeline around it.

Usage:
    python scripts/benchmark_embedding.py --doc-name NIKE_2022_10K
    python scripts/benchmark_embedding.py --doc-name CVSHEALTH_2018_10K --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from retrieval.embedding import DEFAULT_MODEL, encode, load_model

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_DIR = REPO_ROOT / "data" / "processed" / "chunks"


def count_corpus_chunks(chunks_dir: Path) -> int:
    total = 0
    for path in chunks_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            total += sum(1 for _ in f)
    return total


def load_chunk_texts(path: Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding throughput on one document's chunks.")
    parser.add_argument("--doc-name", default="NIKE_2022_10K", help="Document to benchmark against")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    chunk_file = args.chunks_dir / f"{args.doc_name}.jsonl"
    if not chunk_file.exists():
        raise SystemExit(f"No chunk file found at {chunk_file}")

    texts = load_chunk_texts(chunk_file)
    print(f"Loaded {len(texts)} chunks from {args.doc_name}")

    print(f"Loading model {args.embedding_model} ...")
    load_start = time.monotonic()
    load_model(args.embedding_model)
    load_time = time.monotonic() - load_start
    print(f"Model load time: {load_time:.2f}s (one-time cost, not per-chunk)")

    encode_start = time.monotonic()
    encode(texts, model_name=args.embedding_model, batch_size=args.batch_size, show_progress=True)
    encode_time = time.monotonic() - encode_start

    chunks_per_sec = len(texts) / encode_time if encode_time > 0 else float("inf")
    print(f"\nEncoded {len(texts)} chunks in {encode_time:.2f}s -> {chunks_per_sec:.1f} chunks/sec")

    total_chunks = count_corpus_chunks(args.chunks_dir)
    if total_chunks:
        estimate_s = total_chunks / chunks_per_sec
        print(f"Full corpus: {total_chunks} chunks across all processed docs")
        print(
            f"Estimated full-corpus embedding time: {estimate_s:.0f}s "
            f"(~{estimate_s / 60:.1f} min, ~{estimate_s / 3600:.2f} hr)"
        )


if __name__ == "__main__":
    main()
