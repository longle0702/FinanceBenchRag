"""
Step 3: Build the vector index.

Embeds every chunk under data/processed/chunks/*.jsonl and stores the vectors in
one or both vector-store backends (Chroma and/or FAISS). Vector stores are kept
"dumb" (id + vector + minimal filter metadata) — full chunk content is always
resolved through ChunkStore at query time (see RETRIEVAL_PLAN.md).

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --doc-name 3M_2018_10K --backend chroma --overwrite
    python scripts/build_index.py --limit 10 --backend both
    python scripts/build_index.py --embedding-model sentence-transformers/all-mpnet-base-v2
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from retrieval import embedding
from retrieval.store_chroma import ChromaVectorStore
from retrieval.store_faiss import FaissVectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_DIR = REPO_ROOT / "data" / "processed" / "chunks"
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "processed" / "index"


def load_all_chunks(chunks_dir: Path, doc_name: str | None, limit: int | None) -> list[dict]:
    files = sorted(chunks_dir.glob("*.jsonl"))
    if doc_name:
        files = [f for f in files if f.stem == doc_name]
    if limit:
        files = files[:limit]
    chunks = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and build the vector index.")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--backend", choices=["chroma", "faiss", "both"], default="both")
    parser.add_argument("--embedding-model", default=embedding.DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--doc-name", default=None, help="Only index this one document (testing)")
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N document files (testing)")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild the index from scratch instead of appending")
    args = parser.parse_args()

    chunks = load_all_chunks(args.chunks_dir, args.doc_name, args.limit)
    if not chunks:
        raise SystemExit("No chunks found to index.")
    print(f"Loaded {len(chunks)} chunks to embed.")

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "doc_name": c["doc_name"],
            "doc_type": c.get("doc_type", ""),
            "company": c.get("company", ""),
            "doc_period": str(c.get("doc_period", "")),
            "chunk_type": c.get("chunk_type", ""),
        }
        for c in chunks
    ]

    start = time.monotonic()
    embeddings = embedding.encode(texts, model_name=args.embedding_model, batch_size=args.batch_size)
    print(f"Embedded {len(texts)} chunks in {time.monotonic() - start:.1f}s")

    args.index_dir.mkdir(parents=True, exist_ok=True)

    if args.backend in ("chroma", "both"):
        chroma_dir = args.index_dir / "chroma"
        if args.overwrite and chroma_dir.exists():
            shutil.rmtree(chroma_dir)
        store = ChromaVectorStore(chroma_dir)
        store.add(ids, embeddings, metadatas)
        print(f"Chroma index now has {store.count()} vectors at {chroma_dir}")

    if args.backend in ("faiss", "both"):
        faiss_dir = args.index_dir / "faiss"
        if args.overwrite and faiss_dir.exists():
            shutil.rmtree(faiss_dir)
        store = FaissVectorStore(faiss_dir, dim=embeddings.shape[1])
        store.add(ids, embeddings, metadatas)
        store.save()
        print(f"FAISS index now has {store.count()} vectors at {faiss_dir}")

    manifest = {
        "embedding_model": args.embedding_model,
        "backend": args.backend,
        "n_chunks": len(chunks),
        "batch_size": args.batch_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path = args.index_dir / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
