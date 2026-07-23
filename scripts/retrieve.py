"""
Step 4: Query the vector index.

Usage:
    python scripts/retrieve.py "What was 3M's total assets in 2018?" --doc-name 3M_2018_10K --backend chroma --top-k 5
    python scripts/retrieve.py "What does PP&E mean?" --backend faiss
"""
from __future__ import annotations

import argparse
from pathlib import Path

from retrieval import embedding
from retrieval.chunk_store import ChunkStore
from retrieval.config import RetrievalConfig
from retrieval.glossary_lookup import GlossaryLookup
from retrieval.pipeline import RetrievalPipeline
from retrieval.store_chroma import ChromaVectorStore
from retrieval.store_faiss import FaissVectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_DIR = REPO_ROOT / "data" / "processed" / "chunks"
DEFAULT_GLOSSARY_DIR = REPO_ROOT / "data" / "processed" / "glossary"
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "processed" / "index"


def truncate(text: str, n: int = 200) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the retrieval pipeline.")
    parser.add_argument("query", help="The question to search for")
    parser.add_argument("--doc-name", default=None, help="Restrict search to this document")
    parser.add_argument("--backend", choices=["chroma", "faiss"], default="chroma")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--embedding-model", default=embedding.DEFAULT_MODEL)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--glossary-dir", type=Path, default=DEFAULT_GLOSSARY_DIR)
    args = parser.parse_args()

    config = RetrievalConfig(
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        neighbor_window=args.neighbor_window,
    )

    if args.backend == "chroma":
        vector_store = ChromaVectorStore(args.index_dir / "chroma")
    else:
        vector_store = FaissVectorStore(args.index_dir / "faiss")

    chunk_store = ChunkStore(args.chunks_dir)
    glossary_lookup = GlossaryLookup(args.glossary_dir)
    pipeline = RetrievalPipeline(vector_store, chunk_store, glossary_lookup, config)

    result = pipeline.search(args.query, top_k=args.top_k, doc_name=args.doc_name)

    print(f"Query: {result.query}\n")
    if not result.hits:
        print("No hits.")
        return

    for i, hit in enumerate(result.hits, 1):
        chunk = hit.chunk
        print(f"[{i}] score={hit.score:.4f}  {chunk['doc_name']}  page={chunk['page_start']}  section={chunk.get('section')}")
        print(f"    {truncate(chunk['text'])}")
        for n in hit.neighbors:
            print(f"    neighbor ({n['chunk_id']}): {truncate(n['text'], 120)}")
        for g in hit.glossary_matches:
            print(f"    glossary: '{g['term']}' = {g['definition']}")
        if hit.table_image:
            print(f"    table image: {hit.table_image}")
        print()


if __name__ == "__main__":
    main()
