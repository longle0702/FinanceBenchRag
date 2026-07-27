from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so "retrieval" and "generate_answers" are importable

from retrieval import embedding
from retrieval.pipeline import RetrievalPipeline
from generate_answers import (
    LocalLLM,
    build_prompt,
    build_retrieval_pipeline,
    SYSTEM_PROMPT,
    DEFAULT_CHUNKS_DIR,
    DEFAULT_GLOSSARY_DIR,
    DEFAULT_INDEX_DIR,
)


def answer_question(
    llm: LocalLLM,
    pipeline: RetrievalPipeline,
    question: str,
    top_k: int,
    doc_name: str | None,
    show_sources: bool = True,
) -> None:
    """Run retrieval and generation for a single question and display results."""
    t_ret = time.time()
    ret_result = pipeline.search(question, top_k=top_k, doc_name=doc_name)
    ret_time = time.time() - t_ret

    passages = [h.chunk for h in ret_result.hits]
    prompt = build_prompt(question, passages)

    t_gen = time.time()
    generated_answer = llm.generate(SYSTEM_PROMPT, prompt)
    gen_time = time.time() - t_gen

    print("\n============================================================")
    print(f"Question : {question}")
    if doc_name:
        print(f"Doc Filter: {doc_name}")
    print(f"Timing   : Retrieval {ret_time:.2f}s | Generation {gen_time:.2f}s | Total {ret_time + gen_time:.2f}s")
    print("============================================================")
    print(f"\nAnswer:\n{generated_answer}\n")

    if show_sources and ret_result.hits:
        print("------------------------------------------------------------")
        print("Top Retrieved Passages:")
        for i, hit in enumerate(ret_result.hits, 1):
            chunk = hit.chunk
            doc = chunk.get("doc_name", "unknown")
            page = chunk.get("page_start", "?")
            score = hit.score
            print(f"  [Passage {i}] {doc} (page {page}) | score: {score:.4f}")
            snippet = chunk.get("text", "").replace("\n", " ").strip()
            if len(snippet) > 150:
                snippet = snippet[:150] + "..."
            print(f"      \"{snippet}\"")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive inference CLI: ask any question and get an LLM-generated answer."
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="The question to answer (if omitted, enters an interactive loop)"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument("--backend", choices=["chroma", "faiss"], default="faiss")
    parser.add_argument("--embedding-model", default=embedding.DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--doc-name", default=None,
        help="Only search within this document"
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--no-sources", action="store_true", help="Do not print retrieved source passages")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--glossary-dir", type=Path, default=DEFAULT_GLOSSARY_DIR)
    args = parser.parse_args()

    print("[Setup] Loading retrieval pipeline ...")
    pipeline = build_retrieval_pipeline(
        args.backend,
        args.embedding_model,
        args.top_k,
        args.neighbor_window,
        index_dir=args.index_dir,
        chunks_dir=args.chunks_dir,
        glossary_dir=args.glossary_dir,
    )

    print(f"[Setup] Loading LLM: {args.model} ...")
    llm = LocalLLM(args.model, max_new_tokens=args.max_new_tokens, device=args.device)

    if args.query:
        # Single query mode
        answer_question(llm, pipeline, args.query, args.top_k, args.doc_name, show_sources=not args.no_sources)
    else:
        # Interactive loop mode
        print("\n[Interactive Mode] Type your question and press Enter. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                query = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Exiting.")
                break
            answer_question(llm, pipeline, query, args.top_k, args.doc_name, show_sources=not args.no_sources)


if __name__ == "__main__":
    main()
