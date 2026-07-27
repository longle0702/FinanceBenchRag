from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so "retrieval" is importable

from retrieval import embedding
from retrieval.chunk_store import ChunkStore
from retrieval.config import RetrievalConfig
from retrieval.glossary_lookup import GlossaryLookup
from retrieval.pipeline import RetrievalPipeline
from retrieval.store_chroma import ChromaVectorStore
from retrieval.store_faiss import FaissVectorStore

DEFAULT_CHUNKS_DIR   = REPO_ROOT / "data" / "processed" / "chunks"
DEFAULT_GLOSSARY_DIR = REPO_ROOT / "data" / "processed" / "glossary"
DEFAULT_INDEX_DIR    = REPO_ROOT / "data" / "processed" / "index"
DEFAULT_QA_FILE      = REPO_ROOT / "data" / "financebench_open_source.jsonl"
DEFAULT_OUTPUT_DIR   = REPO_ROOT / "results"

# ── prompt helpers ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the question using ONLY the provided context passages. "
    "Be concise and precise. "
    "After your answer, list the passage numbers you relied on in square brackets, "
    "e.g. [Passage 1, Passage 3]."
)


def build_prompt(question: str, passages: list[dict]) -> str:
    """Build the full prompt fed to the LLM."""
    context_parts = []
    for i, p in enumerate(passages, 1):
        doc  = p.get("doc_name", "unknown")
        page = p.get("page_start", "?")
        text = p["text"].replace("\n", " ").strip()
        context_parts.append(f"[Passage {i}] (doc={doc}, page={page})\n{text}")
    context = "\n\n".join(context_parts)

    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (using only the context above):"
    )


# ── LLM wrapper ──────────────────────────────────────────────────────────────

class LocalLLM:
    """Thin wrapper around a HuggingFace causal LM."""

    def __init__(self, model_name: str, max_new_tokens: int = 256, device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"[LLM] Loading tokenizer & model: {model_name} ...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        # dtype selection: use float16 on CUDA, float32 on CPU
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "auto":
            actual_device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=actual_device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device         = actual_device
        self.max_new_tokens = max_new_tokens
        print(f"[LLM] Model loaded in {time.time() - t0:.1f}s on {actual_device}.")

    def generate(self, system: str, user: str) -> str:
        """Apply chat template (if available) then generate a response."""
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        # Use chat template if the tokenizer supports it
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = f"{system}\n\n{user}"
        else:
            text = f"{system}\n\n{user}"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,        # greedy -- deterministic & reproducible
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── retrieval helpers ────────────────────────────────────────────────────────

def build_retrieval_pipeline(
    backend: str,
    embedding_model: str,
    top_k: int,
    neighbor_window: int,
    index_dir: Path = DEFAULT_INDEX_DIR,
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    glossary_dir: Path = DEFAULT_GLOSSARY_DIR,
) -> RetrievalPipeline:
    config = RetrievalConfig(
        embedding_model=embedding_model,
        top_k=top_k,
        neighbor_window=neighbor_window,
    )
    if backend == "chroma":
        vector_store = ChromaVectorStore(index_dir / "chroma")
    else:
        vector_store = FaissVectorStore(index_dir / "faiss")

    chunk_store     = ChunkStore(chunks_dir)
    glossary_lookup = GlossaryLookup(glossary_dir)
    return RetrievalPipeline(vector_store, chunk_store, glossary_lookup, config)


# ── data helpers ─────────────────────────────────────────────────────────────

def load_qa_pairs(qa_file: Path, doc_name_filter: str | None, limit: int | None) -> list[dict]:
    pairs = []
    with qa_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if doc_name_filter and item.get("doc_name") != doc_name_filter:
                continue
            pairs.append(item)
    if limit:
        pairs = pairs[:limit]
    return pairs


def extract_evidence_text(item: dict) -> str:
    """Pull expected evidence from the FinanceBench schema."""
    evidences = item.get("evidence", [])
    if not evidences:
        return ""
    return " ".join(e.get("evidence_text", "") for e in evidences)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer generation using local LLM + retrieval pipeline."
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID (must be free/local-compatible)"
    )
    parser.add_argument("--backend",         choices=["chroma", "faiss"], default="chroma")
    parser.add_argument("--embedding-model", default=embedding.DEFAULT_MODEL)
    parser.add_argument("--top-k",           type=int, default=10)
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--max-new-tokens",  type=int, default=256)
    parser.add_argument("--qa-file",         type=Path, default=DEFAULT_QA_FILE)
    parser.add_argument("--index-dir",       type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--chunks-dir",      type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR / "answers.jsonl"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N questions (useful for testing)"
    )
    parser.add_argument(
        "--doc-name", default=None,
        help="Only process questions for this document"
    )
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--eval", action="store_true",
                        help="Automatically evaluate generated answers after completion")
    parser.add_argument("--eval-output", type=Path, default=None,
                        help="Path to save evaluation JSON report (when --eval is set)")
    parser.add_argument("--no-bertscore", action="store_true",
                        help="Skip BERTScore computation during evaluation")
    args = parser.parse_args()

    # ── setup ────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("[Setup] Loading retrieval pipeline ...")
    pipeline = build_retrieval_pipeline(
        args.backend, args.embedding_model, args.top_k, args.neighbor_window,
        index_dir=args.index_dir, chunks_dir=args.chunks_dir
    )

    print(f"[Setup] Loading LLM: {args.model} ...")
    llm = LocalLLM(args.model, max_new_tokens=args.max_new_tokens, device=args.device)

    qa_pairs = load_qa_pairs(args.qa_file, args.doc_name, args.limit)
    print(f"[Run] Processing {len(qa_pairs)} questions ...")

    results = []
    for idx, item in enumerate(qa_pairs, 1):
        question    = item["question"]
        doc_name    = item.get("doc_name")
        gt_answer   = item.get("answer", "")
        gt_evidence = extract_evidence_text(item)

        # Retrieve relevant passages (optionally restricted to the right document)
        t_ret = time.time()
        ret_result = pipeline.search(
            question,
            top_k=args.top_k,
            doc_name=doc_name,   # restrict to the relevant doc when possible
        )
        ret_time = time.time() - t_ret

        passages = [h.chunk for h in ret_result.hits]

        # Build prompt and generate answer
        prompt = build_prompt(question, passages)
        t_gen  = time.time()
        generated_answer = llm.generate(SYSTEM_PROMPT, prompt)
        gen_time = time.time() - t_gen

        record = {
            "financebench_id":       item.get("financebench_id", f"idx_{idx}"),
            "company":               item.get("company", ""),
            "doc_name":              doc_name,
            "question":              question,
            "ground_truth":          gt_answer,
            "ground_truth_evidence": gt_evidence,
            "generated_answer":      generated_answer,
            "retrieved_passages": [
                {
                    "chunk_id":   h.chunk.get("chunk_id", ""),
                    "doc_name":   h.chunk.get("doc_name", ""),
                    "page_start": h.chunk.get("page_start"),
                    "score":      round(h.score, 4),
                    "text":       h.chunk["text"][:500],   
                }
                for h in ret_result.hits
            ],
            "retrieval_time_s":      round(ret_time, 3),
            "generation_time_s":     round(gen_time, 3),
            "total_inference_time_s": round(ret_time + gen_time, 3),
        }
        results.append(record)

        # Stream output so we can inspect progress early
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"[{idx}/{len(qa_pairs)}] {item.get('financebench_id', '')} "
            f"ret={ret_time:.1f}s gen={gen_time:.1f}s"
        )
        print(f"  Q: {question[:100]}")
        print(f"  A: {generated_answer[:150]}")
        print()

    print(f"\nDone. {len(results)} answers written to {args.output}")

    if args.eval and results:
        from evaluate import evaluate, print_report
        print("\n[Eval] Automatically running evaluation on generated answers...")
        report = evaluate(results, use_bertscore=not args.no_bertscore)
        if args.eval_output:
            args.eval_output.parent.mkdir(parents=True, exist_ok=True)
            args.eval_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[Eval] Full evaluation report saved to {args.eval_output}")
        print_report(report)


if __name__ == "__main__":
    main()
