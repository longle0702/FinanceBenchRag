"""
Step 6: Evaluation pipeline for the Answer Generation stage.

Loads the answers.jsonl file produced by generate_answers.py and scores each
generated answer against the ground truth using:
  - ROUGE-1 / ROUGE-2 / ROUGE-L   (lexical overlap)
  - BERTScore F1                   (semantic similarity)
  - Exact-match (EM)               (strict)
  - Token-level F1                 (SQuAD-style)
  - Context Precision              (fraction of retrieved passages whose text
                                    overlaps with the ground-truth evidence)

Results are printed as a summary table and saved to results/eval_report.json.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --answers results/answers.jsonl --output results/eval_report.json
    python scripts/evaluate.py --no-bertscore   # skip BERTScore (slow on CPU)
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

REPO_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_ANSWERS = REPO_ROOT / "results" / "answers.jsonl"
DEFAULT_OUTPUT  = REPO_ROOT / "results" / "eval_report.json"


# ══════════════════════════════════════════════════════════
#  Lexical helpers
# ══════════════════════════════════════════════════════════

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(_normalise(pred) == _normalise(gold))


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 (SQuAD-style)."""
    pred_tokens = _normalise(pred).split()
    gold_tokens = _normalise(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def context_precision(passages: list[dict], evidence: str) -> float:
    """
    Fraction of retrieved passages that have at least one token overlap with
    the ground-truth evidence text.
    """
    if not passages or not evidence:
        return 0.0
    ev_tokens = set(_normalise(evidence).split())
    hits = 0
    for p in passages:
        p_tokens = set(_normalise(p.get("text", "")).split())
        if ev_tokens & p_tokens:
            hits += 1
    return hits / len(passages)


def rouge_scores(pred: str, gold: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(gold, pred)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ══════════════════════════════════════════════════════════
#  BERTScore (optional — slow on CPU)
# ══════════════════════════════════════════════════════════

def compute_bertscore(preds: list[str], refs: list[str]) -> list[float]:
    """Return per-sample BERTScore F1. Requires bert-score package."""
    from bert_score import score as bs_score
    _, _, F1 = bs_score(preds, refs, lang="en", rescale_with_baseline=True, verbose=False)
    return F1.tolist()


# ══════════════════════════════════════════════════════════
#  Main evaluation loop
# ══════════════════════════════════════════════════════════

def load_answers(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate(records: list[dict], use_bertscore: bool = True) -> dict:
    per_sample: list[dict] = []

    preds = []
    refs  = []

    for r in records:
        pred  = r.get("generated_answer", "")
        gold  = r.get("ground_truth", "")
        evid  = r.get("ground_truth_evidence", "")
        psgss = r.get("retrieved_passages", [])

        em  = exact_match(pred, gold)
        tf1 = token_f1(pred, gold)
        cp  = context_precision(psgss, evid)
        rs  = rouge_scores(pred, gold)

        per_sample.append({
            "financebench_id": r.get("financebench_id", ""),
            "company":         r.get("company", ""),
            "doc_name":        r.get("doc_name", ""),
            "question":        r.get("question", "")[:120],
            "exact_match":     em,
            "token_f1":        round(tf1, 4),
            "context_precision": round(cp, 4),
            **rs,
            "retrieval_time_s":       r.get("retrieval_time_s", 0),
            "generation_time_s":      r.get("generation_time_s", 0),
            "total_inference_time_s": round(
                r.get("retrieval_time_s", 0) + r.get("generation_time_s", 0), 3
            ),
        })

        preds.append(pred)
        refs.append(gold)

    # ── optional BERTScore ────────────────────────────────────────────────
    if use_bertscore and preds:
        print("[Eval] Computing BERTScore (this can take a minute on CPU)...")
        try:
            bert_f1s = compute_bertscore(preds, refs)
            for i, sample in enumerate(per_sample):
                sample["bertscore_f1"] = round(bert_f1s[i], 4)
        except Exception as e:
            print(f"[Eval] BERTScore failed: {e}. Skipping.")
            use_bertscore = False

    # ── aggregate metrics ─────────────────────────────────────────────────
    def mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    aggregate = {
        "n_samples":           len(per_sample),
        "exact_match":         mean([s["exact_match"]         for s in per_sample]),
        "token_f1":            mean([s["token_f1"]            for s in per_sample]),
        "context_precision":   mean([s["context_precision"]   for s in per_sample]),
        "rouge1":              mean([s["rouge1"]              for s in per_sample]),
        "rouge2":              mean([s["rouge2"]              for s in per_sample]),
        "rougeL":              mean([s["rougeL"]              for s in per_sample]),
        "avg_retrieval_time_s":       mean([s["retrieval_time_s"]       for s in per_sample]),
        "avg_generation_time_s":      mean([s["generation_time_s"]      for s in per_sample]),
        "avg_total_inference_time_s": mean([s["total_inference_time_s"] for s in per_sample]),
    }
    if use_bertscore:
        aggregate["bertscore_f1"] = mean([s.get("bertscore_f1", 0) for s in per_sample])

    return {"aggregate": aggregate, "per_sample": per_sample}


def print_report(report: dict) -> None:
    agg = report["aggregate"]
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    metrics = [
        ("Exact Match",            "exact_match"),
        ("Token F1",               "token_f1"),
        ("Context Precision",      "context_precision"),
        ("ROUGE-1",                "rouge1"),
        ("ROUGE-2",                "rouge2"),
        ("ROUGE-L",                "rougeL"),
        ("BERTScore F1",           "bertscore_f1"),
        ("Avg Retrieval (s)",      "avg_retrieval_time_s"),
        ("Avg Generation (s)",     "avg_generation_time_s"),
        ("Avg Total Inference (s)","avg_total_inference_time_s"),
    ]
    for label, key in metrics:
        if key in agg:
            print(f"  {label:<24}: {agg[key]:.4f}")
    print(f"  {'N samples':<24}: {agg['n_samples']}")
    print("=" * 60)

    # Show worst / best few by token_f1
    samples = sorted(report["per_sample"], key=lambda x: x["token_f1"])
    if samples:
        print("\n  LOWEST TOKEN-F1 EXAMPLES:")
        for s in samples[:3]:
            print(f"    [{s['financebench_id']}] tf1={s['token_f1']:.3f}  {s['question'][:80]}")
        print("\n  HIGHEST TOKEN-F1 EXAMPLES:")
        for s in samples[-3:]:
            print(f"    [{s['financebench_id']}] tf1={s['token_f1']:.3f}  {s['question'][:80]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers against FinanceBench ground truth."
    )
    parser.add_argument("--answers",      type=Path, default=DEFAULT_ANSWERS,
                        help="Path to the answers.jsonl file")
    parser.add_argument("--output",       type=Path, default=DEFAULT_OUTPUT,
                        help="Path to save the eval report JSON")
    parser.add_argument("--no-bertscore", action="store_true",
                        help="Skip BERTScore computation (much faster on CPU)")
    args = parser.parse_args()

    if not args.answers.exists():
        raise SystemExit(
            f"Answers file not found: {args.answers}\n"
            "Run generate_answers.py first."
        )

    records = load_answers(args.answers)
    if not records:
        raise SystemExit("No records found in answers file.")
    print(f"[Eval] Loaded {len(records)} answer records from {args.answers}")

    report = evaluate(records, use_bertscore=not args.no_bertscore)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Eval] Full report saved to {args.output}")

    print_report(report)


if __name__ == "__main__":
    main()
