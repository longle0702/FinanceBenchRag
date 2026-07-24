#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
OUT_BASE="$SCRIPT_DIR"

K_VALUES=(3 5 7 10)
NO_BERTSCORE=""

# Allow skipping bertscore
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-bertscore)   NO_BERTSCORE="--no-bertscore"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  Top-K Experiment"
echo "  K Values : ${K_VALUES[*]}"
echo "============================================================"
echo ""

FAILED_RUNS=()

for k in "${K_VALUES[@]}"; do
  OUT_DIR="$OUT_BASE/k_$k"
  ANSWERS_FILE="$OUT_DIR/answers.jsonl"
  REPORT_FILE="$OUT_DIR/eval_report.json"
  LOG_FILE="$OUT_DIR/run.log"

  echo "────────────────────────────────────────────────────────────"
  echo "  Top-K  : $k"
  echo "  Output : $OUT_DIR"
  echo "────────────────────────────────────────────────────────────"

  mkdir -p "$OUT_DIR"

  # ── Step 1: Generate answers ───────────────────────────────────
  echo "[1/2] Generating answers..."
  if python "$SCRIPTS_DIR/generate_answers.py" \
      --top-k            "$k"           \
      --output           "$ANSWERS_FILE"    \
      2>&1 | tee "$LOG_FILE"; then
    echo "[1/2] ✅ Done — $ANSWERS_FILE"
  else
    echo "[1/2] ❌ generate_answers.py failed for k=$k" | tee -a "$LOG_FILE"
    FAILED_RUNS+=("k=$k (generation)")
    echo ""
    continue
  fi

  # ── Step 2: Evaluate ──────────────────────────────────────────
  echo "[2/2] Evaluating..."
  if python "$SCRIPTS_DIR/evaluate.py" \
      --answers "$ANSWERS_FILE" \
      --output  "$REPORT_FILE"  \
      $NO_BERTSCORE             \
      2>&1 | tee -a "$LOG_FILE"; then
    echo "[2/2] ✅ Done — $REPORT_FILE"
  else
    echo "[2/2] ❌ evaluate.py failed for k=$k" | tee -a "$LOG_FILE"
    FAILED_RUNS+=("k=$k (evaluation)")
  fi

  echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  All runs complete."
echo ""
echo "  Results:"
for k in "${K_VALUES[@]}"; do
  REPORT="$OUT_BASE/k_$k/eval_report.json"
  if [[ -f "$REPORT" ]]; then
    python3 - "k=$k" "$REPORT" <<'PYEOF'
import sys, json
name, path = sys.argv[1], sys.argv[2]
r = json.load(open(path))["aggregate"]
print(f"  {name}")
print(f"    Token F1            : {r.get('token_f1', 'N/A')}")
print(f"    ROUGE-L             : {r.get('rougeL', 'N/A')}")
print(f"    Context Precision   : {r.get('context_precision', 'N/A')}")
if 'bertscore_f1' in r:
    print(f"    BERTScore F1        : {r['bertscore_f1']}")
print(f"    Avg Total Infer (s) : {r.get('avg_total_inference_time_s', 'N/A')}")
PYEOF
  else
    echo "  k=$k  ❌ no report found"
  fi
done

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
  echo ""
  echo "  ⚠️  Failed runs:"
  for m in "${FAILED_RUNS[@]}"; do
    echo "    - $m"
  done
fi
echo "============================================================"
