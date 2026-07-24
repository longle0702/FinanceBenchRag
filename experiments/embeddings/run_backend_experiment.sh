#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
OUT_BASE="$SCRIPT_DIR"

BACKENDS=("chroma" "faiss")
NO_BERTSCORE=""

# Allow skipping bertscore
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-bertscore)   NO_BERTSCORE="--no-bertscore"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  Backend Experiment"
echo "  Backends : ${BACKENDS[*]}"
echo "============================================================"
echo ""

FAILED_RUNS=()

for backend in "${BACKENDS[@]}"; do
  OUT_DIR="$OUT_BASE/$backend"
  ANSWERS_FILE="$OUT_DIR/answers.jsonl"
  REPORT_FILE="$OUT_DIR/eval_report.json"
  LOG_FILE="$OUT_DIR/run.log"

  echo "────────────────────────────────────────────────────────────"
  echo "  Backend  : $backend"
  echo "  Output   : $OUT_DIR"
  echo "────────────────────────────────────────────────────────────"

  mkdir -p "$OUT_DIR"

  # ── Step 1: Generate answers ───────────────────────────────────
  echo "[1/2] Generating answers..."
  if python "$SCRIPTS_DIR/generate_answers.py" \
      --backend          "$backend"         \
      --output           "$ANSWERS_FILE"    \
      2>&1 | tee "$LOG_FILE"; then
    echo "[1/2] ✅ Done — $ANSWERS_FILE"
  else
    echo "[1/2] ❌ generate_answers.py failed for backend=$backend" | tee -a "$LOG_FILE"
    FAILED_RUNS+=("backend=$backend (generation)")
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
    echo "[2/2] ❌ evaluate.py failed for backend=$backend" | tee -a "$LOG_FILE"
    FAILED_RUNS+=("backend=$backend (evaluation)")
  fi

  echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  All runs complete."
echo ""
echo "  Results:"
for backend in "${BACKENDS[@]}"; do
  REPORT="$OUT_BASE/$backend/eval_report.json"
  if [[ -f "$REPORT" ]]; then
    python3 - "backend=$backend" "$REPORT" <<'PYEOF'
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
    echo "  backend=$backend  ❌ no report found"
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
