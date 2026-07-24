#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
OUT_BASE="$SCRIPT_DIR"

# ── defaults ──────────────────────────────────────────────────────────────────
BACKEND="faiss"
TOP_K=10
NEIGHBOR_WINDOW=1
MAX_NEW_TOKENS=256
NO_BERTSCORE=""

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --backend)        BACKEND="$2";        shift 2 ;;
    --top-k)          TOP_K="$2";          shift 2 ;;
    --neighbor-window) NEIGHBOR_WINDOW="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --no-bertscore)   NO_BERTSCORE="--no-bertscore"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── model registry ────────────────────────────────────────────────────────────
MODELS=(
  "Qwen2.5-1.5B|Qwen/Qwen2.5-1.5B-Instruct"
  "SmolLM2-1.7B|HuggingFaceTB/SmolLM2-1.7B-Instruct"
  "Phi-1.5|microsoft/phi-1_5"
)

echo "============================================================"
echo "  Model Comparison Experiment"
echo "  Backend   : $BACKEND"
echo "  Top-K     : $TOP_K"
echo "  BERTScore : $(if [[ -n "$NO_BERTSCORE" ]]; then echo "Disabled"; else echo "Enabled"; fi)"
echo "  Models    : ${#MODELS[@]}"
echo "  QA pairs  : 150 (full dataset)"
echo "============================================================"
echo ""

FAILED_MODELS=()

for entry in "${MODELS[@]}"; do
  FOLDER_NAME="${entry%%|*}"
  MODEL_ID="${entry##*|}"
  OUT_DIR="$OUT_BASE/$FOLDER_NAME"
  ANSWERS_FILE="$OUT_DIR/answers.jsonl"
  REPORT_FILE="$OUT_DIR/eval_report.json"
  LOG_FILE="$OUT_DIR/run.log"

  echo "────────────────────────────────────────────────────────────"
  echo "  Model  : $MODEL_ID"
  echo "  Output : $OUT_DIR"
  echo "────────────────────────────────────────────────────────────"

  mkdir -p "$OUT_DIR"

  # ── Step 1: Generate answers ───────────────────────────────────
  echo "[1/2] Generating answers..."
  if python "$SCRIPTS_DIR/generate_answers.py" \
      --model            "$MODEL_ID"        \
      --backend          "$BACKEND"         \
      --top-k            "$TOP_K"           \
      --neighbor-window  "$NEIGHBOR_WINDOW" \
      --max-new-tokens   "$MAX_NEW_TOKENS"  \
      --output           "$ANSWERS_FILE"    \
      2>&1 | tee "$LOG_FILE"; then
    echo "[1/2] ✅ Done — $ANSWERS_FILE"
  else
    echo "[1/2] ❌ generate_answers.py failed for $FOLDER_NAME" | tee -a "$LOG_FILE"
    FAILED_MODELS+=("$FOLDER_NAME (generation)")
    echo ""
    continue   # skip evaluate if generation failed
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
    echo "[2/2] ❌ evaluate.py failed for $FOLDER_NAME" | tee -a "$LOG_FILE"
    FAILED_MODELS+=("$FOLDER_NAME (evaluation)")
  fi

  echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  All runs complete."
echo ""
echo "  Results:"
for entry in "${MODELS[@]}"; do
  FOLDER_NAME="${entry%%|*}"
  REPORT="$OUT_BASE/$FOLDER_NAME/eval_report.json"
  if [[ -f "$REPORT" ]]; then
    # Pretty-print key aggregate metrics inline
    python3 - "$FOLDER_NAME" "$REPORT" <<'PYEOF'
import sys, json
name, path = sys.argv[1], sys.argv[2]
r = json.load(open(path))["aggregate"]
print(f"  {name}")
print(f"    Token F1            : {r.get('token_f1', 'N/A')}")
print(f"    ROUGE-L             : {r.get('rougeL', 'N/A')}")
print(f"    Context Precision   : {r.get('context_precision', 'N/A')}")
print(f"    BERTScore F1        : {r.get('bertscore_f1', 'N/A')}")
print(f"    Avg Total Infer (s) : {r.get('avg_total_inference_time_s', 'N/A')}")
PYEOF
  else
    echo "  $FOLDER_NAME  ❌ no report found"
  fi
done

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  echo ""
  echo "  ⚠️  Failed runs:"
  for m in "${FAILED_MODELS[@]}"; do
    echo "    - $m"
  done
fi
echo "============================================================"
