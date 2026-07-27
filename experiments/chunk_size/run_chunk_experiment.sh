#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
OUT_BASE="$SCRIPT_DIR"

SIZES=(450 350 250)
NO_BERTSCORE=""

# Allow skipping bertscore
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-bertscore)   NO_BERTSCORE="--no-bertscore"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  Chunk Size Experiment"
echo "  Sizes : ${SIZES[*]}"
echo "============================================================"
echo ""

FAILED_RUNS=()

for s in "${SIZES[@]}"; do
  OUT_DIR="$OUT_BASE/size_$s"
  ANSWERS_FILE="$OUT_DIR/answers.jsonl"
  REPORT_FILE="$OUT_DIR/eval_report.json"
  LOG_FILE="$OUT_DIR/run.log"

  # Temporary directory for chunking and indexing
  TMP_DATA_DIR="$OUT_DIR/data"

  echo "────────────────────────────────────────────────────────────"
  echo "  Chunk Size : $s"
  echo "  Output     : $OUT_DIR"
  echo "────────────────────────────────────────────────────────────"

  mkdir -p "$OUT_DIR"
  
  # ── Step 1: Extract Corpus & Build Index ──────────────────────
  echo "[1/3] Preparing index for chunk size $s..."
  if ! python "$SCRIPTS_DIR/extract_corpus.py" --chunk-size "$s" --no-table-images --output-dir "$TMP_DATA_DIR" > /dev/null 2>&1; then
    echo "[1/3] ❌ extract_corpus.py failed for size=$s" | tee -a "$LOG_FILE"
    FAILED_RUNS+=("size=$s (extraction)")
    continue
  fi

  if ! python "$SCRIPTS_DIR/build_index.py" --backend faiss --chunks-dir "$TMP_DATA_DIR/chunks" --index-dir "$TMP_DATA_DIR/index" --overwrite > /dev/null 2>&1; then
    echo "[1/3] ❌ build_index.py failed for size=$s" | tee -a "$LOG_FILE"
    FAILED_RUNS+=("size=$s (indexing)")
    continue
  fi
  echo "[1/3] ✅ Index prepared"

  # ── Step 2: Generate answers ──────────────────────────────────
  echo "[2/3] Generating answers..."
  if python "$SCRIPTS_DIR/generate_answers.py" \
      --backend          faiss \
      --chunks-dir       "$TMP_DATA_DIR/chunks" \
      --index-dir        "$TMP_DATA_DIR/index" \
      --output           "$ANSWERS_FILE" \
      --limit            20 \
      2>&1 | tee "$LOG_FILE"; then
    echo "[2/3] ✅ Done — $ANSWERS_FILE"
  else
    echo "[2/3] ❌ generate_answers.py failed for size=$s" | tee -a "$LOG_FILE"
    FAILED_RUNS+=("size=$s (generation)")
    echo ""
    continue
  fi

  # ── Step 3: Evaluate ──────────────────────────────────────────
  echo "[3/3] Evaluating..."
  if python "$SCRIPTS_DIR/evaluate.py" \
      --answers "$ANSWERS_FILE" \
      --output  "$REPORT_FILE"  \
      $NO_BERTSCORE             \
      2>&1 | tee -a "$LOG_FILE"; then
    echo "[3/3] ✅ Done — $REPORT_FILE"
  else
    echo "[3/3] ❌ evaluate.py failed for size=$s" | tee -a "$LOG_FILE"
    FAILED_RUNS+=("size=$s (evaluation)")
  fi

  # Clean up temporary data to leave only the 3 output files
  rm -rf "$TMP_DATA_DIR"
  echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  All runs complete."
echo ""
echo "  Results:"
for s in "${SIZES[@]}"; do
  REPORT="$OUT_BASE/size_$s/eval_report.json"
  if [[ -f "$REPORT" ]]; then
    python3 - "size=$s" "$REPORT" <<'PYEOF'
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
    echo "  size=$s  ❌ no report found"
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
