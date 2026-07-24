set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
LIMIT=20
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
NO_BERTSCORE=""
BACKEND="chroma"
TOP_K=5
MAX_NEW_TOKENS=256

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --limit)         LIMIT="$2";         shift 2 ;;
    --model)         MODEL="$2";         shift 2 ;;
    --backend)       BACKEND="$2";       shift 2 ;;
    --top-k)         TOP_K="$2";         shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --no-bertscore)  NO_BERTSCORE="--no-bertscore"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo " Model   : $MODEL"
echo " Backend : $BACKEND"
echo " Top-K   : $TOP_K"
echo " Limit   : $LIMIT questions"
echo "============================================================"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$REPO_ROOT/results/answers_${TIMESTAMP}.jsonl"
EVAL_OUT="$REPO_ROOT/results/eval_report_${TIMESTAMP}.json"

mkdir -p "$REPO_ROOT/results"

echo ""
echo "[1/2] Generating answers..."
python "$SCRIPT_DIR/generate_answers.py" \
  --model          "$MODEL"          \
  --backend        "$BACKEND"        \
  --top-k          "$TOP_K"          \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit          "$LIMIT"          \
  --output         "$OUTPUT"

echo ""
echo "[2/2] Running evaluation..."
python "$SCRIPT_DIR/evaluate.py" \
  --answers "$OUTPUT"  \
  --output  "$EVAL_OUT" \
  $NO_BERTSCORE

echo "  Answers : $OUTPUT"
echo "  Report  : $EVAL_OUT"
