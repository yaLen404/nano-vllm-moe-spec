#!/usr/bin/env bash
# One-click ablation benchmark for nano-vllm.
#
#   bash run_bench.sh --dry-run                      # print the matrix (no GPU needed)
#   MODEL=/path/dense MOE_MODEL=/path/moe bash run_bench.sh
#
# Environment / options:
#   MODEL=path        dense model dir, used by E1/E2/regression (default ~/huggingface/Qwen3-0.6B)
#   MOE_MODEL=path    FULL MoE model dir, used by E3; E3 is skipped if unset
#   PY=path           python interpreter (auto: .venv/bin/python, then python3)
#   GPU_GIB=n         VRAM size for the KV preflight (auto via nvidia-smi)
#   --only LIST       subset of r1,e1,e2,e3,combo,e4
#   --runs N          repetitions per configuration (default 3)
#   --scale F         dataset size multiplier (default 1.0)
#   --skip-setup      reuse existing datasets
#   --dry-run         print the matrix and the KV preflight only
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-$HOME/huggingface/Qwen3-0.6B}"
RUNS="${RUNS:-3}"
EXPERIMENTS="${EXPERIMENTS:-r1,e1,e2,e3,combo,e4}"
DATASET_DIR="${DATASET_DIR:-bench/datasets}"
SCALE="${SCALE:-1.0}"
SKIP_SETUP=0
DRY=0
MOE_MODEL="${MOE_MODEL:-}"
PASSTHRU=()

if [[ -z "${PY:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PY=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  else
    PY="python"
  fi
fi

if [[ -z "${GPU_GIB:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_GIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%.1f", $1/1024}')"
fi
GPU_GIB="${GPU_GIB:-8}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) EXPERIMENTS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --moe-model) MOE_MODEL="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) PASSTHRU+=("$1"); shift ;;
  esac
done

if [[ ! -d "$MODEL" ]]; then
  echo "error: model dir not found: $MODEL" >&2
  exit 1
fi

echo "python    : $PY"
echo "dense     : $MODEL"
echo "MoE (E3)  : ${MOE_MODEL:-<not set -> E3 skipped>}"
echo "gpu       : ${GPU_GIB} GiB (dataset scale ${SCALE})"
echo

if [[ $SKIP_SETUP -eq 0 ]]; then
  "$PY" bench/nanovllm_bench.py prepare --model "$MODEL" --out "$DATASET_DIR" \
      --scale "$SCALE" --gpu-gib "$GPU_GIB"
  echo
fi

MOE_ARG=()
[[ -n "$MOE_MODEL" ]] && MOE_ARG=(--moe-model "$MOE_MODEL")

if [[ $DRY -eq 1 ]]; then
  "$PY" bench/nanovllm_bench.py run --model "$MODEL" --dataset-dir "$DATASET_DIR" \
      --out results --runs "$RUNS" --gpu-gib "$GPU_GIB" \
      --experiments "$EXPERIMENTS" --dry-run "${MOE_ARG[@]}"
  exit 0
fi

"$PY" bench/nanovllm_bench.py run --model "$MODEL" --dataset-dir "$DATASET_DIR" \
    --out results --runs "$RUNS" --gpu-gib "$GPU_GIB" \
    --experiments "$EXPERIMENTS" "${MOE_ARG[@]}" "${PASSTHRU[@]}"

"$PY" bench/nanovllm_bench.py analyze --raw results/raw --out BENCH_RESULTS.md

if ! compgen -G "results/raw/*.orig.json" > /dev/null; then
  echo
  echo "note: 还没有原版基线数据,报告里的 'vs 原版' 列会是 '-'"
  echo "      跑一次 bash bench/orig_baseline.sh 再重建报告即可补上:"
  echo "      $PY bench/nanovllm_bench.py analyze --raw results/raw --out BENCH_RESULTS.md"
fi
echo
echo "report:  BENCH_RESULTS.md"
echo "logs:    results/logs/"
echo "raw:     results/raw/"
