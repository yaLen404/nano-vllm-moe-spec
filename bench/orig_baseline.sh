#!/usr/bin/env bash
# Run the shared configurations on the ORIGINAL code (git HEAD) so that the report
# can print a "vs 原版" column for every row.
#
#   bash bench/orig_baseline.sh              # all shared configs
#   bash bench/orig_baseline.sh r1_baseline e1_budget512
#
# Only the configurations both versions support are run here: the original has no
# speculative decoding, no MoE and no chunked-prefill budget control of its own, so
# E2/E3 (and the spec part of the combo) have no original counterpart.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
WORK="$ROOT/.orig-worktree"
DATASET_DIR="${DATASET_DIR:-bench/datasets}"
OUT="${OUT:-results}"
RUNS="${RUNS:-3}"
GPU_GIB="${GPU_GIB:-}"

if [[ -z "${PY:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then PY=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then PY="python3"
  else PY="python"; fi
fi
if [[ -z "$GPU_GIB" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_GIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%.1f", $1/1024}')"
fi
GPU_GIB="${GPU_GIB:-8}"

# tag:dataset:extra-args
ALL_CONFIGS=(
  "r1_baseline:t3_mixed.jsonl:"
  "e1_budget512:t1_long.jsonl:--budget 512"
  "e1_budget1024:t1_long.jsonl:--budget 1024"
  "e1_budget2048:t1_long.jsonl:--budget 2048"
  "e1_budget8192:t1_long.jsonl:--budget 8192"
  "e1_budget16384:t1_long.jsonl:--budget 16384"
  "c1_chunk_t3:t3_mixed.jsonl:--budget 1024"
)
if [[ $# -gt 0 ]]; then
  CONFIGS=()
  for want in "$@"; do
    for c in "${ALL_CONFIGS[@]}"; do
      [[ "${c%%:*}" == "$want" ]] && CONFIGS+=("$c")
    done
  done
  [[ ${#CONFIGS[@]} -gt 0 ]] || { echo "no matching configs for: $*" >&2; exit 1; }
else
  CONFIGS=("${ALL_CONFIGS[@]}")
fi

echo "worktree : $WORK (detached at HEAD)"
git worktree add --detach --force "$WORK" HEAD >/dev/null

# the suite and the datasets are untracked, so bring them along
cp -r bench "$WORK/"
cp run_bench.sh "$WORK/" 2>/dev/null || true
rm -rf "$WORK/bench/datasets" "$WORK/results"
cp -r "$DATASET_DIR" "$WORK/bench/datasets" 2>/dev/null || true

# A 'pip install -e .' editable install uses a meta_path finder that wins over
# sys.path, which would silently import the *modified* tree. sitecustomize runs
# after the .pth files, so it can drop that finder here.
cat > "$WORK/sitecustomize.py" <<'PYEOF'
import sys
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in type(f).__module__.lower()
                 and "editable" not in type(f).__name__.lower()]
PYEOF

# Minimal compatibility patch: HEAD reads hf_config.dtype, which transformers 4.5x
# no longer provides (the working tree already fixed this). Performance-neutral.
"$PY" - "$WORK/nanovllm/engine/model_runner.py" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
s2 = s.replace(
    "torch.set_default_dtype(hf_config.dtype)",
    'dtype_attr = getattr(hf_config, "torch_dtype", None) or "float32"\n'
    '        torch.set_default_dtype(getattr(torch, dtype_attr) if isinstance(dtype_attr, str) else dtype_attr)')
s2 = s2.replace("hf_config.dtype.itemsize", "torch.get_default_dtype().itemsize")
if s2 == s:
    print("compat patch: nothing to do (already fixed)")
else:
    open(p, "w").write(s2)
    print("compat patch: hf_config.dtype -> torch dtype (performance-neutral)")
PYEOF

mkdir -p "$OUT/raw" "$OUT/logs"
failed=0
cd "$WORK"      # make the baseline tree the cwd too: Python prepends cwd for -c/interactive runs
for entry in "${CONFIGS[@]}"; do
  tag="${entry%%:*}"; rest="${entry#*:}"; ds="${rest%%:*}"; extra="${rest#*:}"
  echo "=== orig $tag ($ds ${extra}) ==="
  # shellcheck disable=SC2086
  if PYTHONPATH="$WORK" "$PY" "$WORK/bench/nanovllm_bench.py" one \
      --tag "$tag" --engine-tag orig --expect-root "$WORK" \
      --model "${MODEL:-$HOME/huggingface/Qwen3-0.6B}" --dataset "$ds" \
      --dataset-dir "$WORK/bench/datasets" --out "$WORK/results" \
      --runs "$RUNS" --gpu-util "${GPU_UTIL:-0.9}" $extra \
      > "$OUT/logs/${tag}.orig.log" 2>&1; then
    cp "$WORK/results/raw/$tag.json" "$OUT/raw/$tag.orig.json"
    tail -2 "$OUT/logs/${tag}.orig.log" | head -1
  else
    echo "  FAILED (see $OUT/logs/${tag}.orig.log)"
    tail -5 "$OUT/logs/${tag}.orig.log"
    failed=$((failed + 1))
  fi
done

git worktree remove --force "$WORK"
echo
echo "baseline files: $OUT/raw/*.orig.json"
echo "now rebuild the report:  $PY bench/nanovllm_bench.py analyze --raw $OUT/raw --out BENCH_RESULTS.md"
[[ $failed -eq 0 ]] || echo "WARNING: $failed config(s) failed" >&2
