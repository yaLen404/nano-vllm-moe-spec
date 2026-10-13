<p align="center">
<img width="300" src="assets/logo.png">
</p>

# nano-vllm-moe-spec

A fork of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) that adds three
inference optimizations on top of upstream `bb823b3`. Each feature is measured
against its own control on the same workload, so the numbers below are
attributable to that feature alone.

| Feature | Control | Measured |
|---|---|---|
| MoE sparse experts | per-expert Python loop | **7.68×** throughput (Qwen3-30B-A3B, 128 experts) |
| Speculative decoding | speculative off | **1.7–2.7×** throughput (Qwen3-0.6B, batch 1–4, K=4/8) |
| Fair-quantum chunked prefill | upstream scheduler | short-request p99 TTFT **1.23 s → 0.12 s** |

---

## 1. MoE: Triton grouped GEMM

`nanovllm/layers/moe.py`, `nanovllm/models/qwen3_moe.py`

Routing is per token: a replicated router produces logits over all experts,
softmax, then top-k (8 of 128 for Qwen3-30B-A3B), with optional renormalization.

The expensive part is the FFN. The naive implementation loops over experts in
Python and issues one small GEMM per expert — 128 kernel launches per layer per
step. The fast path instead:

1. expands each token's top-k choices into `T×k` (token, expert) pairs,
2. sorts them by expert id so each expert's tokens are contiguous,
3. pads every expert to a multiple of `BLOCK_M = 64` to align with the Triton tile,
4. runs **two grouped GEMM kernels** (gate-up, then down) whose grid walks token
   blocks and indexes the stacked `[E, N, K]` expert weights by `expert_id`,
5. scatters results back with `index_add_`, weighting each contribution.

Both paths live in the file — `forward_triton` (fast) and `forward_reference`
(the per-expert loop) — so the two can be diffed directly.

**Measured** on Qwen3-30B-A3B: **168.0 tok/s vs 21.9 tok/s** for the per-expert
reference, i.e. **7.68×**. Throughput regression against upstream HEAD on the
dense path is 1.001×, so the change is confined to the MoE path.

**Where it does not win.** Padding is the tax. Utilization of padded slots is
only **2–8%** on decode-heavy batches and ~33% on prefill-heavy ones. At
`batch = 1` the grouped GEMM is **0.86×** — slower than the per-expert loop —
because sort/pad/scatter overhead and low GEMM occupancy outweigh the launch
savings. And because routing is data-dependent, MoE cannot be captured in a CUDA
graph (a captured graph would replay a stale top-k decision), so the MoE path
always runs eager.

---

## 2. Speculative decoding

`nanovllm/engine/speculative.py`, `nanovllm/engine/model_runner.py`

Drafting uses **prompt-lookup (n-gram, n=3)**: match the last 3 tokens against
history and propose what followed. Verification is **Leviathan rejection
sampling**, which accepts a draft token with probability `min(1, p_target / p_draft)`
and always emits one bonus/replacement token — so the output distribution is
identical to non-speculative decoding.

Two implementation changes carried the throughput:

- **Batched accept.** Softmax, the acceptance-prefix cumulative product and the
  bonus/replacement multinomial draws now run as a handful of batched kernels
  with exactly two GPU→CPU synchronizations, instead of a per-sequence Python
  loop with several syncs each.
- **Verify forward in a CUDA graph.** Profiling showed the verify forward
  (5 tokens × 28 layers) cost **74 ms** and dominated the step. It ran eager over
  prefill-shaped attention, so ~300 small kernel launches per step were the real
  cost — not FLOPs. Capturing it fixes the shapes (`max_k + 1` query tokens per
  sequence, `max_seqlen_k = max_model_len`) and only rewrites buffer values per
  replay. Verify dropped to **4.3 ms**.

Correctness: on identical inputs the graph path and the eager path produce
**bit-identical logits** (max abs diff `0.0`).

**Measured** — Qwen3-0.6B, `d3_spec_small`, output 1024, temperature 0.6, median
of 3 runs. The spec-off denominator is the fastest of the repeated baseline
measurements, i.e. the most conservative choice:

| batch | spec off | K=4 | K=8 | speedup (K=4 / K=8) |
|---|---|---|---|---|
| 1 | 157 tok/s | 349 | 428 | 2.2× / 2.7× |
| 2 | 257 | 442 | 547 | 1.7× / 2.1× |
| 4 | 472 | 923 | 850 | 2.0× / 1.8× |

---

## 3. Fair-quantum chunked prefill

`nanovllm/engine/scheduler.py`

A long prompt is sliced into budget-sized chunks that interleave with decode, so
one big prefill no longer monopolizes the step. On top of that, the budget is
split fairly across waiting requests:

```python
chunk = min(num_tokens, remaining, budget // len(waiting))
```

Without the fair split, a single queued request can consume the whole budget
while newly arrived short requests wait behind it. With it, under staggered
arrivals the **short-request p99 TTFT fell from 1.23 s to 0.12 s**.

---

## Quick start

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model", tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, nano-vLLM."], sampling_params)
print(outputs[0]["text"])
```

Speculative decoding is opt-in:

```python
llm = LLM("/path/to/model", speculative=True, max_speculative_tokens=8)
```

## Reproducing the benchmarks

`bench/nanovllm_bench.py` is a single-file harness carrying the workloads used
above:

```bash
python bench/nanovllm_bench.py prepare --model ~/huggingface/Qwen3-0.6B \
    --out bench/datasets
python bench/nanovllm_bench.py one --model ~/huggingface/Qwen3-0.6B \
    --dataset d3_spec_small.jsonl --dataset-dir bench/datasets \
    --tag e2_off --runs 3 --out results
python bench/nanovllm_bench.py one --model ~/huggingface/Qwen3-0.6B \
    --dataset d3_spec_small.jsonl --dataset-dir bench/datasets \
    --tag e2_k4 --runs 3 --out results --spec --spec-k 4
```

`bench/spec_ab.py` runs the whole speculative A/B (spec off vs K=4/8 at batch
1/2/4) in one shot. Raw per-feature outputs are checked in under `results_*/raw/`.

Hardware note: the speculative numbers were measured on an RTX 3060 Laptop
(6 GB); the 30B-A3B MoE numbers on an 84 GB card.

## What this fork does not include

Written down so the numbers above are not over-read:

- **No quantization.** There is no weight, activation or KV-cache quantization
  anywhere in the tree; inference runs in the model's native `bf16`/`fp16`.
- **No expert-parallel all-to-all.** Experts are sharded on the intermediate dim
  (ETP) and the router is replicated; there is no cross-rank token dispatch.
- **MoE cannot use CUDA graphs** (data-dependent routing), so it always runs eager.

## Acknowledgements

Built on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) by
GeeeekExplorer. The scheduler, paged KV cache, prefix caching and
tensor-parallel scaffolding are upstream work; this fork adds the features
documented above.
