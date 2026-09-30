# nano-vllm ablation results

- configurations found: 18
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- note: every row is the median of 3 runs

- **提升 = 对照项耗时 ÷ 本行耗时,>1 就是比对照快**;每行的对照项见 condition
- `vs 原版` = 与改动前 HEAD 同配置的对比(仅原版能跑的配置有值,先跑 `bash bench/orig_baseline.sh`)

## Table 1 - per-item contribution (one variable at a time)

| item | config | dataset | tok/s | 提升 | vs 原版 | TTFT p50 | TTFT p99 | α | 每步token | 对照项 |
|---|---|---|---|---|---|---|---|---|---|---|
| **E1 chunked prefill** (对照 = budget 16384,即不切分) |
| budget 512 | `e1_budget512` | - | - | - | - | - | - | - | - | not run |
| budget 1024 | `e1_budget1024` | - | - | - | - | - | - | - | - | not run |
| budget 2048 | `e1_budget2048` | - | - | - | - | - | - | - | - | not run |
| budget 8192 | `e1_budget8192` | - | - | - | - | - | - | - | - | not run |
| budget 16384 | `e1_budget16384` | - | - | - | - | - | - | - | - | not run |
| **E1b chunked prefill — 错峰到达(d1_interactive,对照 = budget 16384)** |
| budget 16384 | `e1i_budget16384` | d1_interactive.jsonl | 612.6 | **1.000x** | - | 0.070 | 1.276 | - | - | 长请求先到,短请求每 100ms 到达 → 看短请求 TTFT |
| budget 4096 | `e1i_budget4096` | d1_interactive.jsonl | 605.2 | **0.988x** | - | 0.098 | 1.448 | - | - | 长请求先到,短请求每 100ms 到达 → 看短请求 TTFT |
| budget 1024 | `e1i_budget1024` | d1_interactive.jsonl | 609.6 | **0.995x** | - | 0.100 | 2.047 | - | - | 长请求先到,短请求每 100ms 到达 → 看短请求 TTFT |
| **E1c chunked prefill — 饱和 prefill(d2_saturating,~70k token)** |
| budget 16384 | `e1s_budget16384` | d2_saturating.jsonl | 432.8 | **1.000x** | - | 0.490 | 1.432 | - | - | 对照项:budget 16384 |
| budget 4096 | `e1s_budget4096` | d2_saturating.jsonl | 552.3 | **1.276x** | - | 0.180 | 1.533 | - | - | 对照项:budget 16384 |
| budget 1024 | `e1s_budget1024` | d2_saturating.jsonl | 547.5 | **1.265x** | - | 0.167 | 2.079 | - | - | 对照项:budget 16384 |
| **E2 speculative** (对照 = 同一数据集不开投机) |
| spec_k4_t06 | `e2_spec_k4_t06` | - | - | - | - | - | - | - | - | not run |
| spec_k8_t06 | `e2_spec_k8_t06` | - | - | - | - | - | - | - | - | not run |
| spec_k8_t10 | `e2_spec_k8_t10` | - | - | - | - | - | - | - | - | not run |
| spec_k8_open | `e2_spec_k8_open` | - | - | - | - | - | - | - | - | not run |
| **E2b speculative — 小批量(0.6B,d3_spec_small,out 1024)** |
| batch 1, K=4 | `e2s_dense_k4_n1` | d3_spec_small.jsonl | 82.2 | **0.604x** | - | 0.058 | 0.180 | 0.754 | 4.01 | 对照项:同 batch 不开投机 |
| batch 2, K=4 | `e2s_dense_k4_n2` | d3_spec_small.jsonl | 95.6 | **0.384x** | - | 0.081 | 0.622 | 0.765 | 3.66 | 对照项:同 batch 不开投机 |
| batch 4, K=4 | `e2s_dense_k4_n4` | d3_spec_small.jsonl | 98.8 | **0.390x** | - | 0.088 | 0.637 | 0.710 | 3.29 | 对照项:同 batch 不开投机 |
| **E2c speculative — 小批量(完整 30B MoE,这是投机该生效的 regime)** |
| batch 1, K=4 | `e2s_moe_k4_n1` | - | - | - | - | - | - | - | - | not run |
| batch 2, K=4 | `e2s_moe_k4_n2` | - | - | - | - | - | - | - | - | not run |
| batch 4, K=4 | `e2s_moe_k4_n4` | - | - | - | - | - | - | - | - | not run |
| **E3 MoE** (完整 30B-A3B;Triton 分组 GEMM vs 逐专家循环) |
| prefill 重载 Triton | `e3_moe_main` | - | - | - | - | - | - | - | - | not run |
| decode 重载 Triton | `e3d_triton` | - | - | - | - | - | - | - | - | not run |
| 大 prefill Triton | `e3p_triton` | - | - | - | - | - | - | - | - | not run |
| MoE 逐专家循环(19 请求) | `e3_moe_ref` | - | - | - | - | - | - | - | - | not run |
| **E4 TP** (对照 = tp=1) |
| tp=2 | `e4_tp2_t3` | - | - | - | - | - | - | - | - | not run |
| tp=4 | `e4_tp4_t3` | - | - | - | - | - | - | - | - | not run |

### TTFT p50 by request class (seconds)

| config | copy | long | mid | prompt128 | prompt64 | short |
|---|---|---|---|---|---|---|
| `e1i_budget1024` | - | 0.101 | - | - | - | 0.055 |
| `e1i_budget16384` | - | 0.107 | - | - | - | 0.051 |
| `e1i_budget4096` | - | 0.101 | - | - | - | 0.053 |
| `e1if_budget1024` | - | 0.100 | - | - | - | 0.045 |
| `e1if_budget16384` | - | 0.180 | - | - | - | 0.042 |
| `e1if_budget4096` | - | 0.098 | - | - | - | 0.048 |
| `e1s_budget1024` | - | 0.092 | 0.167 | - | - | 0.229 |
| `e1s_budget16384` | - | 0.490 | 0.490 | - | - | 0.490 |
| `e1s_budget4096` | - | 0.180 | 0.180 | - | - | 0.180 |
| `e1sf_budget1024` | - | 0.230 | 0.230 | - | - | 0.230 |
| `e1sf_budget4096` | - | 0.190 | 0.190 | - | - | 0.190 |
| `e2s_dense_k4_n1` | 0.058 | - | - | - | - | - |
| `e2s_dense_k4_n2` | 0.081 | - | - | - | - | - |
| `e2s_dense_k4_n4` | 0.088 | - | - | - | - | - |
| `e2s_dense_off_n1` | 0.062 | - | - | - | - | - |
| `e2s_dense_off_n2` | 0.052 | - | - | - | - | - |
| `e2s_dense_off_n4` | 0.063 | - | - | - | - | - |
| `loctiny_moe` | - | - | - | 0.032 | 0.032 | - |

## Table 2 - cumulative combination (all on t3_mixed)

| step | config | tok/s | speedup vs base | incremental |
|---|---|---|---|---|
| baseline (graph on) | - | - | - | not run |
| + chunked prefill (budget 1024) | - | - | - | not run |
| + speculative K=4 (alone) | - | - | - | not run |
| + both | - | - | - | not run |

## Table 3 - regression vs the original nano-vllm

_no original-engine data found._ To produce it, run the same command in the original checkout:

```bash
python bench/nanovllm_bench.py one --tag r1_baseline --engine-tag orig \
    --model <MODEL> --dataset t3_mixed.jsonl --dataset-dir bench/datasets \
    --out results_orig --runs 3 --no-cuda-graph false
cp results_orig/raw/r1_baseline.json results/raw/r1_baseline.orig.json
```

## Appendix - all raw rows

| tag | engine | dataset | tp | budget | graph | spec | K | tok/s | vs 原版 | wall s | spread s | TTFT p50 | TTFT p99 | peak GiB | alpha | startup s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `e1i_budget1024` | current | d1_interactive.jsonl | 1 | 1024 | True | False | 8 | 609.6 | - | 2.94 | 1.25 | 0.100 | 2.047 | 3.47 | - | 90.6 |
| `e1i_budget16384` | current | d1_interactive.jsonl | 1 | 16384 | True | False | 8 | 612.6 | - | 2.93 | 0.61 | 0.070 | 1.276 | 3.66 | - | 90.4 |
| `e1i_budget4096` | current | d1_interactive.jsonl | 1 | 4096 | True | False | 8 | 605.2 | - | 2.96 | 0.61 | 0.098 | 1.448 | 3.50 | - | 91.0 |
| `e1if_budget1024` | current | d1_interactive.jsonl | 1 | 1024 | True | False | 8 | 617.8 | - | 2.90 | 1.17 | 0.063 | 1.977 | 3.72 | - | 89.6 |
| `e1if_budget16384` | current | d1_interactive.jsonl | 1 | 16384 | True | False | 8 | 630.1 | - | 2.84 | 0.49 | 0.071 | 1.119 | 3.66 | - | 87.5 |
| `e1if_budget4096` | current | d1_interactive.jsonl | 1 | 4096 | True | False | 8 | 621.4 | - | 2.88 | 0.87 | 0.097 | 1.619 | 3.72 | - | 88.9 |
| `e1s_budget1024` | current | d2_saturating.jsonl | 1 | 1024 | True | False | 8 | 547.5 | - | 9.35 | 1.93 | 0.167 | 2.079 | 3.72 | - | 93.0 |
| `e1s_budget16384` | current | d2_saturating.jsonl | 1 | 16384 | True | False | 8 | 432.8 | - | 11.83 | 0.92 | 0.490 | 1.432 | 3.72 | - | 90.2 |
| `e1s_budget4096` | current | d2_saturating.jsonl | 1 | 4096 | True | False | 8 | 552.3 | - | 9.27 | 1.52 | 0.180 | 1.533 | 3.72 | - | 93.8 |
| `e1sf_budget1024` | current | d2_saturating.jsonl | 1 | 1024 | True | False | 8 | 545.2 | - | 9.39 | 1.84 | 0.230 | 2.131 | 3.93 | - | 92.9 |
| `e1sf_budget4096` | current | d2_saturating.jsonl | 1 | 4096 | True | False | 8 | 556.0 | - | 9.21 | 1.78 | 0.190 | 1.906 | 3.85 | - | 90.5 |
| `e2s_dense_k4_n1` | current | d3_spec_small.jsonl | 1 | 16384 | True | True | 4 | 82.2 | - | 12.46 | 2.37 | 0.058 | 0.180 | 3.29 | 0.754 | 93.2 |
| `e2s_dense_k4_n2` | current | d3_spec_small.jsonl | 1 | 16384 | True | True | 4 | 95.6 | - | 21.43 | 5.54 | 0.081 | 0.622 | 3.29 | 0.765 | 93.3 |
| `e2s_dense_k4_n4` | current | d3_spec_small.jsonl | 1 | 16384 | True | True | 4 | 98.8 | - | 20.73 | 3.02 | 0.088 | 0.637 | 3.29 | 0.710 | 93.0 |
| `e2s_dense_off_n1` | current | d3_spec_small.jsonl | 1 | 16384 | True | False | 8 | 135.9 | - | 7.53 | 0.16 | 0.062 | 0.211 | 3.29 | - | 93.9 |
| `e2s_dense_off_n2` | current | d3_spec_small.jsonl | 1 | 16384 | True | False | 8 | 248.7 | - | 8.24 | 0.58 | 0.052 | 0.681 | 3.29 | - | 93.3 |
| `e2s_dense_off_n4` | current | d3_spec_small.jsonl | 1 | 16384 | True | False | 8 | 253.6 | - | 8.07 | 1.64 | 0.063 | 1.583 | 3.29 | - | 87.7 |
| `loctiny_moe` | current | t4_short.jsonl | 1 | 16384 | False | False | 8 | 148.6 | - | 3.45 | 0.71 | 0.032 | 0.597 | 3.66 | - | 10.0 |
