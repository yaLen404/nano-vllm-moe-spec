# nano-vllm 消融测试方案

按社区通行做法:**每项单独测出贡献与边界 → 组合测出总收益与交互 → 回归确认无退化**。
工具链:`bench/nanovllm_bench.py`(单文件套件)+ `run_bench.sh`(一键入口)。

## 1. 四个自变量(一次只改一个)

| 实验 | 自变量 | 取值 | 固定 | 数据集 | 关键指标 |
|---|---|---|---|---|---|
| E1 chunked prefill | `max_num_batched_tokens` | 512 / 1024 / 2048 / 8192 / 16384 | dense, 无投机, tp=1 | T1(2048 长 + 128 短) | **按类别的 TTFT**、decode tok/s |
| E2 投机解码 | `speculative` + `max_speculative_tokens` | off / K=4 / K=8;temp 0.6 / 1.0 | dense, budget 默认 | T2 copy(高可预测)+ T2 open(对照) | α、每步提交 token 数、tok/s |
| E3 MoE | 模型(dense vs **完整 MoE 模型**) | 由 `MOE_MODEL=` 指定,未指定则跳过 E3 | **两边都 eager** | 按模型上下文自动选 | tok/s、峰值显存、激活专家数、不均衡度 |
| E4 TP | `tensor_parallel_size` | 2 / 4(**需 ≥2 张 GPU**) | dense, budget 默认 | T3 | tok/s、每卡显存 |

E3 另有组内机制消融:**Triton 分组 GEMM vs `forward_reference` 逐专家循环**。

## 2. 数据集

| 文件 | 内容 | 用途 |
|---|---|---|
| `t1_long.jsonl` | 8×2048 + 16×128,out 128 | E1(prefill 重) |
| `t2_predictable.jsonl` | 16 条"逐字复述"任务,out 512 | E2(α 高) |
| `t2_open.jsonl` | 16 条开放写作,out 512 | E2 边界(α≈0) |
| `t3_mixed.jsonl` | 2×3584 + 5×1024 + 12×128,out 256 | 组合实验、回归、E4 |
| `t4_short.jsonl` | 6×128 + 10×64 prompt,out 128 | 仅在模型上下文很小时自动启用 |

`prepare` 会打印 KV 需求/容量估算,并对"超出模型上下文"或"prompt+max_tokens > max_model_len"的请求告警。

## 3. 三张报告表(`analyze` 自动生成)

1. **Table 1 单项贡献**:每项相对各自对照的 speedup + 生效条件 + 按类别的 TTFT。
2. **Table 2 累积组合**(均在 T3):base → +chunked(1024) → +spec(K=4) → 两者,再与"单项之积"比较得到交互系数。
3. **Table 3 回归**:当前版本 vs 原版 checkout 同配置,退化 < 2% 视为无差异。

## 4. 硬件分层

| 环境 | 可做 |
|---|---|
| 单卡 6–8 GB | E1、E2、与回归(用 dense 0.6B) |
| 多卡 / 大显存 | 追加 E4;E3 用完整 MoE 模型(BF16 权重体积决定最低配置:30B-A3B ≈61 GB → 8×24 GB / 4×48 GB / 2×80 GB) |

nano-vllm 无量化路径(dtype 取自 `hf_config.torch_dtype`),所以 E3 的显存门槛就是权重体积。
TP 需 ≥2 张物理 GPU,且 NCCL 固定端口 2333 → 各配置必须串行。

## 5. 在目标机器上怎么跑

```bash
cd ~/nano-vllm
MODEL=/path/to/Qwen3-0.6B MOE_MODEL=/path/to/Qwen3-30B-A3B bash run_bench.sh --dry-run
MODEL=/path/to/Qwen3-0.6B MOE_MODEL=/path/to/Qwen3-30B-A3B bash run_bench.sh
```

不设 `MOE_MODEL` 就不跑 E3(其余实验照常)。E3 会自动挑一个不超过该模型上下文的测试集。

## 6. 与原版对比(Table 3)

在**原版 checkout** 里跑同一条命令,再把原始 json 拷回来:

```bash
python bench/nanovllm_bench.py one --tag r1_baseline --engine-tag orig \
    --model <MODEL> --dataset t3_mixed.jsonl --dataset-dir bench/datasets \
    --out results_orig --runs 3
cp results_orig/raw/r1_baseline.json results/raw/r1_baseline.orig.json
python bench/nanovllm_bench.py analyze --raw results/raw --out BENCH_RESULTS.md
```

## 7. 在 6 GB 卡上做过的冒烟验证(Qwen3-0.6B)

| 配置 | tok/s | 说明 |
|---|---|---|
| dense,t3(3 条长请求) | 207.7 | CUDA Graph 开 |
| 投机 K=4,t2 copy | 92.3,**α=0.599** | 复述任务确实能命中,接受到统计正常 |
| E1 budget 512 | 447.4 | long TTFT 4.27 s / short 2.10 s |
| E1 budget 16384 | 762.8 | long TTFT 1.43 s / short 1.55 s |

这些只是流程验证,不作为结论;正式数字在目标机器上用同一脚本产出。
单配置启动约 110 s(CUDA Graph capture 占大头;MoE 强制 eager 时约 11 s)。

## 8. 判读规则与坑

1. 同配置 3 轮波动若为 ±2%,小于波动即记"无差异"。
2. **E1 要看按类别的 TTFT**:离线批量(所有请求同时提交)下 chunked prefill 只体现为吞吐损失;
   它的收益是"短请求不被长 prefill 挡住",报告已单列 `TTFT p50 by request class`。
3. E3 是跨模型对比,**不可归因**为"MoE 加速 X 倍";价值在 Triton vs 参考实现的组内差值。
4. 吞吐用**实际产出 token**,不是 Σ max_tokens;计时必须 `torch.cuda.synchronize()`。
5. 投机解码必须用真实重复文本,随机 token 负载 α≈0。
6. 每配置独立进程(脚本已保证);warmup、图捕获、compile 首次编译不计入。
7. KV 需求接近容量会触发 preemption,污染结果(preflight 会告警)。
8. **MoE 被强制 eager**(model_runner.py:35-38),对比时 dense 侧也要 `--no-cuda-graph`。
9. **prompt + max_tokens 不得超过 `--max-model-len`**:CUDA Graph 的 `block_tables` 按
   `max_model_len` 固定宽度,超出会在 model_runner.py:228 直接崩(实测)。
10. **prompt + max_tokens 不得超过模型的 `max_position_embeddings`**:RoPE 的
    `cos_sin_cache[positions]` 会越界(rotary_embedding.py:44,表现为 torch.compile 的
    `index out of bounds ... < 256` 断言)。套件会提前告警并在 `one` 里直接报错退出。
11. TP 配置串行跑,端口 2333 冲突。
