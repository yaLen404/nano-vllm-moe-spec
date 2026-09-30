# nano-vllm 分功能性能结论(per-feature)

本文回答:**每一项优化在它自己生效的条件下,到底带来多少提升;在什么条件下它是负收益。**

数据来源两类,每行都标注了硬件:
- **[远程]** AutoDL RTX 6000D 84GB / 92GiB RAM,完整模型 Qwen3-30B-A3B(56.9GiB,48 层 128 专家)与 Qwen3-0.6B
- **[本地]** RTX 3060 Laptop 6GB,Qwen3-0.6B / tiny-MoE(79M,4 专家);数据集按 6GB 显存缩放

---

## 1. chunked prefill(`max_num_batched_tokens`)

### 1a. 饱和 prefill —— 功能正向生效 [本地]

负载:`d2_saturating`(约 7 万 prompt token,一次性提交,out 512),即"总 prompt 装不进一个 step"。

| budget | tok/s | 提升 vs 16384 | TTFT p50 |
|---|---|---|---|
| 16384 | 432.8 | 1.000× | 0.490 s |
| 4096 | **552.3** | **1.276×** | **0.180 s** |
| 1024 | **547.5** | **1.265×** | **0.167 s** |

### 1b. 错峰到达 —— 当前实现是负收益,原因在分配策略 [本地]

负载:`d1_interactive`(4 条 3584 长请求先到 + 10 条 128 短请求每 100ms 到达),看**短请求**延迟。

| budget | 短请求 p50 | 短请求 **p99** | 吞吐 tok/s |
|---|---|---|---|
| 16384 | 0.051 | 0.070 | 612.6 |
| 4096 | 0.053 | **0.534** | 605.2 |
| 1024 | 0.055 | **0.909** | 609.6 |

`nanovllm/engine/scheduler.py` 的 `chunk = min(num_tokens, remaining)`:一步里队首请求可以吃掉整个 budget,新到的短请求排在所有 chunk 之后 → 小 budget 下尾巴延迟变差 13 倍。

### 1c. 同一负载 + 公平配额(一行改动,harness 侧实验补丁 `--fair-chunk`)[本地]

`chunk = min(num_tokens, remaining, budget // len(waiting))`

| budget | 现状 p99 | 配额后 p99 | 现状->配额 吞吐 |
|---|---|---|---|
| 16384 | 0.070 | 0.071 | 612.6 -> 630.1 |
| 4096 | 0.534 | **0.241**(2.2×) | 605.2 -> 621.4 |
| 1024 | 0.909 | **0.123**(**7.4×**) | 609.6 -> 617.8 |

饱和负载下加配额不损失吞吐(556.0 / 545.2 vs 552.3 / 547.5)。

**结论**:chunked prefill 的价值(短请求不被长 prefill 挡住)只有在分配策略公平时才出现;当前实现应采纳该改动。

---

## 2. 投机解码(n-gram prompt-lookup,拒绝采样)

### 2a. 并发 16(旧的错误 regime)[远程]

| 配置 | tok/s | 提升 vs 不开 | α |
|---|---|---|---|
| 不开(复述任务) | 4864.5 | 1.000× | - |
| K=4 | 847.7 | 0.174× | 0.690 |
| K=8 | 862.0 | 0.177× | 0.506 |

### 2b. 小批量(投机该生效的 regime)[本地,out 1024]

| 配置 | tok/s | 提升 vs 不开 | α | 每步提交 token |
|---|---|---|---|---|
| batch 1 不开 | 135.9 | 1.000× | - | - |
| batch 1 K=4 | 82.2 | 0.604× | 0.754 | 4.01 |
| batch 2 K=4 | 95.6 | 0.384× | 0.765 | 3.66 |
| batch 4 K=4 | 98.8 | 0.390× | 0.710 | 3.29 |

### 2c. K 的敏感性(batch 1)[本地]

| K | tok/s | 提升 vs 不开 | α | 每步提交 token |
|---|---|---|---|---|
| 2 | 55.5 | 0.41× | 0.805 | 2.61 |
| 4 | 82.2 | 0.60× | 0.754 | 4.01 |
| **8** | **129.8** | **0.96×** | 0.779 | **5.37** |

**结论**:瓶颈是**每步的固定开销**(accept 逐序列 Python 循环 + 每序列 GPU->CPU 同步 + verify 走 prefill 形状前向、不进 CUDA Graph)。K 越大摊得越薄,α 在 K=8 基本不降。当前实现应:K=8 起步;把 accept 批量化(一次 softmax / 一次 multinomial / 只在末尾同步);让 verify 用 `flash_attn_with_kvcache` 以便进图。**30B MoE + 小批量**尚未测(需远程),那里每步模型开销占比更高,是投机最可能翻正的场景。

---

## 3. MoE(grouped GEMM vs 逐专家循环)

### 3a. 大 prefill [远程,30B-A3B,t3_mixed 19 请求]

| 路径 | tok/s | 提升 | TTFT p50 |
|---|---|---|---|
| Triton 分组 GEMM | **168.0** | **7.679×** | 0.321 s |
| 逐专家循环 | 21.9 | 基线 | 1.656 s |
| Triton(prefill 重载 t1_long) | 206.1 | - | 0.373 s |

### 3b. batch 相关性 [本地 tiny-MoE,4 专家;decode 重载]

| batch | Triton tok/s | 逐专家 tok/s | 倍数 | 补齐效率 real/padded | 每专家每步 slot |
|---|---|---|---|---|---|
| 1 | 25.5 | 29.7 | **0.86×** | 2.1% | 0.68 |
| 4 | 156.6 | 132.9 | 1.18× | 4.8% | 2.72 |
| 8 | 369.3 | 297.5 | 1.24× | 8.4% | 5.43 |
| prefill 重载 budget 2048 | 482.1 | 399.6 | 1.21× | 33.7% | - |
| prefill 重载 budget 512 | 326.3 | 300.3 | 1.09× | 31.6% | - |

**结论**:融合路径的优势是 **batch 相关**的。小 batch/decode 时每专家只有不到 1 个 slot,`moe_align` 补齐到 `_BLOCK_M=64` → 97.9% 的 GEMM 算力打在 padding 上,于是**比逐专家循环还慢**。大 prefill 时补齐效率升到 ~33%,30B 上取得 7.68×。若要 decode 也受益,需要按 batch 自适应 tile(或更小的 BLOCK_M / fused MoE kernel)。

### 3c. 内存(必须修)[远程,实测]

`FusedMoE._stack_weights()` 为每层缓存一份全量专家权重副本(+1.1 GiB/层),而原参数因参数<->`weight_loader` 引用环**不释放**:实测 `experts = None` + `empty_cache()` 显存纹丝不动(58.0 GiB),加 `gc.collect()` 才降到 56.9 GiB。后果:30B-A3B 峰值约 114 GiB,84GB 卡上 Triton 路径必 OOM。修法:堆叠后 `del` + `gc.collect()`(或加载期一次性堆叠、或改成专家权重指针表)。

---

## 4. 回归(vs 改动前 HEAD)[远程]

| 引擎 | tok/s | 提速 |
|---|---|---|
| 当前版本 | 4024.2 | **1.001×** |
| 原版 HEAD | 4021.5 | 1.000× |

单项 vs 原版:0.978×–1.005×(budget 512 时略低,E1 其余持平)。

---

## 5. 尚未测得(需要多卡/远程实例)

| 项 | 为什么没测 |
|---|---|
| E2c:30B MoE + batch 1/2/4 的投机 | 远程 SSH 端点失效,需新实例地址 |
| E3b:30B 上的 decode 重载 MoE(batch 相关结论是否在 128 专家下同样成立) | 同上 |
| E4:TP=2/4 | 单卡 |

## 6. 复现方式

```bash
# 数据与矩阵(每项功能的生效条件)
python bench/nanovllm_bench.py prepare --model <MODEL> --out bench/datasets --gpu-gib <VRAM>
bash run_bench.sh --skip-setup --only e1,e2,e3,combo,r1      # 原口径
bash bench/orig_baseline.sh                                   # 原版基线(补 vs 原版列)

# 分功能口径(错峰/饱和/小批量/decode 重载)
bash /root/run_perfeature_remote.sh      # 见 bench/ 内同名脚本
```
关键开关:`--fair-chunk`(公平配额实验)、`--moe-free-experts`(MoE 显存钩子)、`--budget`、`--limit`、`--gpu-util`。
