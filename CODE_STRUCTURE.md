# Nano-vLLM 代码结构梳理

本文档基于当前仓库源码，对 Nano-vLLM 的目录结构、核心模块职责和一次推理请求的执行链路做整体梳理，便于后续阅读、调试和扩展。

## 1. 项目定位

Nano-vLLM 是一个轻量版 vLLM 风格离线推理框架，主要目标是用尽量少的代码实现：

- 批量离线生成接口
- Prefill / Decode 两阶段调度
- KV cache block 管理
- Prefix cache
- Tensor Parallelism
- CUDA Graph 加速 decode
- Qwen3 Causal LM 推理模型

公开 API 很薄，用户主要通过 `LLM` 和 `SamplingParams` 使用框架。

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(["Hello"], SamplingParams(temperature=0.6, max_tokens=128))
```

## 2. 顶层目录结构

```text
.
├── README.md                 # 项目说明、安装、模型下载、快速开始
├── pyproject.toml            # Python 包元数据和依赖声明
├── example.py                # 最小推理示例
├── bench.py                  # 吞吐 benchmark 示例
├── assets/                   # README 使用的图片资源
└── nanovllm/                 # 框架主代码
    ├── __init__.py           # 导出 LLM 和 SamplingParams
    ├── llm.py                # LLM 类，继承 LLMEngine
    ├── config.py             # 推理配置
    ├── sampling_params.py    # 采样配置
    ├── engine/               # 请求生命周期、调度、模型运行器、KV block 管理
    ├── layers/               # 模型基础层、并行线性层、attention、sampler
    ├── models/               # Qwen3 模型结构
    └── utils/                # 全局上下文和权重加载工具
```

## 3. 包入口和配置层

### 3.1 `nanovllm/__init__.py`

对外只导出两个对象：

- `LLM`：推理引擎入口
- `SamplingParams`：采样参数

这让用户侧调用保持和 vLLM 风格相近。

### 3.2 `nanovllm/llm.py`

`LLM` 当前只是 `LLMEngine` 的空子类：

```python
class LLM(LLMEngine):
    pass
```

因此所有初始化、请求添加、生成循环都在 `engine/llm_engine.py` 中实现。

### 3.3 `nanovllm/config.py`

`Config` 是框架级配置对象，负责：

- 保存模型目录和 batch 上限
- 保存最大上下文长度、GPU 显存利用率、Tensor Parallel 大小
- 保存 KV cache block 大小和 block 数量
- 从 Hugging Face 模型目录加载 `AutoConfig`
- 将 `max_model_len` 限制在模型 `max_position_embeddings` 以内

关键字段：

| 字段 | 作用 |
| --- | --- |
| `model` | 本地模型目录 |
| `max_num_batched_tokens` | 单轮 prefill 最多处理 token 数 |
| `max_num_seqs` | 单轮最多调度序列数 |
| `max_model_len` | 最大上下文长度 |
| `gpu_memory_utilization` | KV cache 可使用的显存比例 |
| `tensor_parallel_size` | Tensor Parallel 进程数 / GPU 数 |
| `enforce_eager` | 是否禁用 CUDA Graph |
| `kvcache_block_size` | KV cache block 粒度，默认 256 |
| `num_kvcache_blocks` | warmup 后按显存动态计算 |

### 3.4 `nanovllm/sampling_params.py`

`SamplingParams` 当前支持三个参数：

- `temperature`：采样温度，要求大于 `1e-10`
- `max_tokens`：最大生成 token 数
- `ignore_eos`：是否忽略 EOS token

当前实现不支持 greedy sampling，因为 `temperature` 不能接近 0。

## 4. Engine 层职责

`nanovllm/engine/` 是整个框架的核心，负责把用户请求变成可调度的序列，管理 KV cache，并驱动模型执行。

```text
engine/
├── llm_engine.py      # 对外 generate 主循环
├── scheduler.py       # prefill / decode 调度器
├── sequence.py        # 单个请求序列状态
├── block_manager.py   # KV cache block 和 prefix cache 管理
└── model_runner.py    # CUDA 模型执行、TP 多进程、CUDA Graph
```

### 4.1 `LLMEngine`

`LLMEngine` 是用户请求进入框架后的主控对象。

初始化流程：

1. 根据传入参数构造 `Config`
2. 设置全局 `Sequence.block_size`
3. 按 `tensor_parallel_size` 创建 worker 进程
4. 创建 rank 0 的 `ModelRunner`
5. 加载 tokenizer
6. 设置 EOS token id
7. 创建 `Scheduler`
8. 注册退出清理逻辑

主要方法：

| 方法 | 作用 |
| --- | --- |
| `add_request()` | 将 prompt 编码为 token，并封装成 `Sequence` |
| `step()` | 执行一次调度 + 模型运行 + 后处理 |
| `is_finished()` | 判断 waiting / running 队列是否清空 |
| `generate()` | 对外批量生成接口，循环调用 `step()` 直到结束 |
| `exit()` | 退出模型 runner 和 TP worker |

### 4.2 `Sequence`

`Sequence` 表示一个请求的完整生命周期。

它保存：

- `seq_id`：请求 ID，用于最终按原顺序返回结果
- `status`：`WAITING`、`RUNNING`、`FINISHED`
- `token_ids`：prompt + completion 的完整 token 序列
- `num_prompt_tokens`：prompt token 数量
- `num_cached_tokens`：已经写入 / 命中 KV cache 的 token 数
- `num_scheduled_tokens`：本轮被调度的 token 数
- `block_table`：该序列逻辑 block 到物理 KV block 的映射
- 采样参数：`temperature`、`max_tokens`、`ignore_eos`

它也提供一些便捷属性：

- `prompt_token_ids`
- `completion_token_ids`
- `num_completion_tokens`
- `num_blocks`
- `last_block_num_tokens`

### 4.3 `Scheduler`

调度器维护两个队列：

- `waiting`：还在 prefill 或等待资源的序列
- `running`：已经完成 prefill、进入 decode 的序列

一次 `schedule()` 分为两种路径。

#### Prefill 路径

当 `waiting` 非空时优先做 prefill：

1. 从 `waiting` 队头取序列
2. 通过 `BlockManager.can_allocate()` 判断是否可分配 KV block，并计算 prefix cache 命中数量
3. 如果有 prefix cache 命中，只调度未缓存 token
4. 若 batch token 预算不足，只允许第一个序列做 chunked prefill
5. prefill 完成后把序列从 `waiting` 移入 `running`

返回值为 `(scheduled_seqs, True)`。

#### Decode 路径

当没有 prefill 任务时执行 decode：

1. 从 `running` 取序列
2. 检查追加一个 token 是否需要新 KV block
3. 如果 KV block 不足，会 preempt 某些 running 序列，将其释放并放回 `waiting`
4. 成功调度的序列每轮只生成 1 个 token
5. 调度完成后把这些序列放回 `running` 队列头部

返回值为 `(scheduled_seqs, False)`。

#### Postprocess

`postprocess()` 负责：

- 给已完成 block 计算 hash，用于 prefix cache
- 更新 `num_cached_tokens`
- 将采样得到的新 token 追加到序列
- 遇到 EOS 或达到 `max_tokens` 时标记完成
- 释放完成序列占用的 KV blocks

### 4.4 `BlockManager`

`BlockManager` 管理 KV cache 的物理 block。

核心数据结构：

| 字段 | 作用 |
| --- | --- |
| `blocks` | 所有物理 block 对象 |
| `free_block_ids` | 空闲 block 队列 |
| `used_block_ids` | 正在使用的 block 集合 |
| `hash_to_block_id` | prefix hash 到物理 block 的映射 |

每个 `Block` 保存：

- `block_id`
- `ref_count`
- `hash`
- `token_ids`

Prefix cache 机制：

1. 每个完整 block 用 `xxhash` 计算 hash
2. hash 会串联前一个 block 的 hash，因此表示一段连续前缀
3. 新请求 prefill 前逐 block 查找是否命中已有 hash
4. 命中且 token 内容一致时复用已有 KV block
5. 被复用 block 增加 `ref_count`

### 4.5 `ModelRunner`

`ModelRunner` 负责真正的 CUDA 侧模型执行。

初始化流程：

1. 初始化 NCCL process group
2. 设置当前 CUDA device
3. 临时设置默认 dtype 和默认 device
4. 构造 `Qwen3ForCausalLM`
5. 从本地 safetensors 加载权重
6. 创建 `Sampler`
7. warmup 模型，估算峰值显存
8. 根据剩余显存分配 KV cache
9. 如果未强制 eager，则 capture decode CUDA Graph

多卡 / TP 模式：

- rank 0 由主进程直接持有
- rank > 0 由 `multiprocessing` spawn 出 worker 进程
- rank 0 通过 `SharedMemory` 写入要调用的方法和参数
- worker 通过 `Event` 被唤醒，读取 shared memory 并执行同名方法
- 所有 rank 通过 `torch.distributed` / NCCL 协同执行 TP 层

一次 `run()` 执行：

1. 根据 `is_prefill` 调用 `prepare_prefill()` 或 `prepare_decode()`
2. rank 0 准备采样 temperature
3. 调用 `run_model()` 执行模型
4. rank 0 调用 `Sampler` 得到新 token
5. 重置全局 attention context

## 5. Model 和 Layers 层

### 5.1 `models/qwen3.py`

这里实现 Qwen3 Causal LM：

```text
Qwen3ForCausalLM
└── Qwen3Model
    ├── VocabParallelEmbedding
    ├── Qwen3DecoderLayer × N
    │   ├── RMSNorm
    │   ├── Qwen3Attention
    │   │   ├── QKVParallelLinear
    │   │   ├── RotaryEmbedding
    │   │   ├── Attention
    │   │   └── RowParallelLinear
    │   ├── RMSNorm
    │   └── Qwen3MLP
    │       ├── MergedColumnParallelLinear
    │       ├── SiluAndMul
    │       └── RowParallelLinear
    └── RMSNorm
```

`Qwen3ForCausalLM` 还定义了 `packed_modules_mapping`，用于把 Hugging Face 权重中的多个独立投影合并加载到框架内部的 fused / merged 参数中：

| HF 权重名片段 | 内部参数 | shard 标识 |
| --- | --- | --- |
| `q_proj` | `qkv_proj` | `q` |
| `k_proj` | `qkv_proj` | `k` |
| `v_proj` | `qkv_proj` | `v` |
| `gate_proj` | `gate_up_proj` | `0` |
| `up_proj` | `gate_up_proj` | `1` |

### 5.2 `layers/attention.py`

Attention 层负责：

- 使用 Triton kernel 将当前 K/V 写入预分配 KV cache
- Prefill 时调用 `flash_attn_varlen_func`
- Decode 时调用 `flash_attn_with_kvcache`
- Prefix cache 命中时通过 `block_table` 使用历史 KV block

Attention 的行为依赖 `utils/context.py` 中的全局 `Context`。

### 5.3 `layers/linear.py`

该文件实现 Tensor Parallel 线性层：

| 类 | 分片方式 | 用途 |
| --- | --- | --- |
| `ReplicatedLinear` | 不分片 | 普通复制权重 |
| `ColumnParallelLinear` | 输出维度分片 | QKV、MLP up 等 |
| `MergedColumnParallelLinear` | 多个输出投影合并后按列分片 | gate/up 合并 |
| `QKVParallelLinear` | Q/K/V 合并并分片 | attention qkv projection |
| `RowParallelLinear` | 输入维度分片，输出 all-reduce | attention o_proj、MLP down |

各线性层会给参数挂载 `weight_loader`，让权重加载器按 TP rank 只加载本 rank 需要的 shard。

### 5.4 其他 Layer

| 文件 | 作用 |
| --- | --- |
| `activation.py` | `SiluAndMul`，用于 SwiGLU 风格 MLP 激活 |
| `layernorm.py` | `RMSNorm`，支持 residual add + norm fused 风格调用 |
| `rotary_embedding.py` | RoPE 预计算与应用，使用 `lru_cache` 复用 |
| `embed_head.py` | 词表并行 embedding 和 LM head |
| `sampler.py` | temperature sampling，使用 Gumbel-max 风格采样 |

## 6. Utils 层

### 6.1 `utils/context.py`

`Context` 是 attention 执行时的全局上下文，保存不同阶段需要的元数据：

- Prefill：`cu_seqlens_q`、`cu_seqlens_k`、`max_seqlen_q`、`max_seqlen_k`、`slot_mapping`、`block_tables`
- Decode：`slot_mapping`、`context_lens`、`block_tables`

`ModelRunner.prepare_prefill()` 和 `prepare_decode()` 会在模型前向前调用 `set_context()`；一次模型运行结束后调用 `reset_context()`。

### 6.2 `utils/loader.py`

权重加载器遍历模型目录下的 `*.safetensors` 文件：

1. 对每个权重名检查是否命中 `packed_modules_mapping`
2. 命中时改写为内部参数名，并带上 shard id 调用参数自己的 `weight_loader`
3. 未命中时直接按原权重名加载
4. 如果参数没有自定义 `weight_loader`，使用默认的整张量 copy

## 7. 一次生成请求的完整链路

```text
用户调用 LLM.generate(prompts, sampling_params)
        │
        ▼
LLMEngine.add_request()
将 prompt 转 token，并创建 Sequence
        │
        ▼
Scheduler.add()
进入 waiting 队列
        │
        ▼
LLMEngine.step() 循环
        │
        ├── Scheduler.schedule()
        │   ├── 优先 prefill waiting 队列
        │   └── waiting 为空时 decode running 队列
        │
        ├── ModelRunner.run(seqs, is_prefill)
        │   ├── prepare_prefill / prepare_decode
        │   ├── set_context
        │   ├── Qwen3ForCausalLM.forward
        │   ├── compute_logits
        │   └── Sampler.forward
        │
        ├── Scheduler.postprocess()
        │   ├── 更新 block hash / prefix cache
        │   ├── 追加采样 token
        │   ├── 判断 EOS / max_tokens
        │   └── 释放完成序列 KV block
        │
        ▼
所有序列完成后 tokenizer.decode()
返回 [{"text": ..., "token_ids": ...}]
```

## 8. Prefill 与 Decode 数据准备差异

### Prefill

`prepare_prefill()` 会为一批序列构造：

- `input_ids`
- `positions`
- `cu_seqlens_q`
- `cu_seqlens_k`
- `slot_mapping`
- 可选 `block_tables`

如果存在 prefix cache，`cu_seqlens_k` 会大于 `cu_seqlens_q`，attention 通过 `block_tables` 同时读取旧 KV 与当前新增 token。

### Decode

`prepare_decode()` 每个序列只取最后一个 token，构造：

- `input_ids`：每条序列的 last token
- `positions`：当前位置
- `slot_mapping`：新 token 写入的 KV cache slot
- `context_lens`：每条序列当前长度
- `block_tables`：每条序列完整 KV block 映射

Decode 阶段可走 CUDA Graph：当不是 prefill、未开启 `enforce_eager` 且 batch size 不超过 512 时，`run_model()` 会选择已 capture 的图并 replay。

## 9. 示例与 Benchmark

### `example.py`

示例脚本默认读取：

```text
~/huggingface/Qwen3-0.6B/
```

然后：

1. 用 `AutoTokenizer` 加载 tokenizer
2. 创建 `LLM`
3. 将普通 prompt 包装成 chat template
4. 调用 `llm.generate()`
5. 打印 completion

### `bench.py`

Benchmark 脚本默认：

- 生成 256 条随机请求
- 输入长度在 100 到 1024 之间
- 输出长度在 100 到 1024 之间
- 先 warmup 一次
- 再统计总生成 token、耗时和吞吐

## 10. 阅读源码建议顺序

建议按下面顺序读代码：

1. `example.py`：先看用户如何调用
2. `nanovllm/__init__.py`、`nanovllm/llm.py`：确认公开 API
3. `nanovllm/engine/llm_engine.py`：理解生成主循环
4. `nanovllm/engine/scheduler.py`：理解 prefill / decode 调度
5. `nanovllm/engine/block_manager.py`：理解 KV cache 与 prefix cache
6. `nanovllm/engine/model_runner.py`：理解 CUDA 执行和 TP 多进程
7. `nanovllm/models/qwen3.py`：理解模型结构
8. `nanovllm/layers/attention.py`：理解 flash-attn 和 KV cache 使用
9. `nanovllm/layers/linear.py`、`embed_head.py`：理解 Tensor Parallel 分片
10. `nanovllm/utils/loader.py`：理解 safetensors 权重加载

## 11. 当前实现的几个关键约束

- 依赖 CUDA、NCCL、flash-attn 和 Triton，主要面向 NVIDIA GPU 推理。
- `Config.model` 必须是本地模型目录。
- `tensor_parallel_size` 限制为 1 到 8。
- `kvcache_block_size` 必须是 256 的倍数。
- 当前采样不允许 temperature 接近 0，因此不支持严格 greedy。
- 当前模型实现聚焦 Qwen3，扩展其他模型需要新增 `models/*` 并适配权重映射和层结构。
- TP worker 使用固定 TCP 地址 `tcp://localhost:2333` 初始化 NCCL，同机多进程运行时需避免端口冲突。

