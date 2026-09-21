#!/usr/bin/env python3
"""nano-vllm ablation benchmark suite (single file, no third-party deps beyond
the project's own).

Subcommands
    prepare     build T1/T2/T3 datasets (CPU only, needs the tokenizer)
    tiny-moe    synthesize a small Qwen3-MoE checkpoint + offline load check
    run         build the experiment matrix, run every config in a fresh process
    one         run ONE configuration (called by `run`, usable by hand)
    analyze     raw json results -> the three report tables (markdown)

Report structure
    table 1  per-item contribution   E1 chunked prefill / E2 speculative /
                                     E3 MoE / E4 TP   (one variable at a time)
    table 2  cumulative combination  base -> +chunked -> +spec -> both,
                                     plus interaction vs the product of parts
    table 3  regression              current vs original nano-vllm checkout
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:          # so `python bench/nanovllm_bench.py` works
    sys.path.insert(0, str(REPO))      # on machines without an editable install
DTYPE_BYTES = {
    "float32": 4, "float": 4, "float16": 2, "half": 2, "bfloat16": 2,
    "float64": 8, "double": 8, "int8": 1, "uint8": 1,
}
INSTRUCTION_COPY = (
    "Repeat the following passage verbatim, word for word, without any "
    "commentary or changes:\n\n"
)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def load_hf_config(model_dir) -> dict:
    with open(Path(model_dir) / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def dtype_bytes(hf: dict) -> int:
    return DTYPE_BYTES.get(str(hf.get("torch_dtype") or hf.get("dtype") or "bfloat16").lower(), 2)


def head_dim(hf: dict) -> int:
    return int(hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"])


def kv_bytes_per_token(hf: dict, tp: int = 1) -> int:
    kv_heads = max(1, int(hf["num_key_value_heads"]) // tp)
    return 2 * int(hf["num_hidden_layers"]) * kv_heads * head_dim(hf) * dtype_bytes(hf)


def weights_bytes(model_dir) -> int:
    total = 0
    for pat in ("*.safetensors", "*.bin"):
        for p in Path(model_dir).glob(pat):
            total += p.stat().st_size
    return total


def read_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(values: list[float], q: float):
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def peak_gib_reset():
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch


# --------------------------------------------------------------------------- #
# instrumentation (nanovllm imported lazily so prepare/analyze run CPU-only)
# --------------------------------------------------------------------------- #
_installed: set[str] = set()
_ttft: list[tuple[int, float]] = []      # (seq_id, seconds to first token)
_arrivals: list[int] = []                # seq_ids in request-arrival order
_spec = {"batches": 0, "seqs": 0, "K": 0, "a": 0}
_draft = {"calls": 0, "sec": 0.0, "proposed": 0}
_moe = {"tokens": 0, "slots": 0, "route_sec": 0.0, "hist": None,
        "calls": 0, "real_slots": 0.0, "padded_slots": 0.0, "blocks": 0}


def _install_ttft():
    if "ttft" in _installed:
        return
    from nanovllm.engine.sequence import Sequence
    import nanovllm.engine.scheduler as scheduler

    original_add = scheduler.Scheduler.add

    def scheduler_add(self, seq):
        seq._bench_arrive = time.perf_counter()
        _arrivals.append(seq.seq_id)
        return original_add(self, seq)

    scheduler.Scheduler.add = scheduler_add
    original_append = Sequence.append_token

    def append_token(self, token_id):
        if getattr(self, "_bench_first", None) is None:
            self._bench_first = time.perf_counter()
            arrive = getattr(self, "_bench_arrive", None)
            if arrive is not None:
                _ttft.append((self.seq_id, self._bench_first - arrive))
        return original_append(self, token_id)

    Sequence.append_token = append_token
    _installed.add("ttft")


def _install_spec():
    if "spec" in _installed:
        return
    import nanovllm.engine.llm_engine as engine
    from nanovllm.engine.speculative import NgramDraftHead

    original_accept = engine.speculative_accept

    def speculative_accept(drafts, logits, temperatures):
        results = original_accept(drafts, logits, temperatures)
        _spec["batches"] += 1
        _spec["seqs"] += len(drafts)
        _spec["K"] += sum(len(d) for d in drafts)
        _spec["a"] += sum(int(n) for _, n in results)
        return results

    engine.speculative_accept = speculative_accept
    original_draft = NgramDraftHead.draft

    def draft(self, token_ids):
        t0 = time.perf_counter()
        out = original_draft(self, token_ids)
        _draft["calls"] += 1
        _draft["sec"] += time.perf_counter() - t0
        _draft["proposed"] += len(out)
        return out

    NgramDraftHead.draft = draft
    _installed.add("spec")


def _install_moe(reference: bool = False, free_experts: bool = False):
    import torch
    from nanovllm.layers.moe import FusedMoE

    if "moe" not in _installed:
        original_route = FusedMoE._route

        def _route(self, x):
            t0 = time.perf_counter()
            weights, ids = original_route(self, x)
            _moe["route_sec"] += time.perf_counter() - t0
            if _moe["hist"] is None:
                _moe["hist"] = torch.zeros(self.num_experts, dtype=torch.long, device="cpu")
            flat = ids.reshape(-1).detach().cpu()
            _moe["hist"] += torch.bincount(flat, minlength=self.num_experts)
            _moe["tokens"] += int(ids.shape[0])
            _moe["slots"] += int(ids.numel())
            # padding waste: moe_align pads each expert to a multiple of _BLOCK_M(64),
            # so a decode step (few tokens per expert) wastes most of the GEMM work
            counts = torch.bincount(flat, minlength=self.num_experts).double()
            blocks = torch.ceil(counts / 64.0)
            _moe["calls"] += 1
            _moe["real_slots"] += float(counts.sum())
            _moe["blocks"] += int(blocks.sum())
            _moe["padded_slots"] += float(blocks.sum()) * 64.0
            return weights, ids

        FusedMoE._route = _route
        _installed.add("moe")

    if reference:
        FusedMoE.forward = lambda self, x: self.forward_reference(x)

    if free_experts:
        # FusedMoE._stack_weights() keeps a full stacked copy of every expert per
        # layer while the per-expert parameters stay alive -> ~2x the model size
        # (Qwen3-30B-A3B: ~114 GiB). This benchmark-only hook drops the originals.
        # gc.collect() is required: each Parameter holds its bound weight_loader,
        # which references the module back, so dropping the ModuleList alone does
        # not free anything (measured: 58.0 -> 56.9 GiB only after gc).
        import gc
        original_stack = FusedMoE._stack_weights

        def _stack(self):
            w1, w2 = original_stack(self)
            if getattr(self, "experts", None) is not None:
                self.experts = None
                gc.collect()
                torch.cuda.empty_cache()
            return w1, w2

        FusedMoE._stack_weights = _stack
        print("[moe] per-expert parameters are released after stacking "
              "(benchmark-only hook + gc.collect, halves MoE memory)")


def reset_stats():
    _ttft.clear()
    _arrivals.clear()
    for k in _spec:
        _spec[k] = 0
    for k in _draft:
        _draft[k] = 0.0 if k == "sec" else 0
    _moe["tokens"] = 0
    _moe["slots"] = 0
    _moe["route_sec"] = 0.0
    _moe["hist"] = None
    _moe["calls"] = 0
    _moe["real_slots"] = 0.0
    _moe["padded_slots"] = 0.0
    _moe["blocks"] = 0


def snapshot() -> dict:
    hist = _moe["hist"]
    moe = {"tokens": _moe["tokens"], "slots": _moe["slots"],
           "route_sec": _moe["route_sec"], "calls": _moe["calls"],
           "real_slots": _moe["real_slots"], "padded_slots": _moe["padded_slots"],
           "blocks": _moe["blocks"],
           "hist": hist.tolist() if hist is not None else None}
    if _moe["padded_slots"] > 0:
        # how much of the grouped-GEMM work is real data vs padding (1.0 = no waste)
        moe["pad_efficiency"] = _moe["real_slots"] / _moe["padded_slots"]
        moe["slots_per_expert_per_call"] = (
            _moe["real_slots"] / max(1, _moe["calls"]) / len(hist) if hist is not None else None)
    if hist is not None and int(hist.sum()) > 0:
        mean = float(hist.float().mean())
        moe["active_experts"] = int((hist > 0).sum())
        moe["imbalance_max_over_mean"] = float(hist.max()) / mean if mean > 0 else None
    return {"ttft": list(_ttft), "arrivals": list(_arrivals),
            "spec": dict(_spec), "draft": dict(_draft), "moe": moe}


# --------------------------------------------------------------------------- #
# subcommand: prepare
# --------------------------------------------------------------------------- #
def default_corpus(repo: Path) -> list[str]:
    paths = [repo / "README.md", repo / "CODE_STRUCTURE.md", repo / "BENCHMARK_PLAN.md"]
    paths += sorted((repo / "nanovllm").rglob("*.py"))
    paths += [repo / "bench" / n for n in ("nanovllm_bench.py",)]
    return [str(p) for p in paths if Path(p).exists()]


def build_corpus(paths, tokenizer, min_tokens: int) -> list[int]:
    ids: list[int] = []
    newline = tokenizer.encode("\n\n")
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        ids.extend(tokenizer.encode(text))
        ids.extend(newline)
    if len(ids) < min_tokens:
        if not ids:
            raise SystemExit("empty corpus: pass --corpus with some text files")
        base = list(ids)
        while len(ids) < min_tokens:
            ids.extend(base)
    return ids


def take(corpus: list[int], n: int, rng: random.Random) -> list[int]:
    """A natural contiguous slice, with a distinct random offset so that
    requests never share a leading block (which would fake prefix-cache hits)."""
    if n >= len(corpus):
        out: list[int] = []
        while len(out) < n:
            out.extend(corpus)
        return out[:n]
    start = rng.randrange(0, len(corpus) - n)
    return corpus[start:start + n]


def case(prompt, max_tokens, temperature, tag, arrival_ms=0) -> dict:
    return {"prompt_token_ids": list(prompt), "max_tokens": int(max_tokens),
            "temperature": float(temperature), "ignore_eos": True, "tag": tag,
            "arrival_ms": int(arrival_ms)}


def dataset_t1(rng, corpus, tok, scale):
    """Prefill-heavy. Long prompts (2048) exceed the small prefill budgets so
    chunked prefill actually triggers; short requests keep decode alive."""
    out = [case(take(corpus, 2048, rng), 128, 0.6, "t1_long_2048")
           for _ in range(max(1, round(8 * scale)))]
    out += [case(take(corpus, 128, rng), 128, 0.6, "t1_short_128")
            for _ in range(max(1, round(16 * scale)))]
    return out


def dataset_t2_predictable(rng, corpus, tok, scale):
    """Copy task: the model repeats the passage, so the n-gram draft can hit."""
    prefix = tok.encode(INSTRUCTION_COPY)
    return [case(prefix + take(corpus, 224, rng), 512, 0.6, "t2_copy")
            for _ in range(max(1, round(16 * scale)))]


def dataset_t2_open(rng, corpus, tok, scale):
    """Open-ended control: alpha should collapse here."""
    topics = ["the history of cartography", "why volcanoes erupt", "how tides work",
              "the economics of street food", "the design of subway maps",
              "the biology of sleep", "the mathematics of music", "why languages die",
              "the sociology of queues", "the chemistry of bread", "the physics of curling",
              "the history of paper money", "how vaccines are tested", "the art of translation",
              "the future of ocean farming", "the ethics of echo chambers"]
    out = []
    for i in range(max(1, round(16 * scale))):
        prompt = tok.encode(f"Write a detailed, original essay about {topics[i % len(topics)]}. ")[:96]
        out.append(case(prompt, 512, 0.6, "t2_open"))
    return out


def dataset_t3(rng, corpus, tok, scale):
    """Mixed workload (sized so a 6 GiB card still fits the whole batch)."""
    out = [case(take(corpus, 3584, rng), 256, 0.6, "t3_long_3584")
           for _ in range(max(1, round(2 * scale)))]
    out += [case(take(corpus, 1024, rng), 256, 0.6, "t3_mid_1024")
            for _ in range(max(1, round(5 * scale)))]
    out += [case(take(corpus, 128, rng), 256, 0.6, "t3_short_128")
            for _ in range(max(1, round(12 * scale)))]
    return out


def dataset_t4_short(rng, corpus, tok, scale):
    """Short-prompt set for tiny models whose max_position_embeddings is small."""
    out = [case(take(corpus, 128, rng), 128, 0.6, "t4_prompt128")
           for _ in range(max(1, round(6 * scale)))]
    out += [case(take(corpus, 64, rng), 128, 0.6, "t4_prompt64")
            for _ in range(max(1, round(10 * scale)))]
    return out


def dataset_d1_interactive(rng, corpus, tok, scale):
    """Chunked prefill's actual use case: long prefills already in flight while
    short requests keep arriving and must not queue behind them."""
    out = [case(take(corpus, 3584, rng), 128, 0.6, "d1_long_3584", 0)
           for _ in range(max(1, round(8 * scale)))]
    for i in range(max(4, round(20 * scale))):
        out.append(case(take(corpus, 128, rng), 128, 0.6, "d1_short_128", 100 * (i + 1)))
    return out


def dataset_d3_spec_small(rng, corpus, tok, scale):
    """Few sequences + long outputs: the regime where speculative decoding can pay
    (the decode step is dominated by weight traffic, so extra drafted tokens are
    nearly free). Run with --limit 1/2/4 to vary the batch size."""
    prefix = tok.encode(INSTRUCTION_COPY)
    return [case(prefix + take(corpus, 224, rng), 1024, 0.6, "d3_copy", 0)
            for _ in range(max(1, round(4 * scale)))]


def dataset_d5_moe_decode(rng, corpus, tok, scale):
    """Decode-heavy MoE: tiny prompts, long outputs. Every expert then receives
    only a handful of tokens per step, which is exactly where the BLOCK_M=64
    padding of the grouped GEMM wastes most of its work."""
    return [case(take(corpus, 64, rng), 1024, 0.6, "d5_decode", 0)
            for _ in range(max(1, round(8 * scale)))]


def dataset_d6_moe_prefill(rng, corpus, tok, scale):
    """Prefill-heavy MoE: large prompt volume drives the grouped GEMM at scale and
    pushes peak memory (this is where the Triton path ran out of memory)."""
    return [case(take(corpus, 3584, rng), 128, 0.6, "d6_prefill", 0)
            for _ in range(max(1, round(12 * scale)))]


def write_dummy():  # placeholder removed
    pass
def dataset_d2_saturating(rng, corpus, tok, scale):
    """Enough prompt volume (~70k tokens) that even a large prefill budget has to
    split work, so the throughput cost of chunking becomes measurable."""
    out = [case(take(corpus, 3584, rng), 512, 0.6, "d2_long_3584", 0)
           for _ in range(max(1, round(8 * scale)))]
    out += [case(take(corpus, 2048, rng), 512, 0.6, "d2_mid_2048", 0)
            for _ in range(max(1, round(16 * scale)))]
    out += [case(take(corpus, 512, rng), 512, 0.6, "d2_short_512", 0)
            for _ in range(max(1, round(16 * scale)))]
    return out


def cmd_prepare(args):
    from transformers import AutoTokenizer
    hf = load_hf_config(args.model)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_paths = args.corpus or default_corpus(REPO)
    corpus = build_corpus(corpus_paths, tok, min_tokens=8192)
    rng = random.Random(args.seed)

    datasets = {
        "t1_long.jsonl": dataset_t1(rng, corpus, tok, args.scale),
        "t2_predictable.jsonl": dataset_t2_predictable(rng, corpus, tok, args.scale),
        "t2_open.jsonl": dataset_t2_open(rng, corpus, tok, args.scale),
        "t3_mixed.jsonl": dataset_t3(rng, corpus, tok, args.scale),
        "t4_short.jsonl": dataset_t4_short(rng, corpus, tok, args.scale),
        "d1_interactive.jsonl": dataset_d1_interactive(rng, corpus, tok, args.scale),
        "d2_saturating.jsonl": dataset_d2_saturating(rng, corpus, tok, args.scale),
        "d3_spec_small.jsonl": dataset_d3_spec_small(rng, corpus, tok, args.scale),
        "d5_moe_decode.jsonl": dataset_d5_moe_decode(rng, corpus, tok, args.scale),
        "d6_moe_prefill.jsonl": dataset_d6_moe_prefill(rng, corpus, tok, args.scale),
    }
    for name, cases in datasets.items():
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c) + "\n")

    per_tok = kv_bytes_per_token(hf)
    model_b = weights_bytes(args.model)
    capacity = max(0, int((args.gpu_gib * (1 << 30) * args.gpu_util - model_b * 1.05 - (1 << 30)) / per_tok))

    too_long = [c for cases in datasets.values() for c in cases
                if len(c["prompt_token_ids"]) + c["max_tokens"] > args.max_model_len]
    if too_long:
        worst = max(len(c["prompt_token_ids"]) + c["max_tokens"] for c in too_long)
        print(f"WARNING: {len(too_long)} requests exceed --max-model-len "
              f"{args.max_model_len} (worst {worst} tokens). Decode with CUDA graphs then "
              f"crashes in model_runner.py:228 (fixed-width block_tables). Run the suite "
              f"with a larger --max-model-len, or rebuild the datasets.")
        print()

    print(f"corpus: {len(corpus)} tokens from {len(corpus_paths)} files")
    print(f"KV/token {per_tok / 1024:.1f} KiB | weights {model_b / 2**30:.2f} GiB "
          f"| capacity est. on {args.gpu_gib:.0f} GiB: {capacity} tokens")
    print(f"\n{'dataset':<24}{'reqs':>6}{'tok in':>10}{'demand':>10}{'max req':>9}   status")
    for name, cases in datasets.items():
        tok_in = sum(len(c["prompt_token_ids"]) for c in cases)
        demand = tok_in + sum(c["max_tokens"] for c in cases)
        longest = max(len(c["prompt_token_ids"]) + c["max_tokens"] for c in cases)
        staggered = sum(1 for c in cases if c.get("arrival_ms", 0) > 0)
        if demand < 0.85 * capacity and longest <= args.max_model_len:
            status = f"ok{'  (staggered: %d)' % staggered if staggered else ''}"
        elif longest > args.max_model_len:
            status = f"TOO LONG for --max-model-len {args.max_model_len}"
        else:
            status = "TIGHT -> lower --scale"
        print(f"{name:<24}{len(cases):>6}{tok_in:>10}{demand:>10}{longest:>9}   {status}")
    meta = {"model": args.model, "corpus_tokens": len(corpus),
            "kv_bytes_per_token": per_tok, "weights_bytes": model_b,
            "capacity_tokens_estimate": capacity,
            "datasets": {k: len(v) for k, v in datasets.items()}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwritten to {out_dir.resolve()}")


# --------------------------------------------------------------------------- #
# subcommand: tiny-moe
# --------------------------------------------------------------------------- #
MAPPING = {"q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"),
           "v_proj": ("qkv_proj", "v"), "gate_proj": ("gate_up_proj", 0),
           "up_proj": ("gate_up_proj", 1)}


def map_name(hf_name: str) -> str:
    for k, (v, _) in MAPPING.items():
        if k in hf_name:
            return hf_name.replace(k, v)
    return hf_name


def expected_params(hf: dict) -> set[str]:
    """Mirror of the layer rules used by the nano models themselves:
    dense layers -> mlp.gate_up_proj/down_proj, sparse layers -> mlp.gate +
    per-expert weights (qwen3_moe.py:28-43), and q_norm/k_norm only when the
    attention has no bias (qwen3.py:68-70)."""
    names = {"model.embed_tokens.weight", "model.norm.weight"}
    if not hf.get("tie_word_embeddings", False):
        names.add("lm_head.weight")

    default_bias = hf.get("model_type") == "qwen3"   # qwen3.py:133 defaults to True
    attention_bias = bool(hf.get("attention_bias", default_bias))
    sparse_step = int(hf.get("decoder_sparse_step", 1) or 1)
    mlp_only = set(hf.get("mlp_only_layers") or [])
    num_experts = int(hf.get("num_experts", 0) or 0)

    for i in range(int(hf["num_hidden_layers"])):
        p = f"model.layers.{i}"
        names |= {f"{p}.self_attn.qkv_proj.weight", f"{p}.self_attn.o_proj.weight",
                  f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight"}
        if not attention_bias:
            names |= {f"{p}.self_attn.q_norm.weight", f"{p}.self_attn.k_norm.weight"}
        is_sparse = (i not in mlp_only) and num_experts > 0 and (i + 1) % sparse_step == 0
        if is_sparse:
            names.add(f"{p}.mlp.gate.weight")
            for e in range(num_experts):
                names |= {f"{p}.mlp.experts.{e}.gate_up_proj.weight",
                          f"{p}.mlp.experts.{e}.down_proj.weight"}
        else:
            names |= {f"{p}.mlp.gate_up_proj.weight", f"{p}.mlp.down_proj.weight"}
    return names


def check_names(model_dir, hf) -> bool:
    """Offline check: every internal parameter must be produced by some HF tensor."""
    from safetensors import safe_open
    produced: set[str] = set()
    files = sorted(Path(model_dir).glob("*.safetensors"))
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as fh:
            for name in fh.keys():
                produced.add(map_name(name))
    expected = expected_params(hf)
    missing = sorted(expected - produced)
    extra = sorted(produced - expected)
    print(f"  [names] {len(produced)} tensors map onto {len(expected)} internal params")
    if missing:
        print(f"  [names] MISSING ({len(missing)}): {missing[:6]}")
    if extra:
        print(f"  [names] EXTRA   ({len(extra)}): {extra[:6]}")
    return not missing


def expected_shapes(hf: dict) -> dict:
    """Config-derived shape of every internal parameter (packed params included)."""
    hidden = int(hf["hidden_size"])
    heads = int(hf["num_attention_heads"])
    kv_heads = int(hf.get("num_key_value_heads", heads))
    hd = int(hf.get("head_dim") or hidden // heads)
    inter = int(hf["intermediate_size"])
    vocab = int(hf["vocab_size"])
    layers = int(hf["num_hidden_layers"])
    default_bias = hf.get("model_type") == "qwen3"
    attention_bias = bool(hf.get("attention_bias", default_bias))
    sparse_step = int(hf.get("decoder_sparse_step", 1) or 1)
    mlp_only = set(hf.get("mlp_only_layers") or [])
    num_experts = int(hf.get("num_experts", 0) or 0)
    moe_inter = int(hf.get("moe_intermediate_size", inter))

    shapes = {"model.embed_tokens.weight": (vocab, hidden), "model.norm.weight": (hidden,)}
    if not hf.get("tie_word_embeddings", False):
        shapes["lm_head.weight"] = (vocab, hidden)
    for i in range(layers):
        p = f"model.layers.{i}"
        shapes[f"{p}.self_attn.qkv_proj.weight"] = ((heads + 2 * kv_heads) * hd, hidden)
        shapes[f"{p}.self_attn.o_proj.weight"] = (hidden, heads * hd)
        shapes[f"{p}.input_layernorm.weight"] = (hidden,)
        shapes[f"{p}.post_attention_layernorm.weight"] = (hidden,)
        if not attention_bias:
            shapes[f"{p}.self_attn.q_norm.weight"] = (hd,)
            shapes[f"{p}.self_attn.k_norm.weight"] = (hd,)
        is_sparse = (i not in mlp_only) and num_experts > 0 and (i + 1) % sparse_step == 0
        if is_sparse:
            shapes[f"{p}.mlp.gate.weight"] = (num_experts, hidden)
            for e in range(num_experts):
                shapes[f"{p}.mlp.experts.{e}.gate_up_proj.weight"] = (2 * moe_inter, hidden)
                shapes[f"{p}.mlp.experts.{e}.down_proj.weight"] = (hidden, moe_inter)
        else:
            shapes[f"{p}.mlp.gate_up_proj.weight"] = (2 * inter, hidden)
            shapes[f"{p}.mlp.down_proj.weight"] = (hidden, inter)
    return shapes


def check_shapes(model_dir, hf) -> bool:
    """Header-only shape check: no weights are materialised, so it runs in <2 GiB RAM."""
    from safetensors import safe_open
    rows: dict[str, int] = {}
    cols: dict[str, set] = {}
    found: set[str] = set()
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as fh:
            for name in fh.keys():
                shape = tuple(fh.get_slice(name).get_shape())
                mapped = map_name(name)
                found.add(mapped)
                rows[mapped] = rows.get(mapped, 0) + shape[0]
                cols.setdefault(mapped, set()).add(tuple(shape[1:]))
    expected = expected_shapes(hf)
    missing = sorted(set(expected) - found)
    bad = []
    for name, want in expected.items():
        if name not in found:
            continue
        got_rows = rows[name]
        got_col = next(iter(cols[name])) if cols[name] else ()
        if got_rows != want[0] or (len(want) > 1 and got_col and got_col[0] != want[1]):
            bad.append(f"{name} got({got_rows},{got_col[0] if got_col else '-'}) want{want}")
    print(f"  [shapes] {len(found)} params, {sum(rows.values()):,} rows total")
    if missing:
        print(f"  [shapes] MISSING ({len(missing)}): {missing[:5]}")
    if bad:
        print(f"  [shapes] MISMATCH ({len(bad)}): {bad[:5]}")
    return not missing and not bad


def verify_load(model_dir) -> bool:
    """Offline check on CPU: instantiate the nano model and load the checkpoint
    through the real loader (no GPU needed; gloo world_size=1)."""
    import torch
    import torch.distributed as dist
    from transformers import AutoConfig
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29591",
                                world_size=1, rank=0)
    torch.set_default_device("cpu")
    from nanovllm.utils.loader import load_model

    hf = AutoConfig.from_pretrained(model_dir)
    if getattr(hf, "model_type", None) == "qwen3_moe":
        from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM as NanoModel
    else:
        from nanovllm.models.qwen3 import Qwen3ForCausalLM as NanoModel

    nano = NanoModel(hf)
    load_model(nano, model_dir)

    # value check on the first sparse layer (works for dense/MoE mixed stacks)
    layer_idx = next((i for i, l in enumerate(nano.model.layers)
                      if hasattr(l.mlp, "experts")), None)
    if layer_idx is None:
        print("  [load] dense-only checkpoint: every parameter loaded")
        return True

    from safetensors import safe_open
    key_g = f"model.layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
    key_u = f"model.layers.{layer_idx}.mlp.experts.0.up_proj.weight"
    found = None
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as fh:
            if key_g in fh.keys():
                found = (fh.get_tensor(key_g), fh.get_tensor(key_u))
                break
    if found is None:
        print("  [load] skipped value check (no expert tensors)")
        return True
    gate, up = found
    mi = gate.shape[0]
    local = nano.model.layers[layer_idx].mlp.experts[0].gate_up_proj.weight.data
    ok = (local.shape[0] == 2 * mi
          and torch.allclose(local[:mi].float(), gate.float())
          and torch.allclose(local[mi:2 * mi].float(), up.float()))
    print(f"  [load] expert gate_up merged correctly (layer {layer_idx}): {ok}")
    return ok


def cmd_tiny_moe(args):
    from transformers import AutoConfig, AutoTokenizer
    try:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("transformers too old for Qwen3-MoE "
                         f"({exc}); upgrade transformers") from exc

    base_hf = load_hf_config(args.base)
    vocab_size = int(base_hf["vocab_size"])
    cfg_kwargs = dict(
        vocab_size=vocab_size, hidden_size=args.hidden_size,
        intermediate_size=args.intermediate, num_hidden_layers=args.layers,
        num_attention_heads=args.heads, num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim, max_position_embeddings=args.max_position,
        rms_norm_eps=1e-6, rope_theta=10000.0, attention_bias=False,
        hidden_act="silu", tie_word_embeddings=False,
        num_experts=args.experts, num_experts_per_tok=args.top_k,
        moe_intermediate_size=args.moe_intermediate,
        shared_expert_intermediate_size=0, decoder_sparse_step=1,
        mlp_only_layers=[], norm_topk_prob=True,
    )
    config = Qwen3MoeConfig(**cfg_kwargs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    config.torch_dtype = args.dtype
    torch.manual_seed(args.seed)
    model = Qwen3MoeForCausalLM(config).to(dtype)
    model.save_pretrained(out, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    tok.save_pretrained(out)
    print(f"tiny MoE written to {out.resolve()} "
          f"({sum(p.stat().st_size for p in out.glob('*.safetensors')) / 2**20:.0f} MiB)")

    hf = load_hf_config(out)
    print(f"  model_type={hf.get('model_type')} experts={hf.get('num_experts')} "
          f"top_k={hf.get('num_experts_per_tok')} layers={hf.get('num_hidden_layers')}")
    ok = check_names(out, hf)
    if args.verify_load:
        try:
            ok = verify_load(out) and ok
        except Exception as exc:  # pragma: no cover
            print(f"  [load] verification failed: {type(exc).__name__}: {exc}")
            ok = False
    if not ok and args.strict:
        raise SystemExit("tiny-moe self check failed")
    print("  self check:", "PASS" if ok else "FAIL (see above)")


# --------------------------------------------------------------------------- #
# subcommand: one
# --------------------------------------------------------------------------- #
def _drive_staggered(llm, cases, sps, timeout_s=900.0) -> int:
    """Submit requests on their own arrival schedule and step the engine by hand.

    llm.generate() adds every request at once, which hides the whole point of
    chunked prefill: a short request must be able to overtake a long prefill.
    Returns the number of completion tokens produced.
    """
    order = sorted(range(len(cases)), key=lambda i: cases[i].get("arrival_ms", 0))
    t0 = time.perf_counter()
    pending = list(order)
    produced = 0
    while True:
        now = time.perf_counter() - t0
        while pending and cases[pending[0]].get("arrival_ms", 0) / 1000.0 <= now:
            i = pending.pop(0)
            llm.add_request(cases[i]["prompt_token_ids"], sps[i])
        if pending and not llm.scheduler.running and not llm.scheduler.waiting:
            time.sleep(0.001)          # nothing to do until the next arrival
            continue
        outputs, _, _ = llm.step()
        produced += sum(len(tids) for _, tids in outputs)
        if not pending and llm.is_finished():
            return produced
        if time.perf_counter() - t0 > timeout_s:
            raise SystemExit(f"staggered run exceeded {timeout_s:.0f}s")


def _install_fair_chunk():
    """Harness-side experiment: give every waiting request the same prefill quantum
    (budget / len(waiting)) instead of letting the head of the queue eat the whole
    budget. Without this, a small budget starves the short requests that chunked
    prefill is supposed to protect."""
    from collections import deque
    import nanovllm.engine.scheduler as sch
    from nanovllm.engine.sequence import SequenceStatus

    def fair_schedule(self):
        prefill_seqs, decode_seqs = [], []
        num_batched_tokens = 0
        while self.running and len(decode_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
        self.running.extendleft(reversed(decode_seqs))

        n_wait = len(self.waiting)
        quantum = max(1, self.max_num_batched_tokens // n_wait) if n_wait else 0
        for _ in range(n_wait):
            if num_batched_tokens >= self.max_num_batched_tokens:
                break
            if len(prefill_seqs) >= self.max_num_seqs:
                break
            seq = self.waiting.popleft()
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    self.waiting.appendleft(seq)
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            remaining = self.max_num_batched_tokens - num_batched_tokens
            chunk = min(num_tokens, remaining, quantum)      # <-- the only change
            if chunk <= 0:
                self.waiting.append(seq)
                continue
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = chunk
            num_batched_tokens += chunk
            prefill_seqs.append(seq)
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                self.waiting.append(seq)
        return prefill_seqs, decode_seqs

    sch.Scheduler.schedule = fair_schedule
    print("[sched] fair prefill quantum enabled (chunk <= budget / len(waiting))")


def cmd_one(args):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible: run this on the GPU machine")
    if args.expect_root:
        import nanovllm
        actual = Path(nanovllm.__file__).resolve()
        expected = Path(args.expect_root).resolve()
        if expected not in actual.parents:
            raise SystemExit(
                f"imported nanovllm from {actual}\nbut expected it under {expected}\n"
                f"A 'pip install -e .' editable install takes precedence over sys.path and "
                f"would silently measure the wrong tree. bench/orig_baseline.sh sets up the "
                f"isolation (sitecustomize) for this.")

    dataset = Path(args.dataset)
    if not dataset.exists() and args.dataset_dir:
        dataset = Path(args.dataset_dir) / args.dataset
    cases = read_dataset(dataset)
    if args.limit:
        cases = cases[:args.limit]

    hf_cfg = load_hf_config(args.model)
    max_pos = int(hf_cfg.get("max_position_embeddings", 0) or 0)
    longest = max(len(c["prompt_token_ids"]) + c.get("max_tokens", 128) for c in cases)
    if max_pos and longest > max_pos:
        raise SystemExit(
            f"{Path(args.model).name} supports at most {max_pos} tokens but this dataset needs "
            f"{longest}: RoPE would index past cos_sin_cache (rotary_embedding.py:44). "
            f"Use t4_short.jsonl for tiny models, or a model with a larger context.")

    _install_ttft()
    if args.spec:
        _install_spec()
    if args.moe_stats or args.moe_reference or args.moe_free_experts:
        _install_moe(reference=args.moe_reference, free_experts=args.moe_free_experts)

    if args.fair_chunk:
        _install_fair_chunk()

    from nanovllm import LLM, SamplingParams

    prompts = [c["prompt_token_ids"] for c in cases]
    sps = []
    for c in cases:
        temp = args.temperature if args.temperature is not None else c.get("temperature", 0.6)
        sps.append(SamplingParams(temperature=temp, max_tokens=c.get("max_tokens", 128),
                                  ignore_eos=c.get("ignore_eos", True)))

    t0 = time.perf_counter()
    llm = LLM(args.model, enforce_eager=args.no_cuda_graph,
              tensor_parallel_size=args.tp, max_num_batched_tokens=args.budget,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util,
              kvcache_block_size=args.block_size,
              speculative=args.spec, max_speculative_tokens=args.spec_k)
    startup_s = time.perf_counter() - t0

    cfg = llm.model_runner.config
    capacity_tokens = int(cfg.num_kvcache_blocks) * int(cfg.kvcache_block_size)
    demand_tokens = sum(len(c["prompt_token_ids"]) + c.get("max_tokens", 128) for c in cases)

    llm.generate([[1, 2, 3, 4, 5]], [SamplingParams(max_tokens=1)], use_tqdm=False)
    reset_stats()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    runs = []
    staggered = any(c.get("arrival_ms", 0) for c in cases)
    if staggered:
        print(f"[{args.tag}] staggered arrival: "
              f"{sum(1 for c in cases if c.get('arrival_ms', 0) > 0)} of {len(cases)} "
              f"requests arrive after t=0")
    for _ in range(max(1, args.runs)):
        reset_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        if staggered:
            produced = _drive_staggered(llm, cases, sps)
        else:
            outs = llm.generate(prompts, sps, use_tqdm=False)
            produced = sum(len(o["token_ids"]) for o in outs)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        snap = snapshot()
        snap["wall_s"] = wall
        snap["produced"] = produced
        runs.append(snap)

    walls = [r["wall_s"] for r in runs]
    produced = [r["produced"] for r in runs]
    median_wall = statistics.median(walls)
    median_produced = statistics.median(produced)
    ttft_all = [v for r in runs for _, v in r["ttft"]]

    # TTFT per request class: chunked prefill mainly helps SHORT requests that
    # would otherwise queue behind a long prefill, so a pooled percentile hides it.
    by_class: dict[str, list[float]] = {}
    for r in runs:
        pos = {sid: i for i, sid in enumerate(r["arrivals"])}
        for sid, delta in r["ttft"]:
            i = pos.get(sid)
            if i is None or i >= len(cases):
                continue
            parts = str(cases[i].get("tag", "")).split("_")
            cls = parts[1] if len(parts) > 1 else "all"
            by_class.setdefault(cls, []).append(delta)
    ttft_by_class = {cls: {"p50": pct(v, 0.5), "p99": pct(v, 0.99), "n": len(v)}
                     for cls, v in sorted(by_class.items())}

    spec = {k: sum(r["spec"][k] for r in runs) for k in ("batches", "seqs", "K", "a")}
    draft = {k: sum(r["draft"][k] for r in runs) for k in ("calls", "sec", "proposed")}
    route_sec = sum(r["moe"]["route_sec"] for r in runs)
    moe_calls = sum(r["moe"]["calls"] for r in runs)
    moe_real = sum(r["moe"]["real_slots"] for r in runs)
    moe_padded = sum(r["moe"]["padded_slots"] for r in runs)
    moe_blocks = sum(r["moe"]["blocks"] for r in runs)
    hist = None
    for r in runs:
        if r["moe"]["hist"]:
            hist = [a + b for a, b in zip(hist or [0] * len(r["moe"]["hist"]), r["moe"]["hist"])]

    result = {
        "tag": args.tag, "engine_tag": args.engine_tag, "dataset": str(dataset),
        "n_seqs": len(cases), "runs": len(runs),
        "model": args.model, "tp": args.tp, "budget": args.budget,
        "gpu_util": args.gpu_util, "max_model_len": args.max_model_len,
        "fair_chunk": bool(args.fair_chunk), "moe_free_experts": bool(args.moe_free_experts),
        "no_cuda_graph": bool(args.no_cuda_graph), "spec": bool(args.spec),
        "spec_k": args.spec_k, "temperature": args.temperature,
        "moe_reference": bool(args.moe_reference),
        "startup_s": startup_s,
        "median_wall_s": median_wall,
        "wall_spread_s": max(walls) - min(walls),
        "throughput_tok_s": median_produced / median_wall if median_wall > 0 else None,
        "produced_tokens": median_produced,
        "ttft_p50": pct(ttft_all, 0.5), "ttft_p99": pct(ttft_all, 0.99),
        "ttft_n": len(ttft_all), "ttft_by_class": ttft_by_class,
        "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "capacity_tokens": capacity_tokens, "demand_tokens": demand_tokens,
        "spec_stats": spec, "draft_stats": draft,
        "moe_route_sec": route_sec, "moe_hist": hist,
        "moe_slots": int(sum(r["moe"]["slots"] for r in runs)),
        "run_walls": [round(w, 4) for w in walls],
        "moe_calls": moe_calls, "moe_blocks": moe_blocks,
        "moe_pad_efficiency": (moe_real / moe_padded) if moe_padded > 0 else None,
        "moe_slots_per_expert_per_call": (moe_real / moe_calls / len(hist)) if (moe_calls and hist) else None,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    if spec["K"] > 0:
        result["alpha"] = spec["a"] / spec["K"]
        result["tokens_per_verify"] = (median_produced * len(runs)) / spec["batches"] if spec["batches"] else None
        # tokens submitted per sequence per verify step = 1 bonus + mean accepted drafts.
        # The bonus token is per SEQUENCE, not per batch, so divide by seqs (not batches).
        result["tokens_per_verify_seq"] = (1.0 + spec["a"] / spec["seqs"]) if spec["seqs"] else None
        result["spec_ideal_tokens_per_step"] = 1.0 + result["alpha"] * args.spec_k
    if args.spec and median_wall > 0:
        result["draft_sec_over_wall"] = draft["sec"] / (median_wall * len(runs))

    out_dir = Path(args.out) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    warn = "" if demand_tokens < 0.9 * capacity_tokens else "  [WARN demand near KV capacity -> preemption possible]"
    classes = ", ".join(f"{c}={v['p50']:.3f}s" for c, v in ttft_by_class.items())
    print(f"[{args.tag}] wall={median_wall:.2f}s (+-{result['wall_spread_s']:.2f}) "
          f"throughput={result['throughput_tok_s']:.1f} tok/s "
          f"ttft_p50={result['ttft_p50']:.3f}s peak={result['peak_gib']:.2f}GiB "
          f"alpha={result.get('alpha', 0):.3f} startup={startup_s:.1f}s{warn}")
    if classes:
        print(f"           ttft by class: {classes}")


# --------------------------------------------------------------------------- #
# subcommand: run
# --------------------------------------------------------------------------- #
def config(tag, dataset, kind, model, tp=1, budget=16384, no_cuda_graph=False,
           spec=False, spec_k=8, temperature=None, moe_stats=False,
           moe_reference=False):
    return dict(tag=tag, dataset=dataset, kind=kind, model=model, tp=tp, budget=budget,
                no_cuda_graph=no_cuda_graph, spec=spec, spec_k=spec_k,
                temperature=temperature, moe_stats=moe_stats, moe_reference=moe_reference)


def build_matrix(model, moe_model, n_gpu):
    dense = model
    # E3 must use a dataset the MoE model's context window can actually hold,
    # otherwise RoPE indexes past cos_sin_cache (rotary_embedding.py:44).
    moe_max_pos = 0
    if moe_model:
        try:
            moe_max_pos = int(load_hf_config(moe_model).get("max_position_embeddings", 0) or 0)
        except Exception:
            moe_max_pos = 0
    e3_ds = "t3_mixed.jsonl" if moe_max_pos >= 4096 else "t4_short.jsonl"
    M = [
        config("r1_baseline", "t3_mixed.jsonl", "r1", dense),
        # E1 chunked prefill (T1 = prefill-heavy)
        *[config(f"e1_budget{b}", "t1_long.jsonl", "e1", dense, budget=b)
          for b in (512, 1024, 2048, 8192, 16384)],
        # E2 speculative decoding
        config("e2_spec_off_t06", "t2_predictable.jsonl", "e2", dense, temperature=0.6),
        *[config(f"e2_spec_k{k}_t06", "t2_predictable.jsonl", "e2", dense,
                 spec=True, spec_k=k, temperature=0.6) for k in (4, 8)],
        config("e2_spec_k8_t10", "t2_predictable.jsonl", "e2", dense,
               spec=True, spec_k=8, temperature=1.0),
        config("e2_spec_off_open", "t2_open.jsonl", "e2", dense, temperature=0.6),
        config("e2_spec_k8_open", "t2_open.jsonl", "e2", dense,
               spec=True, spec_k=8, temperature=0.6),
        # E3 MoE (cross-model: both sides eager)
        # combination on T3
        config("c1_chunk_t3", "t3_mixed.jsonl", "combo", dense, budget=1024),
        config("c2_spec_t3", "t3_mixed.jsonl", "combo", dense, spec=True, spec_k=4),
        config("c3_both_t3", "t3_mixed.jsonl", "combo", dense, budget=1024, spec=True, spec_k=4),
    ]
    if moe_model:
        M += [
            # dense reference for E3: same dataset, both sides eager
            config("e3_dense_eager_main", e3_ds, "e3", dense, no_cuda_graph=True),
            config("e3_moe_main", e3_ds, "e3", moe_model,
                   no_cuda_graph=True, moe_stats=True),
            config("e3_moe_ref", e3_ds, "e3", moe_model,
                   no_cuda_graph=True, moe_stats=True, moe_reference=True),
        ]
        if moe_max_pos >= 2176:   # prefill-heavy comparison needs a long context
            M += [config("e3_moe_prefill", "t1_long.jsonl", "e3", moe_model,
                         no_cuda_graph=True, moe_stats=True)]
    M += [config(f"e4_tp{tp}_t3", "t3_mixed.jsonl", "e4", dense, tp=tp)
          for tp in (2, 4) if n_gpu >= tp]
    return M


def preflight(model, dataset_dir, gpu_gib, gpu_util):
    hf = load_hf_config(model)
    max_pos = int(hf.get("max_position_embeddings", 0) or 0)
    per_tok = kv_bytes_per_token(hf)
    capacity = max(0, int((gpu_gib * (1 << 30) * gpu_util - weights_bytes(model) * 1.05 - (1 << 30)) / per_tok))
    print(f"KV/token {per_tok / 1024:.1f} KiB | capacity estimate {capacity} tokens "
          f"(tp=1, {gpu_gib:.0f} GiB @ {gpu_util}) | model context {max_pos} tokens")
    print(f"{'dataset':<22}{'reqs':>6}{'tok in':>10}{'demand':>10}{'max req':>9}   status")
    for name in ("t1_long.jsonl", "t2_predictable.jsonl", "t2_open.jsonl",
                 "t3_mixed.jsonl", "t4_short.jsonl", "d1_interactive.jsonl",
                 "d3_spec_small.jsonl", "d5_moe_decode.jsonl", "d6_moe_prefill.jsonl"):
        p = Path(dataset_dir) / name
        if not p.exists():
            print(f"{name:<22}   (missing)")
            continue
        cases = read_dataset(p)
        tok_in = sum(len(c["prompt_token_ids"]) for c in cases)
        demand = tok_in + sum(c["max_tokens"] for c in cases)
        longest = max(len(c["prompt_token_ids"]) + c.get("max_tokens", 128) for c in cases)
        if max_pos and longest > max_pos:
            status = f"TOO LONG for this model ({longest} > {max_pos})"
        elif demand < 0.85 * capacity:
            status = "ok"
        else:
            status = "TIGHT (lower --scale in prepare)"
        print(f"{name:<22}{len(cases):>6}{tok_in:>10}{demand:>10}{longest:>9}   {status}")
    print()


def cmd_run(args):
    import torch
    if not torch.cuda.is_available() and not args.dry_run:
        raise SystemExit("no CUDA device visible: run this on the GPU machine")
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        gpu_gib = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        print(f"GPUs: {n_gpu} x {torch.cuda.get_device_name(0)} ({gpu_gib:.1f} GiB each)")
    else:
        n_gpu = args.gpu_count or 1
        gpu_gib = args.gpu_gib
        print(f"[dry-run on a GPU-less machine] assuming {n_gpu} x {gpu_gib:.0f} GiB "
              f"(override with --gpu-count / --gpu-gib)")

    preflight(args.model, args.dataset_dir, gpu_gib, args.gpu_util)

    moe_model = args.moe_model or None
    if moe_model is None:
        print("note: --moe-model not given -> E3 (MoE) is skipped\n")
    elif not Path(moe_model).exists():
        raise SystemExit(f"--moe-model not found: {moe_model}")

    todo = build_matrix(args.model, moe_model, n_gpu)
    wanted = {x.strip() for x in args.experiments.split(",") if x.strip()}
    todo = [c for c in todo if c["kind"] in wanted]

    if args.dry_run:
        print(f"{'tag':<22}{'kind':<7}{'dataset':<22}{'tp':>3}{'budget':>8}{'graph':>7}"
              f"{'spec':>6}{'K':>3}{'temp':>6}  model")
        for c in todo:
            print(f"{c['tag']:<22}{c['kind']:<7}{c['dataset']:<22}{c['tp']:>3}{c['budget']:>8}"
                  f"{str(not c['no_cuda_graph']):>7}{str(c['spec']):>6}{c['spec_k']:>3}"
                  f"{str(c['temperature']):>6}  {Path(c['model']).name}")
        print(f"\n{len(todo)} configurations x {args.runs} runs (+1 warmup each)")
        return

    raw = Path(args.out) / "raw"
    logs = Path(args.out) / "logs"
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, c in enumerate(todo, 1):
        result_file = raw / f"{c['tag']}.json"
        if result_file.exists() and not args.force:
            print(f"[{i}/{len(todo)}] {c['tag']}: cached, skip")
            continue
        if not Path(c["model"]).exists():
            print(f"[{i}/{len(todo)}] {c['tag']}: model missing, skip")
            failed.append(c["tag"])
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "one",
               "--tag", c["tag"], "--engine-tag", args.engine_tag,
               "--model", c["model"], "--dataset", c["dataset"],
               "--dataset-dir", str(args.dataset_dir), "--out", str(args.out),
               "--runs", str(args.runs), "--tp", str(c["tp"]),
               "--budget", str(c["budget"]), "--block-size", str(args.block_size),
               "--max-model-len", str(args.max_model_len), "--gpu-util", str(args.gpu_util)]
        if c["no_cuda_graph"]:
            cmd.append("--no-cuda-graph")
        if c["spec"]:
            cmd += ["--spec", "--spec-k", str(c["spec_k"])]
        if c["temperature"] is not None:
            cmd += ["--temperature", str(c["temperature"])]
        if c["moe_stats"]:
            cmd.append("--moe-stats")
        if c["moe_reference"]:
            cmd.append("--moe-reference")
        if args.moe_free_experts:
            cmd.append("--moe-free-experts")
        if args.fair_chunk:
            cmd.append("--fair-chunk")
        if args.limit:
            cmd += ["--limit", str(args.limit)]

        print(f"[{i}/{len(todo)}] {c['tag']}: running "
              f"(tp={c['tp']} budget={c['budget']} graph={not c['no_cuda_graph']} "
              f"spec={c['spec']}) -> {logs / (c['tag'] + '.log')}")
        with open(logs / f"{c['tag']}.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not result_file.exists():
            print(f"    FAILED (rc={rc}) - see the log above")
            failed.append(c["tag"])
        else:
            res = json.loads(result_file.read_text(encoding="utf-8"))
            print(f"    ok: {res['throughput_tok_s']:.1f} tok/s, "
                  f"median wall {res['median_wall_s']:.2f}s")

    print(f"\nfinished {len(todo) - len(failed)}/{len(todo)} configurations")
    if failed:
        print("failed/skipped: " + ", ".join(failed))


# --------------------------------------------------------------------------- #
# subcommand: analyze
# --------------------------------------------------------------------------- #
def fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def rel(var, ref):
    if var is None or ref is None or var <= 0:
        return "-"
    return f"{ref / var:.3f}x"


def cmd_analyze(args):
    raw = Path(args.raw)
    results = {}
    for p in sorted(raw.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        results[p.stem] = data

    def get(tag):
        return results.get(tag)

    def ref_of(tag):
        return results.get(tag + ".orig")

    lines: list[str] = []
    out = lines.append
    out("# nano-vllm ablation results\n")
    out(f"- configurations found: {len(results)}")
    gpus = sorted({r.get("gpu", "?") for r in results.values()})
    out(f"- GPU: {', '.join(gpus)}")
    out(f"- note: every row is the median of "
        f"{next(iter(results.values())).get('runs', '?') if results else '?'} runs\n")
    out("- **提升 = 对照项耗时 ÷ 本行耗时,>1 就是比对照快**;每行的对照项见 condition")
    out("- `vs 原版` = 与改动前 HEAD 同配置的对比(仅原版能跑的配置有值,"
        "先跑 `bash bench/orig_baseline.sh`)\n")

    # ---------------- table 1: per-item ---------------- #
    out("## Table 1 - per-item contribution (one variable at a time)\n")
    out("| item | config | dataset | tok/s | 提升 | vs 原版 | TTFT p50 | TTFT p99 | α | 每步token | 对照项 |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")

    def row(label, tag, ref_tag, condition=""):
        r, b = get(tag), get(ref_tag)
        if r is None:
            out(f"| {label} | `{tag}` | - | - | - | - | - | - | - | - | not run |")
            return
        orig = results.get(tag + ".orig")
        out(f"| {label} | `{tag}` | {Path(r['dataset']).name} | "
            f"{fmt(r['throughput_tok_s'], 1)} | "
            f"**{rel(r['median_wall_s'], (b or {}).get('median_wall_s'))}** | "
            f"{rel(r['median_wall_s'], (orig or {}).get('median_wall_s'))} | "
            f"{fmt(r['ttft_p50'], 3)} | {fmt(r['ttft_p99'], 3)} | "
            f"{fmt(r.get('alpha'), 3)} | {fmt(r.get('tokens_per_verify_seq'), 2)} | {condition} |")

    out("| **E1 chunked prefill** (对照 = budget 16384,即不切分) |")
    for b in (512, 1024, 2048, 8192, 16384):
        row(f"budget {b}", f"e1_budget{b}", "e1_budget16384",
            "对照项:budget 16384")
    out("| **E1b chunked prefill — 错峰到达(d1_interactive,对照 = budget 16384)** |")
    for b in (16384, 4096, 1024):
        row(f"budget {b}", f"e1i_budget{b}", "e1i_budget16384",
            "长请求先到,短请求每 100ms 到达 → 看短请求 TTFT")
    out("| **E1d 错峰到达 + 公平配额实验补丁(对照 = 同 budget 现状)** |")
    for b in (16384, 4096, 1024):
        row(f"budget {b} + fair", f"e1if_budget{b}", f"e1i_budget{b}",
            "对照项:同 budget 无配额(看短请求 p99)")
    out("| **E1c chunked prefill — 饱和 prefill(d2_saturating,~70k token)** |")
    for b in (16384, 4096, 1024):
        row(f"budget {b}", f"e1s_budget{b}", "e1s_budget16384", "对照项:budget 16384")
    out("| **E2 speculative** (对照 = 同一数据集不开投机) |")
    for tag, ref_tag, cond in (
        ("e2_spec_k4_t06", "e2_spec_off_t06", "对照项:spec off(16 并发复述任务, temp 0.6)"),
        ("e2_spec_k8_t06", "e2_spec_off_t06", "对照项:spec off(16 并发复述任务, temp 0.6)"),
        ("e2_spec_k8_t10", "e2_spec_off_t06", "对照项:spec off(temp 1.0 → α 更低)"),
        ("e2_spec_k8_open", "e2_spec_off_open", "对照项:spec off(开放生成 → α≈0)"),
    ):
        row(tag.replace("e2_", ""), tag, ref_tag, cond)
    out("| **E2b speculative — 小批量(0.6B,d3_spec_small,out 1024)** |")
    for n in (1, 2, 4):
        row(f"batch {n}, K=4", f"e2s_dense_k4_n{n}", f"e2s_dense_off_n{n}",
            f"对照项:同 batch 不开投机")
    row("batch 1, K=2", "e2s_dense_k2_n1", "e2s_dense_off_n1", "对照项:同 batch 不开投机")
    row("batch 1, K=8", "e2s_dense_k8_n1", "e2s_dense_off_n1", "对照项:同 batch 不开投机")
    out("| **E2c speculative — 小批量(完整 30B MoE,这是投机该生效的 regime)** |")
    for n in (1, 2, 4):
        row(f"batch {n}, K=4", f"e2s_moe_k4_n{n}", f"e2s_moe_off_n{n}",
            f"对照项:同 batch 不开投机")
    out("| **E3 MoE** (完整 30B-A3B;Triton 分组 GEMM vs 逐专家循环) |")
    row("prefill 重载 Triton", "e3_moe_main", "e3_moe_ref", "对照项:逐专家参考实现(19 请求)")
    row("decode 重载 Triton", "e3d_triton", "e3d_ref", "对照项:逐专家参考实现(8×1024 out)")
    row("大 prefill Triton", "e3p_triton", "e3p_ref", "对照项:逐专家参考实现(12×3584)")
    row("MoE 逐专家循环(19 请求)", "e3_moe_ref", "", "自身即基线")
    out("| **E4 TP** (对照 = tp=1) |")
    for tp in (2, 4):
        row(f"tp={tp}", f"e4_tp{tp}_t3", "r1_baseline", "对照项:tp=1")
    out("")

    # TTFT broken down by request class (this is where chunked prefill shows up)
    per_class = {t: (r.get("ttft_by_class") or {}) for t, r in results.items()}
    per_class = {t: v for t, v in per_class.items() if v}
    if per_class:
        classes = sorted({c for v in per_class.values() for c in v})
        out("### TTFT p50 by request class (seconds)\n")
        out("| config | " + " | ".join(classes) + " |")
        out("|" + "---|" * (len(classes) + 1))
        for t in sorted(per_class):
            v = per_class[t]
            cells = " | ".join(fmt((v.get(c) or {}).get("p50"), 3) for c in classes)
            out(f"| `{t}` | {cells} |")
        out("")

    # MoE mechanism detail
    moe = get("e3_moe_main")
    if not moe:
        moe = get("e3_moe_ref")
    if moe:
        out("### MoE mechanism detail\n")
        out(f"- tokens routed: {moe.get('moe_hist') and sum(moe['moe_hist'])} slots, "
            f"experts touched: {sum(1 for x in (moe.get('moe_hist') or []) if x > 0)}")
        hist = moe.get("moe_hist")
        if hist:
            loads = ", ".join(str(x) for x in hist)
            out(f"- per-expert slot counts: {loads}")
        if moe.get("moe_route_sec") is not None:
            out(f"- router time total: {moe['moe_route_sec']:.3f}s")
        if moe.get("moe_pad_efficiency") is not None:
            out(f"- grouped-GEMM 补齐效率 real/padded slots: "
                f"**{moe['moe_pad_efficiency']:.3f}** (1.0 = 无浪费;BLOCK_M=64)")
            out(f"- 每专家每步平均 slot 数: {fmt(moe.get('moe_slots_per_expert_per_call'), 1)}")
        out("")

    # ---------------- table 2: combination ---------------- #
    out("## Table 2 - cumulative combination (all on t3_mixed)\n")
    base = get("r1_baseline")
    c1 = get("c1_chunk_t3")
    c2 = get("c2_spec_t3")
    c3 = get("c3_both_t3")
    out("| step | config | tok/s | speedup vs base | incremental |")
    out("|---|---|---|---|---|")
    for label, r in (("baseline (graph on)", base), ("+ chunked prefill (budget 1024)", c1),
                     ("+ speculative K=4 (alone)", c2), ("+ both", c3)):
        if r is None:
            out(f"| {label} | - | - | - | not run |")
            continue
        sp = rel(r["median_wall_s"], (base or {}).get("median_wall_s"))
        inc = "-"
        if base and c1 and r is c1:
            inc = rel(r["median_wall_s"], base["median_wall_s"])
        out(f"| {label} | `{r['tag']}` | {fmt(r['throughput_tok_s'], 1)} | {sp} | {inc} |")
    if base and c1 and c2 and c3:
        s1 = base["median_wall_s"] / c1["median_wall_s"]
        s2 = base["median_wall_s"] / c2["median_wall_s"]
        s3 = base["median_wall_s"] / c3["median_wall_s"]
        out(f"\n- product of the two single-item speedups: **{s1 * s2:.3f}x**, "
            f"measured together: **{s3:.3f}x**")
        out(f"- interaction (measured / product): **{s3 / (s1 * s2):.3f}** "
            f"({'sub-additive' if s3 < s1 * s2 else 'super-additive'})")
        out("- expected sources of sub-additivity: speculative verify runs on the prefill "
            "path (no CUDA graph) and competes with chunked prefill for the same budget.")
    out("")

    # ---------------- table 3: regression ---------------- #
    out("## Table 3 - regression vs the original nano-vllm\n")
    cur, org = get("r1_baseline"), ref_of("r1_baseline")
    if org is None:
        out("_no original-engine data found._ To produce it, run the same command in the "
            "original checkout:\n")
        out("```bash\npython bench/nanovllm_bench.py one --tag r1_baseline --engine-tag orig \\\n"
            "    --model <MODEL> --dataset t3_mixed.jsonl --dataset-dir bench/datasets \\\n"
            "    --out results_orig --runs 3 --no-cuda-graph false\n"
            "cp results_orig/raw/r1_baseline.json results/raw/r1_baseline.orig.json\n```\n")
    else:
        out("| engine | tok/s | median wall | 提速 | delta | verdict |")
        out("|---|---|---|---|---|---|")
        if cur:
            delta = (cur["throughput_tok_s"] - org["throughput_tok_s"]) / org["throughput_tok_s"] * 100
            verdict = "no regression" if delta > -2 else "REGRESSION (investigate)"
            out(f"| current | {fmt(cur['throughput_tok_s'], 1)} | {fmt(cur['median_wall_s'])} s | "
                f"**{rel(cur['median_wall_s'], org['median_wall_s'])}** | {delta:+.1f}% | {verdict} |")
        out(f"| original | {fmt(org['throughput_tok_s'], 1)} | {fmt(org['median_wall_s'])} s | "
            f"1.000x | - | baseline |")
        out("")

    # ---------------- appendix ---------------- #
    out("## Appendix - all raw rows\n")
    out("| tag | engine | dataset | tp | budget | graph | spec | K | tok/s | vs 原版 | wall s | "
        "spread s | TTFT p50 | TTFT p99 | peak GiB | alpha | startup s |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, r in sorted(results.items()):
        o = results.get(tag.rsplit(".orig", 1)[0] + ".orig") if not tag.endswith(".orig") else None
        out(f"| `{tag}` | {r.get('engine_tag', '?')} | {Path(r['dataset']).name} | {r.get('tp')} | "
            f"{r.get('budget')} | {not r.get('no_cuda_graph')} | {r.get('spec')} | {r.get('spec_k')} | "
            f"{fmt(r.get('throughput_tok_s'), 1)} | {rel(r.get('median_wall_s'), (o or {}).get('median_wall_s'))} | "
            f"{fmt(r.get('median_wall_s'))} | "
            f"{fmt(r.get('wall_spread_s'))} | {fmt(r.get('ttft_p50'), 3)} | {fmt(r.get('ttft_p99'), 3)} | "
            f"{fmt(r.get('peak_gib'))} | {fmt(r.get('alpha'), 3)} | {fmt(r.get('startup_s'), 1)} |")
    out("")

    text = "\n".join(lines)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten to {Path(args.out).resolve()}")


def cmd_verify(args):
    """Offline checkpoint checks that need no GPU and almost no RAM."""
    ok_all = True
    for d in args.model:
        print("=" * 60)
        print(d)
        if not Path(d).is_dir():
            print("  MISSING DIR")
            ok_all = False
            continue
        hf = load_hf_config(d)
        files = sorted(Path(d).glob("*.safetensors"))
        size = sum(p.stat().st_size for p in files)
        print(f"  model_type={hf.get('model_type')} layers={hf.get('num_hidden_layers')} "
              f"experts={hf.get('num_experts')} ctx={hf.get('max_position_embeddings')}")
        print(f"  {len(files)} safetensors files, {size / 2**30:.1f} GiB")
        names_ok = check_names(d, hf)
        shapes_ok = check_shapes(d, hf)
        ok = names_ok and shapes_ok
        if args.load:
            try:
                ok = verify_load(d) and ok
            except Exception as exc:
                print(f"  [load] failed: {type(exc).__name__}: {exc}")
                ok = False
        print("  RESULT:", "PASS" if ok else "FAIL")
        ok_all = ok_all and ok
    raise SystemExit(0 if ok_all else 1)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="nano-vllm ablation benchmark suite")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="build T1/T2/T3 datasets")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="bench/datasets")
    p.add_argument("--corpus", nargs="*", default=None)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-gib", type=float, default=8.0)
    p.add_argument("--gpu-util", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=4096,
                   help="must be >= prompt + max_tokens of every request")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("tiny-moe", help="synthesize a small MoE checkpoint")
    p.add_argument("--base", required=True, help="model dir providing the tokenizer")
    p.add_argument("--out", default="bench/models/tiny-moe")
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--moe-intermediate", type=int, default=512)
    p.add_argument("--max-position", type=int, default=4096)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify-load", action="store_true", help="CPU load check (gloo, no GPU)")
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=cmd_tiny_moe)

    p = sub.add_parser("one", help="run one configuration")
    p.add_argument("--tag", required=True)
    p.add_argument("--engine-tag", default="current")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-dir", default="bench/datasets")
    p.add_argument("--out", default="results")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--budget", type=int, default=16384)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-util", type=float, default=0.9)
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--spec", action="store_true")
    p.add_argument("--spec-k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--moe-stats", action="store_true")
    p.add_argument("--moe-reference", action="store_true")
    p.add_argument("--moe-free-experts", action="store_true",
                   help="release per-expert params after stacking (benchmark-only; "
                        "the Triton path otherwise needs ~2x the model memory)")
    p.add_argument("--fair-chunk", action="store_true",
                   help="experiment: fair per-request prefill quantum (chunk <= budget/len(waiting))")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--expect-root", default=None,
                   help="abort unless nanovllm was imported from under this path (baseline runs)")
    p.set_defaults(func=cmd_one)

    p = sub.add_parser("run", help="run the whole matrix")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-dir", default="bench/datasets")
    p.add_argument("--moe-model", default=None,
                   help="full MoE model dir for E3 (e.g. Qwen3-30B-A3B); E3 is skipped without it")
    p.add_argument("--out", default="results")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--experiments", default="r1,e1,e2,e3,combo,e4")
    p.add_argument("--engine-tag", default="current")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-util", type=float, default=0.9)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gpu-count", type=int, default=0, help="override GPU count (dry-run)")
    p.add_argument("--gpu-gib", type=float, default=8.0, help="GiB per GPU (dry-run estimate)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--moe-free-experts", action="store_true",
                   help="release per-expert params after stacking (benchmark-only hook)")
    p.add_argument("--fair-chunk", action="store_true",
                   help="experiment: fair per-request prefill quantum")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("verify", help="offline checkpoint checks (names/shapes, optional CPU load)")
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--load", action="store_true", help="also load through the real loader (needs RAM)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("analyze", help="raw json -> report tables")
    p.add_argument("--raw", default="results/raw")
    p.add_argument("--out", default="BENCH_RESULTS.md")
    p.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
