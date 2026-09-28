import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

MODEL_REGISTRY = {
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
}


def get_model(hf_config):
    model_type = getattr(hf_config, "model_type", "qwen3")
    if model_type not in MODEL_REGISTRY:
        raise NotImplementedError(f"unsupported model_type: {model_type}")
    return MODEL_REGISTRY[model_type](hf_config)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        if getattr(hf_config, "model_type", None) == "qwen3_moe" and not self.enforce_eager:
            # MoE routing is data-dependent; CUDA graph capture would replay a
            # stale top-k decision, so force eager until a fused MoE kernel lands.
            self.enforce_eager = True
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        dtype_attr = getattr(hf_config, "torch_dtype", None) or getattr(hf_config, "dtype", None) or "float32"
        torch.set_default_dtype(getattr(torch, dtype_attr) if isinstance(dtype_attr, str) else dtype_attr)
        torch.set_default_device("cuda")
        self.model = get_model(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        if not self.enforce_eager and self.world_size == 1 and config.max_speculative_tokens > 0:
            self.capture_spec_graph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if hasattr(self, "spec_graphs"):
                del self.spec_graphs, self.spec_graph_pool, self.spec_graph_vars
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * torch.get_default_dtype().itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def prepare_verify(self, seqs: list[Sequence], drafts: list[list[int]]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        for seq, draft in zip(seqs, drafts):
            K = len(draft)
            L = seq.num_tokens
            start = L - 1                # position of the last committed token
            seqlen_q = K + 1             # feed [last_token] + K draft tokens
            seqlen_k = L + K             # committed KV + newly written
            input_ids.extend([seq.last_token] + draft)
            positions.extend(range(start, start + seqlen_q))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            start_block = start // self.block_size
            end_block = (start + seqlen_q + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + (start + seqlen_q) - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    @torch.inference_mode()
    def run_speculative(self, seqs: list[Sequence], drafts: list[list[int]]):
        if (self.world_size == 1 and not self.enforce_eager
                and hasattr(self, "spec_graphs") and len(seqs) <= self.spec_graph_bs[-1]):
            return self._run_speculative_graph(seqs, drafts)
        input_ids, positions = self.prepare_verify(seqs, drafts)
        hidden_states = self.model(input_ids, positions)
        logits = self.model.lm_head.forward_all(hidden_states)
        per_seq_logits = None
        if self.rank == 0:
            per_seq_logits = []
            offset = 0
            for draft in drafts:
                n = len(draft) + 1
                per_seq_logits.append(logits[offset:offset + n])
                offset += n
        reset_context()
        return per_seq_logits

    @torch.inference_mode()
    def _run_speculative_graph(self, seqs: list[Sequence], drafts: list[list[int]]):
        # Verify forward via a captured CUDA graph (dense model, TP=1). Every
        # sequence contributes a fixed `max_k + 1` query tokens: its last
        # committed token, then up to `max_k` draft tokens padded with dummy
        # ids when the n-gram draft is short. Padded positions are rolled back
        # by the scheduler and their logits are discarded, so correctness is
        # preserved while all tensor shapes stay graph-capturable.
        max_k = self.config.max_speculative_tokens
        per_seq = max_k + 1
        bs = len(seqs)
        n_tok = bs * per_seq
        gv = self.spec_graph_vars
        bs_graph = next(x for x in self.spec_graph_bs if x >= bs)
        graph = self.spec_graphs[bs_graph]
        n_slots = bs_graph * per_seq
        max_num_blocks = gv["block_tables"].size(1)

        id_list = [0] * n_slots
        pos_list = [0] * n_slots
        sm_list = [-1] * n_slots
        bt_list = [[0] + [-1] * (max_num_blocks - 1) for _ in range(bs_graph)]
        cu_k = gv["cu_seqlens_k"]
        cu_k[0] = 0
        for s in range(bs_graph):
            if s < bs:
                seq, draft = seqs[s], drafts[s]
                K = len(draft)
                L = seq.num_tokens
                start = L - 1
                base = s * per_seq
                for j in range(per_seq):
                    id_list[base + j] = seq.last_token if j == 0 else (draft[j - 1] if j <= K else 0)
                    pos_list[base + j] = min(start + j, self.config.max_model_len - 1)
                sblock = start // self.block_size
                eblock = (start + per_seq + self.block_size - 1) // self.block_size
                sm = []
                for i in range(sblock, eblock):
                    slot_start = seq.block_table[i] * self.block_size
                    if i == sblock:
                        slot_start += start % self.block_size
                    if i != eblock - 1:
                        slot_end = seq.block_table[i] * self.block_size + self.block_size
                    else:
                        slot_end = seq.block_table[i] * self.block_size + (start + per_seq) - i * self.block_size
                    sm.extend(range(slot_start, slot_end))
                sm_list[base:base + per_seq] = sm
                nb = len(seq.block_table)
                bt_list[s][:nb] = seq.block_table
                cu_k[s + 1] = cu_k[s] + (L + max_k)
            else:
                cu_k[s + 1] = cu_k[s] + per_seq

        gv["input_ids"][:n_slots].copy_(torch.tensor(id_list, dtype=torch.int64, device=gv["input_ids"].device))
        gv["positions"][:n_slots].copy_(torch.tensor(pos_list, dtype=torch.int64, device=gv["positions"].device))
        gv["slot_mapping"][:n_slots].copy_(torch.tensor(sm_list, dtype=torch.int32, device=gv["slot_mapping"].device))
        gv["block_tables"][:bs_graph].copy_(torch.tensor(bt_list, dtype=torch.int32, device=gv["block_tables"].device))

        set_context(True, gv["cu_seqlens_q"][:bs_graph + 1], cu_k[:bs_graph + 1],
                    per_seq, self.config.max_model_len,
                    gv["slot_mapping"][:n_slots], None, gv["block_tables"][:bs_graph])
        graph.replay()
        logits = self.model.lm_head.forward_all(gv["outputs"][:n_tok])
        per_seq_logits = []
        offset = 0
        for draft in drafts:
            n = len(draft) + 1
            per_seq_logits.append(logits[offset:offset + n])
            offset += per_seq
        reset_context()
        return per_seq_logits

    @torch.inference_mode()
    def capture_spec_graph(self):
        # Capture the speculative verify forward (dense model only) for a small
        # set of batch sizes. Shapes are fixed per bucket: `max_k + 1` query
        # tokens per sequence and `max_seqlen_k = max_model_len` so flash-attn
        # sizes its workspace once; only buffer VALUES change per replay.
        config = self.config
        hf_config = config.hf_config
        max_k = config.max_speculative_tokens
        per_seq = max_k + 1
        max_bs = min(config.max_num_seqs, 16)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        total = max_bs * per_seq
        input_ids = torch.zeros(total, dtype=torch.int64)
        positions = torch.zeros(total, dtype=torch.int64)
        slot_mapping = torch.zeros(total, dtype=torch.int32)
        cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(max_bs + 1, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(total, hf_config.hidden_size)
        self.spec_graph_bs = [1, 2, 4, 8, 16]
        self.spec_graphs = {}
        self.spec_graph_pool = None
        for bs in reversed(self.spec_graph_bs):
            n_tok = bs * per_seq
            graph = torch.cuda.CUDAGraph()
            cu_seqlens_q[:bs + 1] = torch.arange(0, (bs + 1) * per_seq, per_seq, dtype=torch.int32)
            cu_seqlens_k[:bs + 1] = torch.arange(0, (bs + 1) * config.max_model_len, config.max_model_len, dtype=torch.int32)
            block_tables[:bs] = torch.arange(max_num_blocks, dtype=torch.int32).repeat(bs, 1)
            slot_mapping[:n_tok] = torch.arange(n_tok, dtype=torch.int32)
            set_context(True, cu_seqlens_q[:bs + 1], cu_seqlens_k[:bs + 1], per_seq, config.max_model_len,
                        slot_mapping[:n_tok], None, block_tables[:bs])
            outputs[:n_tok] = self.model(input_ids[:n_tok], positions[:n_tok])
            with torch.cuda.graph(graph, self.spec_graph_pool):
                outputs[:n_tok] = self.model(input_ids[:n_tok], positions[:n_tok])
            if self.spec_graph_pool is None:
                self.spec_graph_pool = graph.pool()
            self.spec_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
        self.spec_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
