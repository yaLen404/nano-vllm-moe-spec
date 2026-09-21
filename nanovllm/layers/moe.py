import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import triton
import triton.language as tl

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import ReplicatedLinear, MergedColumnParallelLinear, RowParallelLinear

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 64


@triton.jit
def grouped_gemm_kernel(
    a_ptr,            # [P, K]  activations, already in sorted/padded expert order
    b_ptr,            # [E, N, K] stacked expert weights
    c_ptr,            # [P, N]  output, in the same sorted/padded order
    expert_ids_ptr,   # [num_blocks_m] int32, expert id of each token block
    N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    expert_id = tl.load(expert_ids_ptr + pid_m)

    b_base = b_ptr + expert_id * N * K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        off_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + off_m[:, None] * K + off_k[None, :]
        a = tl.load(a_ptrs, mask=off_k[None, :] < K, other=0.0)
        b_ptrs = b_base + off_n[:, None] * K + off_k[None, :]
        b = tl.load(b_ptrs, mask=(off_n[:, None] < N) & (off_k[None, :] < K), other=0.0)
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)

    c_ptrs = c_ptr + off_m[:, None] * N + off_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=off_n[None, :] < N)


def moe_align(topk_ids, topk_weights, num_experts, block_size):
    """Sort (token, slot) pairs by expert and pad each expert to a multiple of block_size.

    Returns:
        token_ids_pad: [total_padded] int32, original token id (-1 for padding)
        expert_ids:    [num_blocks] int32, expert id of each block
        valid_mask:    [total_padded] bool, True for real (non-padding) entries
        sorted_tok:    [N] int32, original token ids in sorted order
        sorted_w:      [N] float, routing weights aligned with sorted_tok
    """
    T, K = topk_ids.shape
    device = topk_ids.device
    flat_ids = topk_ids.reshape(-1)      # [N]
    flat_w = topk_weights.reshape(-1)    # [N]
    tok = torch.arange(T, dtype=torch.int32, device=device).repeat_interleave(K)  # [N]

    sorted_ids, order = flat_ids.sort()  # non-decreasing expert ids
    sorted_tok = tok[order]
    sorted_w = flat_w[order]

    counts = torch.bincount(sorted_ids, minlength=num_experts)                     # [E]
    padded_counts = ((counts + block_size - 1) // block_size) * block_size          # [E]
    blocks_per_expert = padded_counts // block_size                                 # [E]
    num_blocks = int(blocks_per_expert.sum().item())
    total_padded = num_blocks * block_size

    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=device), blocks_per_expert
    )

    cum_counts = torch.cumsum(counts, dim=0)
    cum_padded = torch.cumsum(padded_counts, dim=0)
    starts = cum_counts - counts          # sorted-order start of each expert
    pstarts = cum_padded - padded_counts  # padded-order start of each expert
    positions = torch.arange(flat_ids.numel(), device=device)
    sorted_padded_pos = pstarts[sorted_ids] + (positions - starts[sorted_ids])

    token_ids_pad = torch.full((total_padded,), -1, dtype=torch.int32, device=device)
    token_ids_pad[sorted_padded_pos] = sorted_tok
    valid_mask = token_ids_pad >= 0

    return token_ids_pad, expert_ids, valid_mask, sorted_tok, sorted_w


class ExpertMLP(nn.Module):
    """Single routed expert FFN.

    Structurally identical to the dense Qwen3MLP, except that it uses
    ``moe_intermediate_size`` instead of the dense ``intermediate_size``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class FusedMoE(nn.Module):
    """Sparse Mixture-of-Experts FFN.

    Routing follows the HuggingFace Qwen2/Qwen3-MoE semantics: softmax over the
    full router logits, then top-k, then optionally renormalize when
    ``norm_topk_prob`` is set. This is expert-tensor-parallel (ETP): every expert
    lives on every rank and is sharded along its intermediate dim.

    The CUDA fast path sorts tokens by expert and runs two grouped-GEMM kernels
    (gate-up and down) instead of a per-expert Python loop. ``forward_reference``
    keeps the original per-expert loop for correctness comparison.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        norm_topk_prob: bool = False,
        has_shared_experts: bool = False,
        routing_scale: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.routing_scale = routing_scale
        self.tp_size = dist.get_world_size()

        # Router must be replicated across TP ranks so every rank selects the
        # same experts for a given token.
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)])
        self.shared_experts = ExpertMLP(hidden_size, intermediate_size) if has_shared_experts else None

        self._w1 = None
        self._w2 = None

    def _stack_weights(self):
        if self._w1 is None:
            w1 = torch.stack([e.gate_up_proj.weight.data for e in self.experts]).contiguous()  # [E, 2mi, H]
            w2 = torch.stack([e.down_proj.weight.data for e in self.experts]).contiguous()      # [E, H, mi]
            self._w1 = w1
            self._w2 = w2
        return self._w1, self._w2

    def _route(self, x):
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = routing_weights.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = (topk_weights * self.routing_scale).to(x.dtype)
        return topk_weights, topk_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.numel() > 0:
            return self.forward_triton(x)
        return self.forward_reference(x)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        T, H = x.shape
        topk_weights, topk_ids = self._route(x)

        token_ids_pad, expert_ids, valid_mask, sorted_tok, sorted_w = moe_align(
            topk_ids, topk_weights, self.num_experts, _BLOCK_M
        )
        P = token_ids_pad.numel()

        w1, w2 = self._stack_weights()
        E1, N1, K1 = w1.shape   # N1 = 2 * intermediate_size // tp, K1 = hidden_size
        E2, N2, K2 = w2.shape   # N2 = hidden_size, K2 = intermediate_size // tp

        # Gather activations into sorted/padded expert order.
        x_sorted_pad = x[token_ids_pad.clamp(min=0)]
        x_sorted_pad = x_sorted_pad.masked_fill_((token_ids_pad < 0).unsqueeze(1), 0)

        num_blocks_m = expert_ids.numel()

        gate_up = torch.empty((P, N1), dtype=x.dtype, device=x.device)
        grid1 = (num_blocks_m, (N1 + _BLOCK_N - 1) // _BLOCK_N)
        grouped_gemm_kernel[grid1](
            x_sorted_pad, w1, gate_up, expert_ids, N1, K1,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        )

        mi = N1 // 2
        h1 = (F.silu(gate_up[:, :mi]) * gate_up[:, mi:]).contiguous()  # [P, mi]

        out_contrib = torch.empty((P, N2), dtype=x.dtype, device=x.device)
        grid2 = (num_blocks_m, (N2 + _BLOCK_N - 1) // _BLOCK_N)
        grouped_gemm_kernel[grid2](
            h1, w2, out_contrib, expert_ids, N2, K2,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        )

        c_valid = out_contrib[valid_mask]               # [N, H]
        c_valid = c_valid * sorted_w.unsqueeze(1)
        out = torch.zeros((T, H), dtype=x.dtype, device=x.device)
        out.index_add_(0, sorted_tok, c_valid)

        if self.tp_size > 1:
            dist.all_reduce(out)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids = self._route(x)

        T = x.size(0)
        token_indices = torch.arange(T, device=x.device).repeat_interleave(self.top_k)
        flat_ids = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1)

        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = flat_ids == e
            if not mask.any():
                continue
            tok = token_indices[mask]
            w = flat_weights[mask].unsqueeze(-1)
            out.index_add_(0, tok, self.experts[e](x[tok]) * w)

        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out