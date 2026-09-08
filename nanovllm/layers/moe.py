import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import ReplicatedLinear, MergedColumnParallelLinear, RowParallelLinear


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


    def _route(self, x):
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = routing_weights.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = (topk_weights * self.routing_scale).to(x.dtype)
        return topk_weights, topk_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_reference(x)

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