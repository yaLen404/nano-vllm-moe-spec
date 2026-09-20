"""Speculative decoding: pluggable draft sources + GPU rejection sampling.

The draft source is a simple callable ``draft(token_ids) -> list[int]`` so it can
be swapped later (n-gram today, an EAGLE/DFlash draft head tomorrow) without
touching the verify / accept / rollback pipeline.
"""
import torch


class NgramDraftHead:
    """Prompt-lookup draft: match the last ``n`` tokens against history and
    propose the tokens that followed the most recent match."""

    def __init__(self, n: int = 3, max_k: int = 8):
        self.n = n
        self.max_k = max_k

    def draft(self, token_ids):
        if len(token_ids) < self.n + 1:
            return []
        suffix = tuple(token_ids[-self.n:])
        for i in range(len(token_ids) - self.n - 1, -1, -1):
            if tuple(token_ids[i:i + self.n]) == suffix:
                j = i + self.n
                return list(token_ids[j:j + self.max_k])
        return []


def speculative_accept(drafts, logits, temperatures):
    """GPU-vectorized rejection sampling (Leviathan et al.).

    Args:
        drafts:       list[list[int]], K_i proposed tokens per sequence.
        logits:       list[Tensor], each [K_i + 1, vocab] target logits (GPU) for
                      positions (L-1 .. L+K_i-1); logits[i] predicts position L+i.
        temperatures: list[float] per sequence.

    Returns:
        list of (accepted_tokens, num_accepted_draft):
          accepted_tokens   tokens to append (a accepted draft + 1 bonus/replacement)
          num_accepted_draft  a = how many DRAFT tokens were accepted (for KV rollback)
    """
    results = []
    for draft, lg, T in zip(drafts, logits, temperatures):
        K = len(draft)
        if K == 0:
            results.append(([], 0))
            continue
        lg = lg.float() / T
        probs = torch.softmax(lg, dim=-1)                       # [K+1, vocab]
        draft_t = torch.tensor(draft, dtype=torch.long, device=lg.device)
        p_draft = probs[:-1].gather(1, draft_t.unsqueeze(1)).squeeze(1)  # [K]
        u = torch.rand(K, device=lg.device)
        # cumprod gives 1 only for the longest accepted prefix
        a = int((u < p_draft).float().cumprod(0).sum().item())

        accepted = draft[:a]
        if a == K:
            bonus = int(torch.multinomial(probs[K], 1).item())
            accepted = accepted + [bonus]
        else:
            adj = probs[a].clone()
            d = draft[a]
            adj[d] = 0.0
            s = adj.sum()
            adj = adj / s if s > 0 else probs[a]
            replacement = int(torch.multinomial(adj, 1).item())
            accepted = accepted + [replacement]
        results.append((accepted, a))
    return results