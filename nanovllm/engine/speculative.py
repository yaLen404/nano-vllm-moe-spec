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
    """GPU-batched rejection sampling (Leviathan et al.).

    Same contract as before: for each sequence, accept the longest draft prefix
    that passes the rejection test, then append one bonus/replacement token. The
    acceptance math is identical; only the execution changed — softmax, the
    acceptance-prefix cumprod and the bonus/replacement draws now run as a few
    batched kernels, and the GPU->CPU sync happens twice overall instead of
    several times per sequence.

    Args:
        drafts:       list[list[int]], K_i proposed tokens per sequence.
        logits:       list[Tensor], each [K_i + 1, vocab] target logits (GPU) for
                      positions (L-1 .. L+K_i-1); logits[i] predicts position L+i.
        temperatures: list[float] per sequence.

    Returns:
        list of (accepted_tokens, num_accepted_draft): accepted_tokens are the
        a accepted draft tokens plus one bonus/replacement token; num_accepted_draft
        is a (used for KV rollback).
    """
    n = len(drafts)
    results = [None] * n
    active = [i for i in range(n) if len(drafts[i]) > 0]
    for i in range(n):
        if len(drafts[i]) == 0:
            results[i] = ([], 0)
    if not active:
        return results

    # one softmax over every sequence's logits, each scaled by its own temperature
    lg_all = torch.cat([logits[i].float() / temperatures[i] for i in active], dim=0)
    probs_all = torch.softmax(lg_all, dim=-1)          # [sum(K_i+1), vocab]
    dev = lg_all.device

    # row offsets inside lg_all (K_i draft rows + 1 final row per sequence) and
    # element offsets inside the flattened per-draft flag vector (K_i elements).
    offs, doffs, Ks = [], [], []
    o = d = 0
    for i in active:
        K = len(drafts[i])
        offs.append(o)
        doffs.append(d)
        Ks.append(K)
        o += K + 1
        d += K

    draft_tokens = [t for i in active for t in drafts[i]]
    row_ids = torch.tensor([offs[k] + t for k in range(len(active)) for t in range(Ks[k])],
                           dtype=torch.long, device=dev)
    col_ids = torch.tensor(draft_tokens, dtype=torch.long, device=dev)
    p_draft = probs_all[row_ids, col_ids]              # [sum K]

    u = torch.rand(p_draft.shape, device=dev)
    flags = (u < p_draft).float()

    # padded 2D cumprod: pad = 0 makes the running product die at the end of each
    # real segment, so the column-wise sum equals the accepted-prefix length a_i
    Kmax = max(Ks)
    acc2d = torch.zeros(len(active), Kmax, device=dev)
    for k in range(len(active)):
        acc2d[k, :Ks[k]] = flags[doffs[k]:doffs[k] + Ks[k]]
    a_list = acc2d.cumprod(dim=1).sum(dim=1).long().tolist()   # one sync for all a_i

    # batched bonus / replacement draws
    last_rows = torch.tensor([offs[k] + Ks[k] for k in range(len(active))], device=dev)
    sampled = torch.empty(len(active), dtype=torch.long, device=dev)
    full = [k for k in range(len(active)) if a_list[k] == Ks[k]]
    rej = [k for k in range(len(active)) if a_list[k] < Ks[k]]
    if full:
        rows = torch.tensor([last_rows[k] for k in full], device=dev)
        sampled[full] = torch.multinomial(probs_all[rows], 1).squeeze(1)
    if rej:
        rows = torch.tensor([offs[k] + a_list[k] for k in rej], device=dev)
        adj = probs_all[rows].clone()
        d = torch.tensor([drafts[active[k]][a_list[k]] for k in rej], device=dev)
        adj[torch.arange(len(rej), device=dev), d] = 0.0
        s = adj.sum(dim=1, keepdim=True)
        adj = torch.where(s > 0, adj / s, probs_all[rows])
        sampled[rej] = torch.multinomial(adj, 1).squeeze(1)
    sampled_list = sampled.tolist()                      # second (and last) sync

    for k, i in enumerate(active):
        a = a_list[k]
        results[i] = (drafts[i][:a] + [sampled_list[k]], a)
    return results
