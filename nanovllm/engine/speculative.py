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
