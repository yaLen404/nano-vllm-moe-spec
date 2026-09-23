from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        prefill_seqs: list[Sequence] = []
        decode_seqs: list[Sequence] = []
        num_batched_tokens = 0

        # ---- decode first: service existing decoders every step (bounded latency) ----
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

        # ---- chunked prefill: round-robin, every waiting seq gets one chunk ----
        for _ in range(len(self.waiting)):
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
            chunk = min(num_tokens, remaining)
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

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
    def postprocess_speculative(self, seqs: list[Sequence], results: list):
        for seq, (accepted_tokens, num_accepted_draft) in zip(seqs, results):
            L = seq.num_tokens
            # stop early on EOS / max_tokens (mid-sequence truncation)
            stop = len(accepted_tokens)
            for idx, t in enumerate(accepted_tokens):
                if (not seq.ignore_eos and t == self.eos) or seq.num_completion_tokens + idx + 1 >= seq.max_tokens:
                    stop = idx + 1
                    break
            # keep KV only for the draft tokens actually committed
            self.block_manager.truncate(seq, L + min(stop, num_accepted_draft))
            seq.num_scheduled_tokens = 0
            for t in accepted_tokens[:stop]:
                seq.append_token(t)
            if (not seq.ignore_eos and seq.last_token == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
