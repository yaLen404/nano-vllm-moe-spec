import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.speculative import NgramDraftHead, speculative_accept


class LLMEngine:

    def __init__(self, model, speculative: bool = False, max_speculative_tokens: int = 8, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        if speculative:
            config.max_speculative_tokens = max_speculative_tokens
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.draft_head = NgramDraftHead(max_k=max_speculative_tokens) if speculative else None
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        prefill_seqs, decode_seqs = self.scheduler.schedule()
        prefill_tokens = decode_tokens = 0
        outputs = []
        if prefill_seqs:
            prefill_tokens = sum(seq.num_scheduled_tokens for seq in prefill_seqs)
            token_ids = self.model_runner.call("run", prefill_seqs, True)
            self.scheduler.postprocess(prefill_seqs, token_ids, True)
        if decode_seqs:
            decode_tokens = len(decode_seqs)
            if self.draft_head is None:
                token_ids = self.model_runner.call("run", decode_seqs, False)
                self.scheduler.postprocess(decode_seqs, token_ids, False)
            else:
                self._speculative_decode(decode_seqs)
        for seq in prefill_seqs + decode_seqs:
            if seq.is_finished:
                outputs.append((seq.seq_id, seq.completion_token_ids))
        return outputs, prefill_tokens, decode_tokens

    def _speculative_decode(self, decode_seqs):
        drafts = []
        for seq in decode_seqs:
            remaining = seq.max_tokens - seq.num_completion_tokens
            if remaining <= 1:
                drafts.append([])
            else:
                k = min(self.draft_head.max_k, remaining - 1)
                drafts.append(self.draft_head.draft(seq.token_ids)[:k])

        spec_seqs = [s for s, d in zip(decode_seqs, drafts) if d]
        spec_drafts = [d for d in drafts if d]
        fallback_seqs = [s for s, d in zip(decode_seqs, drafts) if not d]

        if fallback_seqs:
            token_ids = self.model_runner.call("run", fallback_seqs, False)
            self.scheduler.postprocess(fallback_seqs, token_ids, False)

        if spec_seqs:
            for seq, draft in zip(spec_seqs, spec_drafts):
                self.scheduler.block_manager.extend_for_tokens(seq, self.draft_head.max_k)
            verify_logits = self.model_runner.call("run_speculative", spec_seqs, spec_drafts)
            temperatures = [seq.temperature for seq in spec_seqs]
            results = speculative_accept(spec_drafts, verify_logits, temperatures)
            self.scheduler.postprocess_speculative(spec_seqs, results)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, prefill_tokens, decode_tokens = self.step()
            dt = perf_counter() - t
            if prefill_tokens > 0:
                prefill_throughput = prefill_tokens / dt
            if decode_tokens > 0:
                decode_throughput = decode_tokens / dt
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs