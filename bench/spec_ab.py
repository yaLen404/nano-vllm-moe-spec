#!/usr/bin/env python3
"""Re-run missing spec A/B configs, saving to a durable repo dir."""
import os, sys, time, statistics, subprocess, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
OUT = f"{ROOT}/results_spec_graph/raw"
os.makedirs(OUT, exist_ok=True)

worker = r'''
import os, sys, time, json, statistics
sys.path.insert(0, os.environ["NANOVLLM_ROOT"])
from transformers import AutoTokenizer
M = os.environ.get("NANOVLLM_MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B"))
tok = AutoTokenizer.from_pretrained(M)
corpus = tok.encode(open(os.path.join(os.environ["NANOVLLM_ROOT"], "README.md")).read())
prefix = tok.encode("Repeat the following passage verbatim, word for word:\n\n")
prompt = prefix + corpus[:224]
from nanovllm import LLM, SamplingParams
spec = json.loads(sys.argv[1]); k = int(sys.argv[2]); batch = int(sys.argv[3]); runs = int(sys.argv[4])
llm = LLM(M, speculative=spec, max_speculative_tokens=k,
          max_num_batched_tokens=16384, max_model_len=4096,
          gpu_memory_utilization=0.85)
sp = SamplingParams(temperature=0.6, max_tokens=1024, ignore_eos=True)
prompts = [prompt] * batch
sps = [sp] * batch
llm.generate([prompt], [sp], use_tqdm=False)
walls = []
produced = 0
import torch
for _ in range(runs):
    torch.cuda.synchronize()
    t = time.perf_counter()
    outs = llm.generate(prompts, sps, use_tqdm=False)
    torch.cuda.synchronize()
    walls.append(time.perf_counter() - t)
    produced = sum(len(o["token_ids"]) for o in outs)
json.dump({"walls": walls, "produced": produced}, open(sys.argv[5], "w"))
med = statistics.median(walls)
print(f"spec={spec} K={k} batch={batch}: median {med:.2f}s -> {produced/med:.1f} tok/s", flush=True)
'''

w = f"{OUT}/worker.py"
open(w, "w").write(worker)

def run(spec, k, batch, runs, tag):
    outfile = f"{OUT}/{tag}.json"
    env = {**os.environ, "NANOVLLM_ROOT": ROOT}
    r = subprocess.run([PY, w, json.dumps(spec), str(k), str(batch), str(runs), outfile],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    print(r.stdout.strip(), flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:]); return None
    return json.load(open(outfile))

for b in (1, 2, 4):
    run(False, 4, b, 3, f"off_b{b}")          # spec-off baseline, once per batch
    for k in (4, 8):
        run(True, k, b, 3, f"on_k{k}_b{b}")

print("DONE")
