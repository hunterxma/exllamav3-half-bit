#!/usr/bin/env python3
"""
Fit an EXL3 MoE model to this machine: the smallest number of experts per layer to run on the CPU worker
(EXL3_MOE_CPU_SPLIT) at which the model loads with the requested context and generates. With more than one GPU the
model is split across all of them (PCI bus order; CUDA_VISIBLE_DEVICES limits the cards; a VRAM budget per card: its free memory less a reserve, the reserve
raised before any expert moves to the CPU), and a model that fits whole runs every expert on the GPUs (layers stored at
two bitrates as two fused modules, exllamav3's GPU rate tier). Starts from an estimate
(free VRAM vs the model's expert and non-expert bytes), then moves 8 experts per layer to the CPU per failed attempt,
or back to the GPU while attempts still fit; each attempt runs in its own process so a failed load leaves nothing
behind. Each attempt loads through model_init with exactly the arguments run.sh passes to chat.py (RUN_ARGS: 8-bit
cache, MTP drafting, chunk size), so the fit holds for what runs. Checks free RAM for the CPU-resident experts. Writes {"split", "ctx", "args", "env", "gpu",
"vram_mib", "tok_s", "sample"} to --out.

    python examples/fit.py <model dir> [--ctx 32768] [--split N] --out fit.json

install.sh runs this when given MODEL=<model dir> and writes run.sh (chat) and serve.sh (OpenAI-compatible server)
with the settings it finds.
"""
import argparse, json, os, re, subprocess, sys, math

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--ctx", type = int, default = 32768)
ap.add_argument("--split", type = int); ap.add_argument("--out", required = True)
ap.add_argument("--attempt", action = "store_true", help = argparse.SUPPRESS)
ap.add_argument("--gs", help = argparse.SUPPRESS)                 # attempt: per-GPU budgets, GiB ("9.6,14.4")
ap.add_argument("--ndt", type = int, default = 4, help = argparse.SUPPRESS)
args = ap.parse_args()

# chat.py arguments, also written to run.sh: 8-bit KV cache; MTP drafts, each drafted only at >= 85% drafter
# probability, through a 16k-column sparse draft head; 2048-token prefill chunks (+46-54% prompt speed at 8k-30k for
# +84 MiB). Drafts per step: 4 with experts on the CPU (one 16 GB card: 109.8 tok/s at 32k, 88% accepted), 6 with every
# expert on the GPUs, where verifying a longer draft costs less (two GPUs: 170 vs 160 tok/s at 32k, 166 vs 159 at 96k)
def run_args(ndt, gs = None):
    return (["-cs", str(args.ctx), "-cq", "8", "-mtp", "-ndt", str(ndt), "-dpm", "0.85", "-sdh", "16384",
             "-chunk_size", "2048", "-gcs", "2048"] + (["-gs", gs] if gs else []))
RUN_ARGS = run_args(args.ndt, args.gs)
DRAFTS_GPU, DRAFTS_CPU = 6, 4

if args.attempt:   # one load + generation in this process; prints a JSON line
    import time, torch
    from exllamav3 import Generator, Job, ArgmaxSampler, model_init
    p = argparse.ArgumentParser(); model_init.add_args(p, cache = True, add_draft_model_args = True)
    p.add_argument("-gcs", type = int)
    a = p.parse_args(["-m", args.model] + RUN_ARGS)
    model, config, cache, tok, dmodel, dconfig, dcache = model_init.init(a, quiet = True)
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dmodel, draft_cache = dcache,
                    num_draft_tokens = a.num_draft_tokens, draft_p_min = a.draft_p_min,
                    sparse_draft_head = a.sparse_draft_head, max_chunk_size = a.gcs)
    # chat.py's qwen35 prompt format (thinking off), so the smoke test needs no transformers install
    ids = tok.encode("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n"
                     "Name three prime numbers.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                     encode_special_tokens = True)
    gen.enqueue(Job(input_ids = ids, max_new_tokens = 48, sampler = ArgmaxSampler(), stop_conditions = config.eos_token_id_list + [tok.single_id("<|im_end|>")]))
    out, n, t0 = "", 0, time.time()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            out += r.get("text", ""); n += len(r.get("token_ids", torch.empty(0)).flatten()) if "token_ids" in r else 0
    dt = time.time() - t0
    print(json.dumps({"ok": bool(out.strip()), "sample": out.strip()[:200], "tok_s": round(n / dt, 1) if dt else None,
                      "vram_mib": torch.cuda.max_memory_reserved() // 2**20}))
    sys.exit(0)

# Device order for the fit and for run.sh / serve.sh: PCI bus order (nvidia-smi's numbering) unless the user set one
DEVICE_ORDER = os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

def visible_gpus():
    """[(name, free MiB)] of the GPUs the runs will see, in their CUDA order. Asks CUDA in a child process with the
    runs' device order, so CUDA_VISIBLE_DEVICES is respected (nvidia-smi always lists every GPU)."""
    code = ("import json, torch; print(json.dumps([[torch.cuda.get_device_name(i), torch.cuda.mem_get_info(i)[0] >> 20]"
            " for i in range(torch.cuda.device_count())]))")
    r = subprocess.run([sys.executable, "-c", code], env = dict(os.environ, CUDA_DEVICE_ORDER = DEVICE_ORDER),
                       capture_output = True, text = True)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("[")), None)
    if r.returncode != 0 or not line or not json.loads(line):
        sys.exit("no CUDA GPU visible" + (f" (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
                                          if "CUDA_VISIBLE_DEVICES" in os.environ else "")
                 + (f": {r.stderr.strip().splitlines()[-1]}" if r.stderr.strip() else ""))
    return json.loads(line)

names, frees = map(list, zip(*visible_gpus()))
gpu, free_mib = " + ".join(names), sum(frees)
multi = len(frees) > 1
RESERVES = [1.6, 2.1, 2.6, 3.0] if multi else [None]   # GiB per card left outside the budget (cache, activations)

def budgets(reserve):
    """-gs for the multi-GPU split: each card's free memory less the reserve, GiB"""
    return ",".join(f"{max(f / 1024 - reserve, 1.0):.1f}" for f in frees) if reserve is not None else None
q = json.load(open(os.path.join(args.model, "quantization_config.json")))["tensor_storage"]
expert = rest = 0
for k, v in q.items():
    b = sum(t["n_bytes"] for t in v["stored_tensors"].values())
    if re.search(r"\.mlp\.experts\.\d+\.", k): expert += b
    else: rest += b
n_exp = 1 + max(int(m.group(1)) for k in q if (m := re.search(r"\.mlp\.experts\.(\d+)\.", k)))
if args.split is not None:
    split = args.split
else:
    budget = free_mib * 2**20 - rest - (2 << 30) * len(frees) - args.ctx * 40 * 1024   # transients + 8-bit cache (rough)
    split = 0 if budget >= expert else max(0, min(n_exp - 8, math.ceil(n_exp * (1 - max(budget, 0) / expert) / 8) * 8))
print(f"{gpu}: {free_mib} MiB free; model experts {expert / 1e9:.1f} GB + rest {rest / 1e9:.1f} GB; first try: {split} of {n_exp} experts per layer on the CPU", flush = True)

def runtime_env():
    """The CPU-worker settings run.sh exports (EXL3_* in the environment win):
    EXL3_MOE_CPU_SPLIT_DRAFT=0: the MTP layer keeps all its experts on the GPU (it runs once per drafted token).
    EXL3_MOE_CPU_SPLIT_STATS: per-layer routing counts; each layer keeps its most-used experts on the GPU.
    EXL3_MOE_PINNED_ARENA=1: CPU experts in a shared arena, a hot expert is moved to the GPU when it is missed.
    Layers whose experts are stored at two rates put the smaller group on the CPU (fixed by the model)."""
    env = {"EXL3_MOE_CPU_SPLIT_DRAFT": "0", "EXL3_MOE_PINNED_ARENA": "1",
           "CUDA_DEVICE_ORDER": DEVICE_ORDER}   # CUDA device i = the budgets' card i
    if "CUDA_VISIBLE_DEVICES" in os.environ:   # run.sh / serve.sh use the GPUs the fit used
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    stats = os.path.join(os.path.abspath(args.model), "expert_stats.json")
    if os.path.exists(stats): env["EXL3_MOE_CPU_SPLIT_STATS"] = stats
    env.update({k: os.environ[k] for k in ("EXL3_MOE_CPU_SPLIT_DRAFT", "EXL3_MOE_PINNED_ARENA", "EXL3_MOE_CPU_SPLIT_STATS")
                if k in os.environ})
    return env

def attempt(split, reserve = None):   # -> (result dict or None, error line or None); exits on a non-memory failure or too little RAM
    need = expert * split / n_exp
    avail = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) * 1024
    if need > avail * 0.9:
        sys.exit(f"needs {need / 1e9:.1f} GB of free RAM for {split} CPU experts per layer, {avail / 1e9:.1f} GB available")
    # the MTP layer keeps all its experts on the GPU (it runs once per drafted token): 101.6 vs 91.4 tok/s, ~0.26 GB
    env = dict(os.environ, EXL3_MOE_CPU_SPLIT = str(split), **runtime_env())
    gs = budgets(reserve)
    r = subprocess.run([sys.executable, __file__, args.model, "--ctx", str(args.ctx), "--out", args.out, "--attempt",
                        "--ndt", str(DRAFTS_GPU if split == 0 else DRAFTS_CPU)] + (["--gs", gs] if gs else []),
                       env = env, capture_output = True, text = True)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("{")), None)
    res = json.loads(line) if line else None
    if r.returncode == 0 and res and res["ok"]:
        return res, None
    err = (r.stderr.strip().splitlines() or ["no output"])[-1]
    if "EXL3_MOE_PINNED_ARENA" in err and os.environ.get("EXL3_MOE_PINNED_ARENA") != "0":
        print(f"pinned expert arena refused ({err[:120]}); retrying without it", flush = True)
        os.environ["EXL3_MOE_PINNED_ARENA"] = "0"
        return attempt(split, reserve)
    if not re.search(r"VRAM|out of memory|OutOfMemory", err):
        sys.exit(f"load/generation failed at split {split}: {err}")
    return None, err

def attempt_any(split):
    """attempt() at each per-card reserve in turn (one GPU: once): -> (result, error, reserve)"""
    for rv in RESERVES:
        res, err = attempt(split, rv)
        if res is not None:
            return res, None, rv
        if multi:
            print(f"split {split}, {rv} GiB reserve per card: does not fit ({err[:100]})", flush = True)
    return None, err, None

# Up: move 8 more experts per layer to the CPU until it fits. Down (when the first try fit and no --split was
# given): move 8 back to the GPU while it still fits, since every expert on the GPU is faster.
res, err, reserve = attempt_any(split)
first_fit = res is not None
while res is None:
    if split >= n_exp - 8:
        sys.exit(f"does not fit even with {split} experts per layer on the CPU: {err}")
    print(f"split {split}: does not fit ({err[:100]}); moving 8 more experts per layer to the CPU", flush = True)
    split += 8
    res, err, reserve = attempt_any(split)
while first_fit and args.split is None and split > 0:
    print(f"split {split}: fits; trying {split - 8}", flush = True)
    lower, _, rv = attempt_any(split - 8)
    if lower is None:
        break
    split, res, reserve = split - 8, lower, rv
drafts = DRAFTS_GPU if split == 0 else DRAFTS_CPU
res.update(split = split, ctx = args.ctx, args = run_args(drafts, budgets(reserve)), gpu = gpu, env = runtime_env(),
           drafts = drafts, gs = budgets(reserve) or "auto")
json.dump(res, open(args.out, "w"), indent = 1)
print(f"fits with {split} experts per layer on the CPU: peak VRAM {res['vram_mib']} MiB, {res['tok_s']} tok/s cold start (first call; see the model card for steady-state speed); sample: {res['sample'][:80]!r}")
