#!/usr/bin/env python3
"""Startup calibration of beta_r/b for dynamic_chunk, with fingerprint cache.

Runs a SMALL controlled (c x L) forward grid on the live engine, fits
  dt = F0 + a*Σc + beta_r*ΣD + b*Σ(c·D)
and writes {beta_r, b, brb, ...} to a json keyed by a hardware/model
fingerprint.  The scheduler reads it via LICHT_DYN_BRB_FILE.

Fingerprint = sha1(model, gpu_name, dtype, max_model_len).  If a cached json
with the SAME fingerprint already exists, calibration is skipped (instant) —
so it runs ONCE per (model, hardware), and auto-recomputes when any of those
change.  c=1 rows anchor beta_r; large-c rows identify b.

Usage: python calibrate_brb.py <gpu> [cache_dir]
  -> writes <cache_dir>/brb_<fingerprint>.json  and prints its path.
"""
import os, sys, json, asyncio, random, hashlib

GPU = sys.argv[1] if len(sys.argv) > 1 else "1"
CACHE_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "brb_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["VLLM_USE_V1"] = "1"

# Config MUST match the served engine (else wrong fingerprint + wrong model).
# Passed from the launch script via env; defaults = current llama-8b setup.
MODEL = os.environ.get("LICHT_CAL_MODEL",
                       "/data/huggingface/models--meta-llama--Llama-3.1-8B-Instruct")
DTYPE = os.environ.get("LICHT_CAL_DTYPE", "float16")
MAX_MODEL_LEN = int(os.environ.get("LICHT_CAL_MAXLEN", "0"))   # 0 = model default
GMU = float(os.environ.get("LICHT_CAL_GMU", "0.95"))
# beta_r/b depend on parallelism (TP shards compute per GPU + adds comm), so
# they are part of the fingerprint AND the calibration engine must use them.
TP = int(os.environ.get("LICHT_CAL_TP", "1"))   # tensor_parallel_size
PP = int(os.environ.get("LICHT_CAL_PP", "1"))   # pipeline_parallel_size
_EFFLEN = MAX_MODEL_LEN if MAX_MODEL_LEN > 0 else 120000        # cap the L sweep
# small grid: c=1 anchors beta_r, large c identifies b
CS = [1, 1024, 4096]
LS = [L for L in [0, 20000, 50000, 80000] if L + max(CS) < _EFFLEN]
REPS = 3

def fingerprint(gpu_name, n_gpu):
    # n_gpu = number of visible GPUs (must match TP*PP); gpu_name = their model.
    key = "|".join([MODEL, gpu_name, DTYPE, str(MAX_MODEL_LEN),
                    "tp%d" % TP, "pp%d" % PP, "ngpu%d" % n_gpu])
    return hashlib.sha1(key.encode()).hexdigest()[:16]

def _gpu_name():
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", GPU, "--query-gpu=name",
             "--format=csv,noheader"], text=True).strip().splitlines()[0]
        return out.strip()
    except Exception:
        return "unknown_gpu"

async def main():
    gpu_name = _gpu_name()          # via nvidia-smi -> no CUDA init in parent
    n_gpu = len([g for g in GPU.split(",") if g.strip() != ""])
    fp = fingerprint(gpu_name, n_gpu)
    out = os.path.join(CACHE_DIR, f"brb_{fp}.json")
    if os.path.exists(out):
        d = json.load(open(out))
        print(f"CACHED brb={d['brb']:.1f} (fingerprint {fp}, {gpu_name}) -> {out}",
              flush=True)
        print("BRB_RESULT_PATH=" + out, flush=True)
        return out

    import numpy as np
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm import SamplingParams

    probe = out + ".steps.jsonl"
    open(probe, "w").close()
    os.environ["LICHT_BRB_PROBE"] = probe
    eargs = dict(
        model=MODEL, gpu_memory_utilization=GMU,
        tensor_parallel_size=TP, pipeline_parallel_size=PP,
        max_num_batched_tokens=265944, enforce_eager=True, dtype=DTYPE,
        trust_remote_code=True, enable_chunked_prefill=True,
        long_prefill_token_threshold=0, licht_v2=True)
    if MAX_MODEL_LEN > 0:
        eargs["max_model_len"] = MAX_MODEL_LEN
    args = AsyncEngineArgs(**eargs)
    engine = AsyncLLM.from_engine_args(args)
    print(f"ENGINE UP calib gpu={GPU} ({gpu_name}) fingerprint={fp}", flush=True)
    rng = random.Random(0); n = 0
    for L in LS:
        for C in CS:
            if L + C >= _EFFLEN:        # _EFFLEN = maxlen or 120000 if 0/auto
                continue
            for _ in range(REPS):
                prompt = [rng.randint(100, 30000) for _ in range(L + C)]
                rid = "exp_%d_L%d_cal" % (n, L); n += 1
                async for _ in engine.generate(
                        TokensPrompt(prompt_token_ids=prompt),
                        SamplingParams(max_tokens=1, temperature=0.0), rid):
                    pass
                await asyncio.sleep(0.04)
    await asyncio.sleep(0.5)
    engine.shutdown()

    R = [json.loads(l) for l in open(probe) if l.strip()]
    R = [r for r in R if r["n_sched"] > 0 and r["dt"] > 0 and r["sum_c"] > 0
         and r["sum_ctx"] > 0]
    # Guard: too few points -> do NOT write a json (would poison the cache with
    # a garbage brb forever). Leave no file -> scheduler falls back to default,
    # and a future launch retries.
    if len(R) < 8:
        print("CALIB FAILED: only %d valid steps (engine/probe issue); not "
              "caching -> default brb used." % len(R), flush=True)
        return None
    sc = np.array([r["sum_c"] for r in R], float)
    sd = np.array([r["sum_ctx"] for r in R], float)
    scd = np.array([r["sum_c_ctx"] for r in R], float)
    dt = np.array([r["dt"] for r in R], float) * 1000.0
    X = np.column_stack([np.ones_like(sc), sc, sd, scd])
    coef, *_ = np.linalg.lstsq(X, dt, rcond=None)
    F0, a, beta_r, b = [float(x) for x in coef]
    if b <= 0:
        print("CALIB FAILED: fit gave b=%.2e <= 0; not caching." % b, flush=True)
        return None
    pred = X @ coef
    r2 = float(1 - ((dt - pred) ** 2).sum() /
               max(((dt - dt.mean()) ** 2).sum(), 1e-9))
    brb = beta_r / b if b > 0 else 350.0
    rec = {"beta_r_ms_per_tok": beta_r, "b_ms_per_c_ctx": b, "brb": brb,
           "F0_ms": F0, "a_ms_per_tok": a, "r2": r2, "n_steps": len(R),
           "fingerprint": fp, "gpu": gpu_name, "n_gpu": n_gpu, "model": MODEL,
           "dtype": DTYPE, "max_model_len": MAX_MODEL_LEN, "tp": TP, "pp": PP}
    json.dump(rec, open(out, "w"), indent=2)
    print("CALIB DONE r2=%.3f beta_r=%.3e b=%.3e brb=%.1f -> %s"
          % (r2, beta_r, b, brb, out), flush=True)
    print("BRB_RESULT_PATH=" + out, flush=True)
    return out

if __name__ == "__main__":
    asyncio.run(main())
