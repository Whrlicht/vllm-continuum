# SPDX-License-Identifier: Apache-2.0
"""Correctness + speed check for the fused scatter op.

Build first (creates licht_fused_scatter*.so):
    cd vllm/v1/core/sched/licht_v3/csrc
    export CUDA_HOME=/usr/local/cuda-12.2
    python setup.py build_ext --inplace      # local .so, or `pip install .`
    python test_fused_scatter.py
"""
import os
import random
import subprocess
import sys
import time

import torch

# allow importing the locally-built .so from this dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import licht_fused_scatter as ext  # noqa: E402

_free = subprocess.check_output(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
    "|awk -F', ' '$2+0<2000{print $1;exit}'", shell=True).decode().strip()
dev = f"cuda:{_free or 0}"
torch.zeros(1, device=dev)


def prod(t):
    p = 1
    for x in t:
        p *= x
    return p


def correctness(name, mk, nL, NBLK, dim, rest):
    P = prod(rest)
    kv = [mk(NBLK) for _ in range(nL)]
    ref = [t.clone() for t in kv]
    nb = 200
    staging = ((torch.arange(nb * nL * 2 * P, device=dev, dtype=torch.int64)
                % 97).to(torch.float16).view(nb, nL, 2, *rest))
    idx = torch.as_tensor(random.sample(range(NBLK), nb), device=dev,
                          dtype=torch.long)
    lp = torch.tensor([t.data_ptr() for t in kv], dtype=torch.int64, device=dev)
    ext.licht_scatter(staging.contiguous(), idx, lp, nb, nL, dim, NBLK, P)
    torch.cuda.synchronize()
    for li in range(nL):
        s = staging[:, li]
        if dim == 1:
            ref[li][:, idx, ...] = s.permute(1, 0, *range(2, s.dim()))
        else:
            ref[li][idx, ...] = s
    torch.cuda.synchronize()
    bad = sum(int(not torch.equal(kv[li], ref[li])) for li in range(nL))
    print(f"{name}: nb={nb} P={P} dim={dim} mismatch={bad}/{nL} -> "
          f"{'PASS' if bad == 0 else 'FAIL'}")


def speed(nL=32, NBLK=16853, rest=(16, 8, 128), cap=256, total=10000):
    P = prod(rest)
    kv = [torch.zeros((2, NBLK, *rest), dtype=torch.float16, device=dev)
          for _ in range(nL)]
    lp = torch.tensor([t.data_ptr() for t in kv], dtype=torch.int64, device=dev)
    stg = torch.empty((cap, nL, 2, *rest), dtype=torch.float16, device=dev)
    nchunk = (total + cap - 1) // cap
    idxs = [torch.as_tensor(random.sample(range(NBLK), min(cap, total - c * cap)),
                            device=dev, dtype=torch.long) for c in range(nchunk)]
    MB = 2 * P * 2 / 1e6

    def go(kernel):
        for _ in range(2):  # warm
            for c in range(nchunk):
                nb = idxs[c].numel()
                if kernel:
                    ext.licht_scatter(stg[:nb].contiguous(), idxs[c], lp, nb,
                                      nL, 1, NBLK, P)
                else:
                    for li in range(nL):
                        s = stg[:nb, li]
                        kv[li][:, idxs[c], ...] = s.permute(
                            1, 0, *range(2, s.dim()))
        torch.cuda.synchronize()
        t = time.time()
        for c in range(nchunk):
            nb = idxs[c].numel()
            if kernel:
                ext.licht_scatter(stg[:nb].contiguous(), idxs[c], lp, nb, nL, 1,
                                  NBLK, P)
            else:
                for li in range(nL):
                    s = stg[:nb, li]
                    kv[li][:, idxs[c], ...] = s.permute(1, 0, *range(2, s.dim()))
        torch.cuda.synchronize()
        dt = time.time() - t
        gb = total * MB / 1e3
        print(f"  {'KERNEL' if kernel else 'python'}: {total}blk {nchunk}chunk "
              f"{dt * 1e3:7.1f}ms ({gb / dt:6.1f} GB/s, "
              f"{nchunk * (1 if kernel else nL)} launches)")

    print("=== scatter speed (serving scale) ===")
    go(False)
    go(True)


print("=== correctness ===")
correctness("FA ", lambda N: torch.zeros((2, N, 16, 8, 128),
            dtype=torch.float16, device=dev), 32, 4000, 1, (16, 8, 128))
correctness("MLA", lambda N: torch.zeros((N, 2, 512), dtype=torch.float16,
            device=dev), 32, 4000, 0, (512,))
speed()
