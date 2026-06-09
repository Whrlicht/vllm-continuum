# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-round KV-cache store (CPU/SSD) for multi-turn agent reuse.

Goal
----
In PD-disaggregated multi-turn serving, the KV produced for round N
(prompt_N + output_N) is a prefix of round N+1's prompt
(prompt_{N+1} = prompt_N + output_N + tool_result).  Persisting that KV
when round N's decode finishes lets round N+1's *prefill* skip
recomputing the shared prefix: it loads the saved blocks straight into
the GPU paged buffer and only chunk-prefills the new tool_result.

Design (incremental + fully-async, see DESIGN discussion)
---------------------------------------------------------
* INCREMENTAL store.  We keep a job's KV across rounds (NO delete on
  load) and each round only stores the *new complete blocks* since the
  last store (round N+1 appends [last, end)).  Layout per job:
      {root}/{job}/inc_{start:09d}_{end:09d}.safetensors   # block range
      {root}/{job}/manifest.json   # {"total_blocks", "token_ids"}
  This cuts the per-round write/gather from the full (growing) sequence
  to just the delta — which is what makes the async path keep up.
* FULLY ASYNC store (B + C).  The decode engine thread only ENQUEUES a
  store task (block_ids + token_ids snapshot); a pool of background
  threads does the GPU→CPU gather (on its own CUDA stream, so it never
  serialises with the engine's forward) AND the disk write.  The engine
  forward never waits.  A request's GPU blocks are released as soon as
  its gather completes (drain_done), not after the write.
* Block granularity, indexed exactly like P2pNcclConnector
  (`layer[:, blocks]` FlashAttention, `layer[blocks]` MLA/FlashInfer).
* Keyed by ``job_id`` (the agent conversation).  Cross-process via the
  filesystem; point ``storage_path`` at ``/dev/shm/...`` (RAM/"CPU") or a
  disk mount ("SSD").

Cleanup: NO delete-on-load (incremental keeps the history).  A job dir is
removed only via `delete(job_id)` (trajectory end / eviction — wired
later).  Startup clears the root.

All entry points are best-effort and never raise.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

_MANIFEST = "manifest.json"


def align_blocks(num_tokens: int, block_size: int) -> int:
    """Number of COMPLETE blocks in `num_tokens` (floor).  Only complete
    blocks are reusable as a prefix (a partial last block changes as the
    next round keeps filling it)."""
    if num_tokens <= 0 or block_size <= 0:
        return 0
    return num_tokens // block_size


def _safe_job(job_id: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in str(job_id)]
    return "".join(keep)[:200]


class RoundKVStore:
    """Per-host incremental KV store with a background gather+write pool."""

    def __init__(self, storage_path: str, block_size: int):
        self.storage_path = storage_path
        self.block_size = max(int(block_size), 1)
        self._kv_caches: dict = {}            # layer_name -> paged GPU tensor
        self._device = None
        self._is_cuda = False
        try:
            os.makedirs(self.storage_path, exist_ok=True)
        except OSError as e:  # pragma: no cover
            logger.warning("RoundKVStore: cannot create %s: %s",
                           self.storage_path, e)

        # ---- async store pool (B + C) ----
        self._num_writers = int(os.environ.get("LICHT_ROUND_KV_WRITERS", "4"))
        # Bounded queue = high-water back-pressure.  Default high enough
        # that normal waves never block the engine; only sustained
        # overload makes enqueue wait (never drops, never OOMs).
        self._high_water = int(os.environ.get(
            "LICHT_ROUND_KV_HIGH_WATER", "256"))
        self._queue: "queue.Queue" = queue.Queue(maxsize=self._high_water)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        # request_ids whose gather is done -> their GPU blocks may free.
        self._done: set = set()
        self._done_lock = threading.Lock()
        # per-job last-stored COMPLETE block count (decode is the only
        # writer; authoritative in-memory, mirrored to manifest for the
        # prefill process to read).
        self._last_stored: dict = {}
        self._job_locks: dict = defaultdict(threading.Lock)
        self._job_locks_guard = threading.Lock()
        self._started = False
        # ---- parallel LOAD pool (prefill side) ----
        # The reuse gap ≈ the whole prior prefix, so a wave of admits would
        # otherwise read GBs serially on the engine thread.  These workers
        # do ONLY the CPU file reads (safe_open + get_slice → CPU tensors)
        # in parallel; the H2D scatter stays on the engine (single CUDA
        # thread), so we get aggregate RAM bandwidth without multi-thread
        # CUDA risk.
        self._load_workers = int(os.environ.get(
            "LICHT_ROUND_KV_LOAD_WORKERS", "8"))
        self._load_pool = None
        # H2D via a reusable PINNED double-buffer (async on the engine
        # stream).  cudaHostAlloc bypasses RLIMIT_MEMLOCK so large pinned is
        # fine; set LICHT_ROUND_KV_PINNED=0 to fall back to pageable copies.
        self._use_pinned = (
            os.environ.get("LICHT_ROUND_KV_PINNED", "1") == "1")
        self._pin_bufs: list = [None, None]
        self._pin_events: list = [None, None]
        self._pin_idx = 0
        # ---- increment coalescing ----
        # Super-multi-turn jobs accumulate one inc file per round; merge
        # them into a single file once the count exceeds this, off the
        # engine (in the bg store thread).  0 disables.
        self._coalesce_threshold = int(os.environ.get(
            "LICHT_ROUND_KV_COALESCE", "24"))
        # ---- layer-wise pipelined load (LICHT_ROUND_KV_PIPELINE=1) ----
        # Instead of reading+scattering ALL layers before the forward
        # (blocking), a background driver thread reads+scatters layer by
        # layer on its own copy stream; the forward syncs per layer via
        # wait_layer(), so the load of layer i+1 overlaps the compute of
        # layer i.  Off by default (synchronous load_batch) for easy A/B.
        self._pipeline = (
            os.environ.get("LICHT_ROUND_KV_PIPELINE", "0") == "1")
        # ★ consumer(decode)也 cudaHostRegister arena → 走直读 kernel(无 GPU
        # staging). 默认开. 不开则 consumer 的复用 load 走逐请求 staging, 把整段
        # 前缀一次性搬上 GPU, 长前缀(SWE-bench 等)在 gpu-mem-util~0.95 下 OOM.
        # 代价: decode 启动多付一次 cudaHostRegister(大 arena ~分钟级) + 双向 pin.
        # LICHT_ARENA_CONSUMER_DIRECT=0 退回旧 consumer-mmap-only(省启动 register).
        self._consumer_direct = (
            os.environ.get("LICHT_ARENA_CONSUMER_DIRECT", "1") == "1")
        self._load_stream = None
        self._pl_active = False
        self._pl_gen = 0
        # layer_name -> threading.Event: set once the driver has ENQUEUED
        # that layer's scatter on the copy stream.
        self._pl_issued: dict = {}
        # layer_name -> torch.cuda.Event: fires when that layer's scatter
        # COMPLETES on the copy stream (compute stream waits on it).
        self._pl_cuda_evt: dict = {}
        self._pl_driver = None
        # ---- diagnostic profiling (LICHT_ROUND_KV_PROFILE=1) ----
        # Adds a contention probe + per-segment (pin_copy/h2d/index) timing
        # with cuda syncs to load_batch.  The syncs serialise the path (so
        # the run is slower) but give exact attribution + the *achieved* H2D
        # bandwidth at load time -> proves/quantifies HBM contention.  Off
        # by default (no syncs, no overhead).
        self._profile = (
            os.environ.get("LICHT_ROUND_KV_PROFILE", "0") == "1")
        self._probe_pin = None
        self._probe_gpu = None
        # ---- batched scatter (default ON; LICHT_ROUND_KV_BATCHED=0 -> old) ----
        # The probe shows ~21 GB/s available but per-layer scatter only gets
        # ~0.85 GB/s: it's ~32*reqs tiny serial H2Ds.  Batch each request's
        # layers into ONE big H2D (reused pinned -> reused GPU staging) then
        # GPU-tensor-indexed scatter.  Chunked to bound staging memory
        # (LICHT_ROUND_KV_STAGE_MB) since gpu-mem-util is ~0.95.
        self._batched = (
            os.environ.get("LICHT_ROUND_KV_BATCHED", "1") == "1")
        self._stage_cap_bytes = int(os.environ.get(
            "LICHT_ROUND_KV_STAGE_MB", "512")) * 1024 * 1024
        self._stage_pins: list = [None, None]   # double-buffered pinned
        self._stage_events: list = [None, None]
        self._stage_idx = 0
        self._stage_cur = 0
        self._stage_gpu = None                  # single reused GPU staging
        # Run the scatter on a dedicated copy stream (not the default/forward
        # stream) so it OVERLAPS the forward via the copy engine instead of
        # serialising behind forward kernels.  The probe shows the copy engine
        # has bandwidth during serving, so this is the real serving win.
        self._load_stream_scatter = (
            os.environ.get("LICHT_ROUND_KV_LOAD_STREAM", "1") == "1")
        self._acc_pin_ms = 0.0   # CPU pin-fill time accumulated per load_batch
        # Pipelined load: read each request DIRECTLY into one pinned buffer on
        # the pool (no separate CPU pin-fill), and scatter each request as its
        # read COMPLETES (overlap file I/O with GPU H2D/scatter).  Default on.
        self._pipelined = (
            os.environ.get("LICHT_ROUND_KV_PIPELINED", "1") == "1")
        # RAW contiguous .bin chunks (default on): store KV layer-major
        # contiguous (no safetensors); load = mmap + ONE bulk H2D per chunk +
        # GPU-side sub-range scatter.  Kills the strided get_slice + cat read
        # bottleneck (read ~1.4 GB/s -> bulk H2D ~12-24 GB/s).
        # LICHT_ROUND_KV_RAW=0 -> old safetensors path.
        self._raw = (os.environ.get("LICHT_ROUND_KV_RAW", "1") == "1")
        self._raw_pin = None   # reused pinned staging for raw H2D
        # Deferred GPU-scatter timing: CUDA events recorded on the copy stream,
        # read on the NEXT load_batch call (events done by then -> NO sync,
        # NO engine stall — unlike the profile path).  Lets us separate
        # "engine blocked by read" from "GPU scatter (async, non-blocking)".
        self._pipe_ev = None
        # ---- ASYNC load (LICHT_ROUND_KV_ASYNC=1, default on) ----
        # The load runs on a background thread (NOT the engine), so the prefill
        # engine never blocks on it.  Completion is reported via drain_loaded()
        # -> connector get_finished recving -> scheduler moves the request from
        # WAITING_FOR_REMOTE_KVS to running.  This is what makes round-kv match
        # iteration-level scheduling (engine free during the load).
        self._load_q: "queue.Queue" = queue.Queue()
        self._load_done: set = set()
        self._load_done_lock = threading.Lock()
        self._async_load_started = False

        # ---- Phase 1: save-on-preempt sync infrastructure ----
        # When a decode request is preempted, the scheduler asks us to save
        # its KV increment to arena and waits for completion (so blocks are
        # safe to free).  We reuse the existing async store pool (enqueue
        # + bg gather + write_inc) but expose a sync wait API via a per-
        # request threading.Event signalled in _mark_done.
        self._preempt_lock = threading.Lock()
        self._preempt_events: dict = {}    # request_id -> threading.Event
        # ---- Phase 2: post-store completion hook ----
        # Installed via set_done_hook by p2p_nccl_connector on the
        # producer side so it can route ARENA_SINK completions onto
        # the fast-release queue.  None = no-op.  Fires from the
        # store-pool thread inside _mark_done.
        self._done_hook = None

        # ---- RESIDENT SHARED PINNED ARENA (LICHT_ROUND_KV_ARENA=1, default) --
        # The mainstream (LMCache/Mooncake) design: KV lives in a fixed,
        # pre-registered pinned region; a load is a DIRECT H2D from it — NO
        # per-load file read, NO mmap->pinned copy, NO per-load alloc, NO
        # cross-process page faults.  We back the region with ONE big /dev/shm
        # file shared by the decode (writer) and prefill (reader) processes and
        # cudaHostRegister it ONCE (prefill side, ~2GB/s -> a one-time startup
        # cost) so the shared physical pages become DMA-able at ~24GB/s.
        #   * slot = one block's KV, block-major [nL, 2, *rest] (2.097MB here).
        #   * RING (FIFO) bump allocator at INCREMENT granularity: each stored
        #     increment is a CONTIGUOUS run of slots (load = one bulk H2D); the
        #     ring wraps -> oldest increments are overwritten (FIFO≈LRU).
        #   * a slot is still valid iff its bump_base >= next_slot - num_slots;
        #     lookup() only claims valid prefixes so the connector never asks to
        #     load an evicted block (correctness under eviction).
        # Data store replaces the per-block .bin; the per-increment INDEX is a
        # tiny inc_*.slot file holding the bump_base.  LICHT_ROUND_KV_ARENA=0
        # falls back to the .bin (raw) path.
        self._arena = (os.environ.get("LICHT_ROUND_KV_ARENA", "1") == "1")
        self._arena_gb = float(os.environ.get(
            "LICHT_ROUND_KV_ARENA_GB", "24"))
        # DIAGNOSTIC (LICHT_ROUND_KV_SYNC_FIRST=1): drain the GPU before the
        # load and time it separately, so we can tell whether the load is slow
        # because it contends with the prior step's still-running forward
        # (drain large + post-drain load fast) or because the op itself is slow
        # on the real paged cache (drain ~0 + load still slow).
        self._sync_first = (
            os.environ.get("LICHT_ROUND_KV_SYNC_FIRST", "0") == "1")
        # ---- FUSED scatter (LICHT_ROUND_KV_FUSED=1) ----
        # Replace the per-chunk nL `index_put`s with ONE custom-CUDA-kernel
        # launch (see fused_scatter.py).  Cuts per-chunk dispatches nL->1 so the
        # busy serving process can keep the GPU fed (the per-layer loop starves
        # it).  Opt-in until serving-validated; falls back to Python if the
        # kernel won't compile or the layout is unsupported.
        self._fused = (os.environ.get("LICHT_ROUND_KV_FUSED", "0") == "1")
        self._fused_fn = None          # compiled licht_scatter callable
        self._layer_ptrs = None        # int64 [nL] data_ptr() of each KV layer
        self._fused_layers = None      # keep refs alive
        self._fused_P = 0              # prod(rest)
        self._fused_NBLK = 0           # blocks per layer
        self._arena_mm = None          # python mmap over the shm arena file
        self._arena_fd = None
        self._arena_addr = 0           # base host pointer (for register)
        self._arena_bytes = 0
        self._arena_view = None        # torch tensor [num_slots, nL, 2, *rest]
        self._slot_bytes = 0           # bytes per block (all layers)
        self._num_slots = 0
        self._arena_dtype = None
        self._arena_rest = None        # *rest dims of a layer block
        self._arena_dim = None         # 1 FlashAttention / 0 MLA
        self._arena_nL = 0
        self._arena_mapped = False     # mmap + slot layout derived
        self._arena_registered = False  # cudaHostRegister done (prefill)
        self._is_producer = True        # set in bind_kv_caches (role label)
        self._arena_lock = threading.Lock()   # guards the bump counter
        # ---- Phase A: arena eviction policy (job-aware + prefix-protect)
        # _slot_to_inc: physical slot_id (in [0, num_slots)) -> (job_id,
        #   inc_start_block, inc_end_block).  Updated on every successful
        #   _write_inc_arena; older mappings are overwritten when bump
        #   wraps and reuses the slot.
        # _job_to_slots: job_id -> set of slot_ids it currently occupies.
        #   Inverse of _slot_to_inc; lets delete(job_id) run in O(blocks)
        #   instead of O(num_slots).
        # _finished_jobs: job_ids the scheduler told us are done.  These
        #   jobs' slots are NOT protected (free to overwrite) and their
        #   manifests are deleted by mark_finished.
        # All three are guarded by _arena_lock.
        self._slot_to_inc: dict = {}
        self._job_to_slots: dict = {}
        self._finished_jobs: set = set()
        self._hdr_mm = None            # shm header: [0]=next_slot (int64 bump)
        self._hdr = None               # numpy int64 view over the header
        self._cudart = None            # cached ctypes libcudart
        # caches to kill the per-load `meta` cost (file I/O): .slot files are
        # write-once-immutable -> cache path->bump_base forever; the increment
        # list is cached per job and invalidated by the job dir's mtime (bumps
        # whenever decode adds/replaces an increment).
        self._slot_cache: dict = {}    # path -> bump_base (immutable)
        # job_id -> (monotonic_ts, [(s,e,path)]).  TTL-gated (C): decode writes
        # a new increment every round so dir mtime changes constantly and an
        # mtime cache never hits; a short TTL avoids re-listdir'ing for every
        # request in a wave.  Stale <= TTL only delays seeing a brand-new
        # increment (-> that sliver recomputes), never wrong data.
        self._inc_cache: dict = {}
        self._inc_ttl = float(os.environ.get("LICHT_ROUND_KV_INC_TTL", "1.0"))

        # ============================================================
        # Stage 2 LRU arena (env: LICHT_ROUND_KV_LRU=1)
        # ============================================================
        # 当开启时, arena 使用全新 slot-paged LRU 实现:
        #   - hdr 格式: mutex + bitmap + slot_state (~1MB, 含 Stage 6 预留)
        #   - 分配: bitmap first-fit
        #   - 淘汰: per-job LRU (manifest mtime) + tail-first
        #   - .slot 文件格式: per-block (slot_id, gen)
        #   - 跨进程同步: pthread_mutex_t PROCESS_SHARED + ROBUST
        #   - reader 保护: per-slot atomic pin
        #   - self-heal: evict 后 _last_stored 回退
        #
        # FIFO 代码路径不变, 不开 env 时默认仍走 FIFO bump 环.
        self._lru_enabled = (
            os.environ.get("LICHT_ROUND_KV_LRU", "0") == "1")
        self._lru_store = None    # LruArenaStore, 由 _arena_init 设置
        # 直读 kernel (arena host pinned -> paged, 无 GPU staging). prefill 端
        # arena cudaHostRegister 后可用; 由 _setup_arena_direct 设置.
        self._arena_direct_fn = None      # licht_scatter_from_arena callable
        self._arena_direct_layer_fn = None  # ★ per-layer 流水 callable
        self._arena_gather_layer_fn = None  # ★ per-layer 直写 callable (store)
        # store-direct: GPU kernel 直写 KV 到 arena (省 D2H gather + CPU memcpy).
        # 需 content_addr + 直写 kernel + arena registered. 默认开; 缺前提自动回退
        # 旧 gather+memcpy. LICHT_ROUND_KV_STORE_DIRECT=0 强制关.
        self._store_direct = (
            os.environ.get("LICHT_ROUND_KV_STORE_DIRECT", "1") == "1")
        self._arena_direct_layer_names = []  # 有序层名 (与 _arena_direct_layers 对齐)
        self._arena_direct_layer_ptrs = None  # int64 GPU [nL] paged data_ptr
        self._arena_direct_layers = []    # 各层 paged tensor (keep alive)
        self._arena_direct_P = 0          # prod(rest)
        self._arena_direct_NBLK = 0       # blocks per layer
        # LRU 直读 load 的 pipelined GPU timing (上一波 load 的 event, 下波读)
        self._lru_pipe_ev = None
        # ★ arena 逐层流水 load 状态: 每层 copy-stream scatter 的 CUDA event
        # (wait_layer 用) + 整波 pin handle (finish_pipelined 用).
        self._plr_events: dict = {}       # layer_name -> cuda.Event
        self._plr_handle = None           # BatchLoadHandle (unpin)
        self._plr_active = False
        self._plr_pipe_ev = None          # (e0, e1, gb) 上波逐层总耗时, 下波打
        # ★ 索引 GPU tensor 必须留活到 forward 后 (copy stream 跨 32 层引用它们):
        # 否则函数返回即被 allocator 标记可复用, main stream 分配抢占覆写 → scatter
        # 读乱 slot 越界. (批量版靠全局 wait_event 顺带保护; 流水版无全局 wait.)
        self._plr_src = None
        self._plr_dst = None
        # post-load gen 校验失败计数 (金丝雀). 理论上恒为 0: pin+gen 双原子机制
        # 保证 (1) pin 前被 evict -> gen 变 -> try_pin 失败 -> miss 重算;
        # (2) pin 后 -> can_evict 挡住 evict -> gen 不变. 所以 load 完 gen 必与
        # pin 时一致. 此计数 > 0 == pin/evict 不变量被破坏 (严重 bug), error 报警.
        self._lru_postload_fail_count = 0

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def bind_kv_caches(self, kv_caches: dict,
                       is_producer: bool = True) -> None:
        self._kv_caches = kv_caches or {}
        self._is_producer = bool(is_producer)
        for v in self._kv_caches.values():
            layer = v[0] if isinstance(v, (list, tuple)) else v
            try:
                self._device = layer.device
                self._is_cuda = (layer.device.type == "cuda")
            except Exception:
                pass
            break
        if self._arena and self._is_cuda:
            try:
                # Prefill (producer) LOADS -> needs cudaHostRegister for fast H2D
                # + the direct (no-staging) scatter kernel.  Decode (consumer)
                # 原本只 memcpy 写 shm 故跳过 register; 但 consumer 现在也做复用
                # LOAD, 不 register 就走逐请求 staging(整段前缀搬 GPU)→ 长前缀 OOM.
                # 故 _consumer_direct 默认让 consumer 也 register → 直读无 staging.
                register = bool(is_producer) or self._consumer_direct
                self._arena_init(register=register)
            except Exception as e:  # pragma: no cover
                logger.warning("round-kv ARENA init failed (%s); "
                               "falling back to .bin path", e)
                self._arena_mapped = False
                self._arena_registered = False
        self._maybe_start_hbm_probe()

    # ------------------------------------------------------------------
    # Resident shared pinned ARENA
    # ------------------------------------------------------------------
    def _arena_path(self) -> str:
        return os.path.join(self.storage_path, "_arena.bin")

    def _arena_hdr_path(self) -> str:
        return os.path.join(self.storage_path, "_arena.hdr")

    def _arena_meta_path(self) -> str:
        # worker 写 {num_slots, block_size}, scheduler 侧据此 lazy 开只读表
        return os.path.join(self.storage_path, "_arena_meta.json")

    def _ensure_lookup_store(self) -> None:
        """★ scheduler 侧 lazy 开一个"只读表"的 LruArenaStore (无 GPU 绑定).

        bind_kv_caches 只在 worker 侧调 → 只有 worker 的 _lru_store 被建; 但 lookup
        (get_num/update_state) 跑在 scheduler 侧实例上, _lru_store=None → lookup_resolve
        过去落到 self.lookup 读 .slot 文件 (~32ms). 这里从 worker 写的 meta 拿 num_slots,
        open_or_create 共享同一 shm hdr (只用来 ht_probe 查表), 让 lookup 走 C 表快路径.
        worker 侧 _lru_store 已存在, 直接返回不进这里.
        """
        if self._lru_store is not None or not self._lru_enabled:
            return
        try:
            with open(self._arena_meta_path()) as f:
                meta = json.load(f)
            from vllm.v1.core.sched.licht_v3.arena_lru_store import (
                LruArenaStore)
            self._lru_store = LruArenaStore.open_or_create(
                self.storage_path,
                num_slots=int(meta["num_slots"]),
                block_size=int(meta["block_size"]),
                wait_timeout_s=5.0)
            logger.info(
                "round-kv scheduler-side lookup store opened (table-only, "
                "num_slots=%d, content_addr=%s)",
                int(meta["num_slots"]), self._lru_store.content_addr)
        except Exception:
            # meta 还没写 (worker 未起完) / 打开失败 → 本次 lookup 退回文件路径,
            # 下次再试 (文件 open 失败很快, 不 log 避免刷屏).
            pass

    def _cuda_host_register(self, addr: int, size: int) -> int:
        import ctypes
        if self._cudart is None:
            rt = ctypes.CDLL("libcudart.so")
            rt.cudaHostRegister.restype = ctypes.c_int
            rt.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                            ctypes.c_uint]
            rt.cudaGetErrorString.restype = ctypes.c_char_p
            rt.cudaGetErrorString.argtypes = [ctypes.c_int]
            self._cudart = rt
        return self._cudart.cudaHostRegister(
            ctypes.c_void_p(addr), ctypes.c_size_t(size), ctypes.c_uint(0))

    def _cuda_err_str(self, rc: int) -> str:
        try:
            import ctypes
            s = self._cudart.cudaGetErrorString(ctypes.c_int(int(rc)))
            return s.decode() if s else "?"
        except Exception:
            return "?"

    def _arena_numa_interleave(self, addr: int, size: int) -> None:
        """mbind(MPOL_INTERLEAVE) 把 arena 页摊到所有 NUMA 节点, 提聚合内存带宽.
        默认关 (LICHT_ARENA_NUMA_INTERLEAVE=1 开): 拓扑相关, 单 GPU DMA 可能跨
        socket 反受 UPI 限制, 建议实测 A/B. 仅作用于 arena mmap, 不动模型权重.
        必须在 cudaHostRegister 前调 (注册 fault 页时才按此策略落节点)."""
        if os.environ.get("LICHT_ARENA_NUMA_INTERLEAVE", "0") != "1":
            return
        try:
            import ctypes
            nodes = sorted(
                int(d[4:]) for d in os.listdir("/sys/devices/system/node")
                if d.startswith("node") and d[4:].isdigit())
            if len(nodes) < 2:
                return                       # 单节点无可摊
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            SYS_mbind = 237                  # x86_64
            MPOL_INTERLEAVE = 3
            nwords = (max(nodes) // 64) + 1
            words = [0] * nwords
            for n in nodes:
                words[n // 64] |= (1 << (n % 64))
            NMask = ctypes.c_ulong * nwords
            nodemask = NMask(*words)
            maxnode = nwords * 64
            rc = libc.syscall(
                ctypes.c_long(SYS_mbind), ctypes.c_void_p(addr),
                ctypes.c_size_t(size), ctypes.c_int(MPOL_INTERLEAVE),
                nodemask, ctypes.c_ulong(maxnode), ctypes.c_uint(0))
            if rc != 0:
                logger.warning(
                    "round-kv ARENA mbind(INTERLEAVE) 失败 rc=%d errno=%d "
                    "-> 留单节点", rc, ctypes.get_errno())
            else:
                logger.info("round-kv ARENA NUMA interleave 跨节点 %s", nodes)
        except Exception as e:  # pragma: no cover
            logger.warning("round-kv ARENA NUMA interleave 异常: %s", e)

    def _arena_init(self, register: bool) -> None:
        """mmap the shared arena + header, derive the per-slot block-major
        layout from kv_caches, and (prefill only) cudaHostRegister the region
        so H2D loads run at ~24GB/s.  Idempotent across the two processes:
        both O_CREAT+ftruncate to the same size (no-op the second time).  Fresh
        ftruncate zeroes the header -> next_slot starts at 0 (startup clears
        /dev/shm so no stale counter)."""
        import ctypes
        import mmap as _mmap
        import numpy as np
        import torch
        if not self._kv_caches:
            return
        ln0 = next(iter(self._kv_caches))
        kv0 = self._kv_caches[ln0]
        layer0 = kv0[0] if isinstance(kv0, (list, tuple)) else kv0
        rest = tuple(int(x) for x in layer0.shape[2:])
        if layer0.shape[0] == 2:
            dim = 1                                  # FlashAttention
        elif layer0.shape[1] == 2:
            dim = 0                                  # MLA / FlashInfer
        else:
            logger.warning("round-kv ARENA: unsupported layout %s",
                           tuple(layer0.shape))
            return
        nL = len(self._kv_caches)
        elsize = layer0.element_size()
        prest = 1
        for x in rest:
            prest *= x
        self._slot_bytes = nL * 2 * prest * elsize   # one block, all layers
        req = int(self._arena_gb * (1024 ** 3))
        self._num_slots = max(req // self._slot_bytes, 1)
        self._arena_bytes = self._num_slots * self._slot_bytes
        os.makedirs(self.storage_path, exist_ok=True)
        # ★ 写 meta, 让 scheduler 侧 _ensure_lookup_store 能 lazy 开只读表 lookup.
        try:
            _mp = self._arena_meta_path()
            _fd, _tmp = tempfile.mkstemp(dir=self.storage_path,
                                         prefix=".meta_", suffix=".tmp")
            with os.fdopen(_fd, "w") as _f:
                json.dump({"num_slots": int(self._num_slots),
                           "block_size": int(self.block_size)}, _f)
            os.replace(_tmp, _mp)
        except Exception:
            pass
        # ---- arena data file ----
        fd = os.open(self._arena_path(), os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, self._arena_bytes)
        mm = _mmap.mmap(fd, self._arena_bytes, _mmap.MAP_SHARED,
                        _mmap.PROT_READ | _mmap.PROT_WRITE)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        # ★ P2 提带宽: NUMA interleave arena 跨节点, 摊内存控制器带宽 (arena 读/写
        # 不再挤单节点). 必须在 register 前设 (注册 fault 页时按此策略落节点). 默认
        # 关 (拓扑相关, 可能因 GPU 跨 socket DMA 反而略降); 用 mbind 只作用于 arena.
        self._arena_numa_interleave(addr, self._arena_bytes)
        if register:
            # ① 防创建竞态: 确保 arena.bin 已是全尺寸 (两端各自 ftruncate, 但若
            #    另一端正在创建, 这里再夹一次, 避免 register 的范围超出文件)。
            try:
                if os.fstat(fd).st_size < self._arena_bytes:
                    os.ftruncate(fd, self._arena_bytes)
            except OSError:
                pass
            _t = time.time()
            # ② 失败重试退避: decode 早到注册 256GB 偶发 rc=1 (瞬态: CUDA ctx/节点
            #    内存/文件就绪时序), 同 arena 稍后注册即成 (实测 prefill 晚 2min 成功)。
            #    退避重试跨过瞬态, 让两端都拿到直读 kernel。次数/退避可 env 调。
            _retries = int(os.environ.get("LICHT_ARENA_REG_RETRIES", "6"))
            _backoff = [3, 6, 12, 20, 30]
            rc = 1
            for _att in range(max(1, _retries)):
                rc = self._cuda_host_register(addr, self._arena_bytes)
                # 712 = cudaErrorHostMemoryAlreadyRegistered: 物理页已 pin (re-bind
                # 或同进程两 store) -> 区域可用, 视为成功.
                if rc in (0, 712):
                    break
                if _att < max(1, _retries) - 1:
                    _slp = _backoff[min(_att, len(_backoff) - 1)]
                    logger.warning(
                        "round-kv ARENA cudaHostRegister attempt %d/%d rc=%d (%s)"
                        " -> %ds 后重试", _att + 1, _retries, rc,
                        self._cuda_err_str(rc), _slp)
                    time.sleep(_slp)
            if rc in (0, 712):
                self._arena_registered = True
                logger.info(
                    "round-kv ARENA: registered %.1fGB in %.1fs (%.1f GB/s)",
                    self._arena_bytes / 1e9, time.time() - _t,
                    (self._arena_bytes / 1e9) / max(time.time() - _t, 1e-3))
            else:
                # ★ 注册失败 (如 consumer 跨进程双 pin 返回 rc=1): cudaHostRegister
                # 只是【直读 kernel】的优化, 失败【绝不能】废掉整个 arena/LRU!
                # 旧代码这里 close+return → decode 退回 .bin/FIFO, content_addr 关,
                # 不插哈希表 → prefill 查表全 miss → 跨轮复用归零 (回归). 现仅禁用
                # 直读: _arena_registered=False, 继续用未注册 mmap arena (staging
                # load), content_addr/LRU/复用全部正常.
                self._arena_registered = False
                logger.warning(
                    "round-kv ARENA cudaHostRegister 重试 %d 次仍失败 rc=%d (%s) -> "
                    "直读 kernel 禁用, 改用未注册 arena + staging load "
                    "(content_addr/复用不受影响)",
                    _retries, rc, self._cuda_err_str(rc))
        at = torch.frombuffer(mm, dtype=layer0.dtype)
        self._arena_view = at.view((self._num_slots, nL, 2) + rest)
        self._arena_mm = mm
        self._arena_fd = fd
        self._arena_addr = addr
        self._arena_dtype = layer0.dtype
        self._arena_rest = rest
        self._arena_dim = dim
        self._arena_nL = nL
        # ---- Stage 2 LRU arena: 跳过 FIFO bump hdr, 用 LruArenaStore ----
        if self._lru_enabled:
            try:
                from vllm.v1.core.sched.licht_v3.arena_lru_store import (
                    LruArenaStore)
                # 用 open_or_create: 跨进程 fcntl flock 互斥,
                # 谁先到谁创建+init bitmap, 后到的 mmap+sync.
                # prefill/decode 启动顺序无关.
                self._lru_store = LruArenaStore.open_or_create(
                    self.storage_path,
                    num_slots=self._num_slots,
                    block_size=self.block_size,
                    wait_timeout_s=60.0)
                self._lru_store.bind_data_writer(self._lru_data_writer)
                self._arena_mapped = True
                # 直读 kernel 需 arena 真的 cudaHostRegister 成功 (_arena_registered).
                # 按【实际注册结果】门控, 不按请求值 register: 注册失败 (rc!=0/712)
                # 时仍走 staging load, 但 LRU/content_addr/复用照常.
                if self._arena_registered:
                    self._setup_arena_direct()
                from vllm.v1.core.sched.licht_v3.arena_lru_store import (
                    _HAS_C_LOOKUP)
                logger.info(
                    "round-kv LRU arena bound (role=%s, registered=%s, "
                    "num_slots=%d, slot_bytes=%.2fMB, free=%d, direct_kernel=%s, "
                    "content_addr=%s, lookup=%s)",
                    "producer" if self._is_producer else "consumer",
                    self._arena_registered,
                    self._num_slots, self._slot_bytes / 1e6,
                    self._lru_store.free_count(),
                    self._arena_direct_fn is not None,
                    self._lru_store.content_addr,
                    ("C-fast" if (_HAS_C_LOOKUP
                                  and self._lru_store.content_addr)
                     else "PY/own-slot"))
                return
            except Exception as e:
                logger.warning(
                    "round-kv LRU bind failed (%s); falling back to FIFO", e)
                self._lru_store = None
                # 继续走下面 FIFO 路径
        # ---- shared header (bump counter, FIFO 路径) ----
        hfd = os.open(self._arena_hdr_path(), os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(hfd, 4096)
        hmm = _mmap.mmap(hfd, 4096, _mmap.MAP_SHARED,
                         _mmap.PROT_READ | _mmap.PROT_WRITE)
        self._hdr = np.frombuffer(hmm, dtype=np.int64)
        self._hdr_mm = hmm
        self._arena_mapped = True
        logger.info("round-kv ARENA mapped: %.1fGB, %d slots, slot=%.3fMB, "
                    "register=%s", self._arena_bytes / 1e9, self._num_slots,
                    self._slot_bytes / 1e6, register)
        if self._fused and register:
            self._setup_fused()

    # ==================================================================
    # Stage 2 LRU arena 路径 (when LICHT_ROUND_KV_LRU=1)
    # ==================================================================
    def _lru_data_writer(self, slot_id: int, block_idx: int,
                         source_obj) -> None:
        """传给 LruArenaStore.bind_data_writer 的钩子.

        source_obj: block-major torch tensor [n_blocks, nL, 2, *rest] (CPU)
                    在 _write_inc_arena_lru 里已 permute 好
        copy source_obj[block_idx] -> self._arena_view[slot_id]
        """
        # arena_view 是 CPU shm 张量, source_obj 是 CPU 张量, host->host memcpy
        self._arena_view[slot_id].copy_(source_obj[block_idx])

    def _write_inc_arena_lru(self, job_id, start, end, tensors,
                             token_ids=None) -> bool:
        """LRU 版 write_inc: 把 tensors 字典 permute 成 block-major,
        交给 LruArenaStore.write_inc 处理 (内部 alloc + evict + memcpy + publish).

        返回 True 成功 / False 失败 (让 _do_store 失败时不推进进度, 下轮重试).
        """
        import torch
        layers = []
        for ln in self._kv_caches:
            t = tensors.get(ln)
            if t is None:
                return False                     # incomplete -> skip (不推进)
            layers.append(t.contiguous())
        stk = torch.stack(layers)
        if stk.shape[1] == 2:                    # FA per-layer [2, nbc, *rest]
            perm = [2, 0, 1] + list(range(3, stk.dim()))
        else:                                    # MLA per-layer [nbc, 2, *rest]
            perm = [1, 0, 2] + list(range(3, stk.dim()))
        bm = stk.permute(*perm).contiguous()     # [nbc, nL, 2, *rest]

        # token_ids: 由 _do_store 传入完整序列 (len >= end*block_size).
        # ★ Stage 6 内容寻址必须有真实 token_ids 才能算链式 block hash 做 dedup;
        # 非 content-addr 路径只用它写 manifest. 缺省 [] 时 dedup 会失败回退.
        ok = self._lru_store.write_inc(
            job_id=str(job_id),
            start_block=int(start),
            end_block=int(end),
            token_ids=list(token_ids) if token_ids is not None else [],
            source_obj=bm)
        if not ok:
            logger.warning(
                "round-kv LRU write_inc failed (job=%s start=%d end=%d nbc=%d)",
                str(job_id)[:32], start, end, bm.shape[0])
        return bool(ok)

    def _lookup_lru(self, job_id: str,
                    cur_token_ids: list):
        """LRU 版 lookup: 调 LruArenaStore.lookup (content_addr 下它内部走哈希表
        lookup_resolve, 不读 manifest token_ids)."""
        return self._lru_store.lookup(str(job_id), cur_token_ids)

    def _load_request_arena_lru(self, job_id: str,
                                 dst_block_ids: list,
                                 src_block_offset: int,
                                 slot_gen: list = None) -> bool:
        """LRU 版 load_request: 用 LruArenaStore.load_request 拿 LoadHandle,
        然后 gather + H2D + per-layer scatter.

        slot_gen: ★ 跨 job 显式 (slot,gen) 列表 (lookup_resolve 解析的). 提供时
        走 load_pin_explicit (不查自己的 .slot); 否则 own-job 查 .slot.

        返回: True 成功, False miss/race
        """
        import torch
        if slot_gen:
            handle = self._lru_store.load_pin_explicit(
                list(slot_gen), list(dst_block_ids))
        else:
            handle = self._lru_store.load_request(
                str(job_id), list(dst_block_ids), int(src_block_offset))
        if handle is None:
            return False
        try:
            if not handle.slot_ids:
                return True
            # Gather: arena_view[slot_ids] -> CPU tensor [n, nL, 2, *rest]
            slot_ids_t = torch.tensor(handle.slot_ids, dtype=torch.long)
            src_cpu = self._arena_view[slot_ids_t]  # CPU shm fancy index
            # H2D
            src_gpu = src_cpu.to(self._device, non_blocking=True)
            # 目标 paged 索引
            dst_t = torch.tensor(handle.dst_block_ids,
                                  dtype=torch.long, device=self._device)
            # per-layer scatter
            layer_names = list(self._kv_caches.keys())
            for li, ln in enumerate(layer_names):
                kv = self._kv_caches[ln]
                layer = kv[0] if isinstance(kv, (list, tuple)) else kv
                srcl = src_gpu[:, li]            # [n, 2, *rest]
                if self._arena_dim == 1:         # FA
                    layer[:, dst_t, ...] = srcl.permute(
                        1, 0, *range(2, srcl.dim()))
                else:                            # MLA
                    layer[dst_t, ...] = srcl
            # post-load gen 校验 (金丝雀, 理论恒 True; 见 __init__ 注释).
            # 这条单请求 fallback 路径已 return False 让请求走 miss/重算 (安全),
            # 同时 error+计数 报警: 真失败 = pin/evict 不变量被破坏.
            if not handle.post_load_validate():
                self._lru_postload_fail_count += 1
                logger.error(
                    "round-kv LRU post-load gen MISMATCH (canary, total=%d): "
                    "pin/evict invariant BROKEN job=%s — returning miss "
                    "(recompute). Investigate evict bypassing can_evict.",
                    self._lru_postload_fail_count, str(job_id)[:32])
                return False
            return True
        finally:
            handle.release()

    def _setup_arena_direct(self) -> None:
        """准备直读 kernel: 拿 licht_scatter_from_arena callable + 预算 layer_ptrs
        (每层 paged tensor 的 data_ptr) + P/NBLK. 任一前提不满足 (dtype 非 2 字节 /
        P%8!=0 / kernel 未编译) 则 _arena_direct_fn=None, load 走 fallback."""
        import torch
        try:
            from vllm.v1.core.sched.licht_v3.fused_scatter import (
                get_arena_scatter)
            fn = get_arena_scatter()
            if fn is None:
                self._arena_direct_fn = None
                return
            layer_names = list(self._kv_caches.keys())
            layers = [(self._kv_caches[ln][0]
                       if isinstance(self._kv_caches[ln], (list, tuple))
                       else self._kv_caches[ln]) for ln in layer_names]
            P = 1
            for x in self._arena_rest:
                P *= int(x)
            if layers[0].element_size() != 2 or P % 8 != 0:
                logger.warning(
                    "round-kv DIRECT: unsupported (elem=%dB P=%d); LRU load "
                    "falls back to per-request path",
                    layers[0].element_size(), P)
                self._arena_direct_fn = None
                return
            # NBLK = 每层 paged tensor 的 block 维大小
            self._arena_direct_NBLK = int(
                layers[0].shape[1 if self._arena_dim == 1 else 0])
            self._arena_direct_P = int(P)
            self._arena_direct_layer_ptrs = torch.tensor(
                [int(l.data_ptr()) for l in layers], dtype=torch.int64,
                device=self._device)
            self._arena_direct_layers = layers   # keep refs alive
            self._arena_direct_layer_names = layer_names  # 有序, 与 layers 对齐
            self._arena_direct_fn = fn
            # ★ 流水线: per-layer scatter callable (旧 .so 没这符号则 None,
            # 流水回退批量直读). 与 attention 的 wait_for_layer_load 配合逐层重叠.
            try:
                from vllm.v1.core.sched.licht_v3.fused_scatter import (
                    get_arena_scatter_layer)
                self._arena_direct_layer_fn = get_arena_scatter_layer()
            except Exception:
                self._arena_direct_layer_fn = None
            # ★ store-direct: per-layer 直写 callable (GPU paged -> arena).
            try:
                from vllm.v1.core.sched.licht_v3.fused_scatter import (
                    get_arena_gather_layer)
                self._arena_gather_layer_fn = get_arena_gather_layer()
            except Exception:
                self._arena_gather_layer_fn = None
            logger.info(
                "round-kv DIRECT arena scatter ON: nL=%d P=%d NBLK=%d dim=%d "
                "arena_host=0x%x pipeline=%s store_direct=%s",
                len(layers), self._arena_direct_P, self._arena_direct_NBLK,
                self._arena_dim, self._arena_addr,
                self._arena_direct_layer_fn is not None,
                self._arena_gather_layer_fn is not None)
        except Exception as e:  # pragma: no cover
            logger.warning("round-kv DIRECT setup failed: %s; per-request "
                           "fallback", e)
            self._arena_direct_fn = None

    def _store_direct_available(self) -> bool:
        """store-direct 前提: 开关 + content_addr + 直写 kernel + arena registered
        + 直读 layer 元数据已就绪 (复用 load 的 _arena_direct_layers/NBLK/P)."""
        return bool(
            self._store_direct
            and self._is_cuda
            and self._lru_store is not None
            and getattr(self._lru_store, "content_addr", False)
            and self._arena_gather_layer_fn is not None
            and self._arena_registered
            and self._arena_direct_layers)

    def _gpu_write_miss(self, miss_slots: list, miss_paged: list,
                        ev, stream) -> None:
        """GPU 直写: 把 MISS 块从 paged KV 经 PCIe 直写进 arena slot (全 nL 层).
        wait_event(ev) 等本步 forward 把 KV 算完; 写完 sync, 保证 CS2 publish gen
        前 arena 数据已落地 (reader 看到 gen 即数据就绪, 维持 gen/pin 不变量)."""
        import torch
        if not miss_slots:
            return
        dst_slots = torch.tensor(miss_slots, dtype=torch.int64,
                                 device=self._device)
        src_idx = torch.tensor(miss_paged, dtype=torch.int64,
                               device=self._device)
        nb = len(miss_slots)
        sctx = (torch.cuda.stream(stream) if stream is not None
                else _nullctx())
        with sctx:
            if stream is not None and ev is not None:
                stream.wait_event(ev)
            for i in range(self._arena_nL):
                layer_ptr = int(self._arena_direct_layers[i].data_ptr())
                self._arena_gather_layer_fn(
                    int(self._arena_addr), dst_slots, src_idx, layer_ptr,
                    nb, self._arena_nL, i, self._arena_dim,
                    self._arena_direct_NBLK, self._arena_direct_P)
        # 必须同步: CS2 publish gen 前 arena 数据要已写完 (否则 reader pin 到半写)
        if stream is not None:
            stream.synchronize()
        else:
            torch.cuda.current_stream().synchronize()

    def _store_direct_arena_lru(self, job_id, start, end, inc_block_ids,
                                token_ids, ev, stream,
                                protected: bool = False,
                                protect_key=None) -> bool:
        """store-direct: 不 gather, 不 CPU memcpy. CS1 定 MISS slot → GPU 直写
        (paged->arena) → CS2 publish. dedup 记账复用 write_inc (gpu_write_fn 钩子).
        protected: ARENA_SINK/preempt 的"在途"KV, pin 住不被淘 (修2)."""
        def _gpu_write(miss_slots, miss_pos):
            miss_paged = [int(inc_block_ids[p]) for p in miss_pos]
            self._gpu_write_miss(miss_slots, miss_paged, ev, stream)
        return bool(self._lru_store.write_inc(
            job_id=str(job_id),
            start_block=int(start),
            end_block=int(end),
            token_ids=list(token_ids) if token_ids is not None else [],
            source_obj=None,
            gpu_write_fn=_gpu_write,
            protected=protected, protect_key=protect_key))

    def _load_batch_arena_lru(self, items: list) -> list:
        """LRU 版 batch load.

        优先走直读 kernel (arena host pinned -> paged, 无 GPU staging, 一次
        launch). 不可用时 fallback 到逐请求路径.
        """
        if not items:
            return []
        if self._arena_direct_fn is not None:
            return self._load_batch_arena_lru_direct(items)
        # fallback: 逐请求 (consumer 端 / kernel 不可用)
        results = []
        for item in items:
            job_id, dst_block_ids, src_block_offset = item[0], item[1], item[2]
            slot_gen = item[3] if len(item) > 3 else None
            try:
                ok = self._load_request_arena_lru(
                    job_id, dst_block_ids, src_block_offset, slot_gen)
            except Exception as e:  # pragma: no cover
                logger.warning("round-kv LRU load_request error job=%s: %s",
                               str(job_id)[:32], e)
                ok = False
            results.append(ok)
        return results

    def _load_batch_arena_lru_direct(self, items: list) -> list:
        """直读 kernel 路径: 一波 admit 请求一次性 resolve+pin -> 一次 H2D 索引
        -> 一次 kernel launch (arena host pinned 直接 PCIe 读 + 散写 paged) ->
        post-load gen 校验 -> unpin. 无 GPU staging buffer, 无 per-layer 循环."""
        import torch
        n_items = len(items)
        # 上一波的 GPU 计时 (event 此时已完成, 不 sync 不阻塞)
        if self._lru_pipe_ev is not None:
            try:
                _e0, _e1, _gb = self._lru_pipe_ev
                _gpu = _e0.elapsed_time(_e1)
                logger.info(
                    "round-kv LRU DIRECT gpu(prev): %.2fGB scatter_ms=%.1f "
                    "(%.1f GB/s, async)", _gb, _gpu,
                    (_gb / (_gpu / 1e3)) if _gpu else 0.0)
            except Exception:  # pragma: no cover
                pass
            self._lru_pipe_ev = None

        _t0 = time.time()
        # 1) 一次性 resolve + pin 整波
        bh = self._lru_store.load_batch_pin(items)
        results = list(bh.per_item_ok)
        nblk = len(bh.slot_ids)
        if nblk == 0:
            return results
        gb = nblk * self._slot_bytes / 1e9
        try:
            # 2) 索引 H2D (一次, 小)
            src_slots = torch.tensor(bh.slot_ids, dtype=torch.int64,
                                     device=self._device)
            dst_idx = torch.tensor(bh.dst_block_ids, dtype=torch.int64,
                                   device=self._device)
            # 3) 一次 kernel launch, 在 copy stream 上 (与 forward 尾巴并发)
            lstream = (self._get_load_stream()
                       if self._load_stream_scatter else None)
            sctx = (torch.cuda.stream(lstream) if lstream is not None
                    else _nullctx())
            e0 = e1 = None
            with sctx:
                if lstream is not None:
                    e0 = torch.cuda.Event(enable_timing=True)
                    e0.record(lstream)
                self._arena_direct_fn(
                    int(self._arena_addr), src_slots, dst_idx,
                    self._arena_direct_layer_ptrs,
                    nblk, self._arena_nL, self._arena_dim,
                    self._arena_direct_NBLK, self._arena_direct_P)
                if lstream is not None:
                    e1 = torch.cuda.Event(enable_timing=True)
                    e1.record(lstream)
            if lstream is not None:
                # forward (default stream) 等 copy stream 的 scatter 完成
                ev = torch.cuda.Event()
                ev.record(lstream)
                torch.cuda.current_stream().wait_event(ev)
                if e0 is not None and e1 is not None:
                    self._lru_pipe_ev = (e0, e1, gb)
            else:
                # 无独立 copy stream: 在当前 stream 上, 需要 sync 保证 KV 就位
                torch.cuda.current_stream().synchronize()

            # 4) post-load gen 校验 (金丝雀). 理论上恒 True (pin+gen 机制保证,
            #    见 __init__ 注释). 若失败说明 pin/evict 不变量被破坏 = 严重 bug,
            #    error 级报警 + 计数, 而非静默. 当前不触发 fallback 重算 (因为
            #    此分支理论不可达; 真触发了要先查为什么 pin 没挡住 evict).
            if not bh.post_load_validate():
                self._lru_postload_fail_count += 1
                logger.error(
                    "round-kv LRU DIRECT post-load gen MISMATCH "
                    "(canary, total=%d): pin/evict invariant BROKEN — a "
                    "pinned slot's gen changed during load. This should be "
                    "impossible; investigate evict bypassing can_evict. "
                    "reqs=%d blocks=%d",
                    self._lru_postload_fail_count, n_items, nblk)
        except Exception as e:  # pragma: no cover
            logger.warning("round-kv LRU DIRECT load failed: %s", e)
            results = [False] * n_items
        finally:
            bh.release()

        logger.info(
            "round-kv LOAD lru-direct: reqs=%d blocks=%d fail=%d GB=%.2f "
            "engine_block_ms=%.0f (host-pinned arena -> paged, no staging)",
            n_items, nblk, sum(1 for r in results if not r), gb,
            (time.time() - _t0) * 1000.0)
        return results

    def _delete_lru(self, job_id: str) -> None:
        self._lru_store.delete_job(str(job_id))

    def _mark_finished_lru(self, job_id: str) -> None:
        self._lru_store.mark_finished_job(str(job_id))

    # ==================================================================

    def _setup_fused(self) -> None:
        """Compile + wire the fused-scatter kernel (prefill side).  On any
        problem, disable fused and keep the Python per-layer scatter."""
        import torch
        try:
            from vllm.v1.core.sched.licht_v3.fused_scatter import get_scatter
            fn = get_scatter()
            if fn is None:
                self._fused = False
                return
            layer_names = list(self._kv_caches.keys())
            layers = [(self._kv_caches[ln][0]
                       if isinstance(self._kv_caches[ln], (list, tuple))
                       else self._kv_caches[ln]) for ln in layer_names]
            P = 1
            for x in self._arena_rest:
                P *= x
            if layers[0].element_size() != 2 or P % 8 != 0:
                logger.warning("round-kv FUSED: unsupported (elem=%dB P=%d); "
                               "using per-layer scatter",
                               layers[0].element_size(), P)
                self._fused = False
                return
            self._fused_NBLK = int(
                layers[0].shape[1 if self._arena_dim == 1 else 0])
            self._fused_P = int(P)
            self._layer_ptrs = torch.tensor(
                [int(l.data_ptr()) for l in layers], dtype=torch.int64,
                device=self._device)
            self._fused_layers = layers          # keep alive
            self._fused_fn = fn
            logger.info("round-kv FUSED scatter ON: nL=%d P=%d NBLK=%d dim=%d",
                        len(layers), self._fused_P, self._fused_NBLK,
                        self._arena_dim)
        except Exception as e:  # pragma: no cover
            logger.warning("round-kv FUSED setup failed: %s; per-layer", e)
            self._fused = False
            self._fused_fn = None

    def _arena_alloc(self, n: int):
        """Allocate n CONTIGUOUS slots from the ring; return the absolute
        bump_base, or None if n > num_slots.  If the run would wrap the ring
        end, skip the tail so every increment stays contiguous (load = one
        bulk H2D).

        Phase A: bump skips over PROTECTED slots (active job's head
        increment).  Worst-case the whole ring is protected -> we fall
        back to forcing an overwrite (warning) so the engine never
        deadlocks on a full arena."""
        if n <= 0 or n > self._num_slots:
            return None
        with self._arena_lock:
            base = int(self._hdr[0])
            # Safety budget: at most 2 ring traversals.  Bumping +1 per
            # protected slot in pathological cases would still terminate,
            # but we cap it to make the loop's worst-case bounded and
            # easy to reason about.
            budget = 2 * self._num_slots
            while budget > 0:
                off = base % self._num_slots
                # Stay contiguous: never cross the ring boundary.
                if off + n > self._num_slots:
                    skip = self._num_slots - off
                    base += skip
                    budget -= skip
                    continue
                # Scan [off, off+n) for the first protected slot.
                bad = -1
                for k in range(n):
                    if self._is_protected(off + k):
                        bad = k
                        break
                if bad < 0:
                    # Whole run is overwritable.
                    self._hdr[0] = base + n
                    return base
                # Skip past the protected slot and retry.
                base += bad + 1
                budget -= bad + 1
            # Pathological case: arena is fully covered by protected
            # head-increments.  Force an overwrite at the original
            # bump position and log a warning so the operator sees it.
            logger.warning(
                "round-kv ARENA fully protected (num_slots=%d, "
                "active_head_inc=%d) — forcing overwrite of HEAD "
                "increment.  Increase ROUND_KV_ARENA_GB.",
                self._num_slots, len(self._slot_to_inc))
            base = int(self._hdr[0])
            off = base % self._num_slots
            if off + n > self._num_slots:
                base += (self._num_slots - off)
            self._hdr[0] = base + n
            return base

    def _slot_index_path(self, job_id: str, start: int, end: int) -> str:
        return os.path.join(self._job_dir(job_id),
                            f"inc_{start:09d}_{end:09d}.slot")

    def _write_slot_index(self, job_id, start, end, bump_base) -> None:
        import numpy as np
        os.makedirs(self._job_dir(job_id), exist_ok=True)
        final = self._slot_index_path(job_id, start, end)
        fd, tmp = tempfile.mkstemp(dir=self._job_dir(job_id),
                                   prefix=".slot_", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(np.array([int(bump_base)], dtype=np.int64).tobytes())
            os.replace(tmp, final)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read_slot_index(self, path: str):
        import numpy as np
        try:
            with open(path, "rb") as f:
                b = f.read(8)
            if len(b) < 8:
                return None
            return int(np.frombuffer(b, dtype=np.int64)[0])
        except OSError:
            return None

    def _read_slot_index_cached(self, path: str):
        """Cached read of an inc_*.slot file.  The file is written once and
        never modified (each block range has a unique name), so caching the
        bump_base by path is always safe and kills the per-load read cost."""
        v = self._slot_cache.get(path)
        if v is not None:
            return v
        v = self._read_slot_index(path)
        if v is not None:
            self._slot_cache[path] = v
        return v

    # ------------------------------------------------------------------
    # HBM headroom probe (LICHT_HBM_PROBE=1) — diagnostic
    # ------------------------------------------------------------------
    def _maybe_start_hbm_probe(self) -> None:
        """Background dual-probe to measure, concurrently with the REAL
        forward, (A) pure-H2D bandwidth (copy engine only) and (B) H2D+scatter
        bandwidth (adds an SM kernel like the real load).  A<idle => DMA is
        starved; B<A => the SM scatter is starved by the forward.  Samples
        every ~30ms and logs p50/p10 each second (p10 catches the burst-time
        contention that 1s dmon averages away).  Off by default."""
        if (os.environ.get("LICHT_HBM_PROBE", "0") != "1"
                or not self._is_cuda
                or getattr(self, "_hbm_probe_started", False)):
            return
        self._hbm_probe_started = True
        t = threading.Thread(target=self._hbm_probe_loop, daemon=True,
                             name="HBMProbe")
        t.start()

    def _hbm_probe_loop(self) -> None:
        try:
            import torch
            torch.cuda.set_device(self._device)
            stream = torch.cuda.Stream()
        except Exception as e:  # pragma: no cover
            logger.warning("HBM probe init failed: %s", e)
            return
        interval = float(os.environ.get("LICHT_HBM_PROBE_INTERVAL", "0.03"))
        H, Dh, BS = 8, 128, 16
        nblk = 1024                         # ~134MB transfer per probe
        shp = (2, nblk, BS, H, Dh)
        try:
            pin = torch.empty(2 * nblk * BS * H * Dh, dtype=torch.float16,
                              pin_memory=True)
            gpu = torch.empty_like(pin, device=self._device)
            dst = torch.empty((2, nblk * 3, BS, H, Dh), dtype=torch.float16,
                              device=self._device)
            idx = torch.randperm(nblk * 3, device=self._device)[:nblk]
        except Exception as e:  # pragma: no cover
            logger.warning("HBM probe alloc failed: %s", e)
            return
        gb = (pin.numel() * 2) / 1e9
        a_bw, b_bw = [], []
        last = time.time()
        while not self._stop.is_set():
            try:
                # A: pure H2D (copy engine)
                ea0 = torch.cuda.Event(enable_timing=True)
                ea1 = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    ea0.record()
                    gpu.copy_(pin, non_blocking=True)
                    ea1.record()
                ea1.synchronize()
                a_bw.append(gb / (ea0.elapsed_time(ea1) / 1e3))
                # B: H2D + indexed scatter (adds an SM kernel)
                eb0 = torch.cuda.Event(enable_timing=True)
                eb1 = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    eb0.record()
                    gpu.copy_(pin, non_blocking=True)
                    dst[:, idx, ...] = gpu.view(shp)
                    eb1.record()
                eb1.synchronize()
                b_bw.append(gb / (eb0.elapsed_time(eb1) / 1e3))
            except Exception:  # pragma: no cover
                pass
            if time.time() - last >= 1.0 and a_bw:
                a = sorted(a_bw)
                b = sorted(b_bw) if b_bw else [0.0]
                p = lambda v, q: v[min(len(v) - 1, int(len(v) * q))]
                logger.info(
                    "round-kv HBM-PROBE: samples=%d | H2D(DMA) p50=%.1f "
                    "p10=%.1f GB/s | H2D+scatter p50=%.1f p10=%.1f GB/s",
                    len(a_bw), p(a, .5), p(a, .1), p(b, .5), p(b, .1))
                a_bw, b_bw, last = [], [], time.time()
            time.sleep(interval)

    @property
    def ready(self) -> bool:
        return bool(self._kv_caches)

    @property
    def pipeline_enabled(self) -> bool:
        return self._pipeline

    def _ensure_pool(self) -> None:
        if self._started:
            return
        self._started = True
        for i in range(max(self._num_writers, 1)):
            t = threading.Thread(target=self._store_loop, args=(i,),
                                 daemon=True, name=f"RoundKVStore-{i}")
            t.start()
            self._threads.append(t)

    def _job_lock(self, job_id: str):
        with self._job_locks_guard:
            return self._job_locks[job_id]

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _job_dir(self, job_id: str) -> str:
        return os.path.join(self.storage_path, _safe_job(job_id))

    def _manifest_path(self, job_id: str) -> str:
        return os.path.join(self._job_dir(job_id), _MANIFEST)

    def _inc_path(self, job_id: str, start: int, end: int) -> str:
        return os.path.join(self._job_dir(job_id),
                            f"inc_{start:09d}_{end:09d}.safetensors")

    def _inc_path_raw(self, job_id: str, start: int, end: int) -> str:
        return os.path.join(self._job_dir(job_id),
                            f"inc_{start:09d}_{end:09d}.bin")

    # ------------------------------------------------------------------
    # Block gather / scatter (mirror of P2pNcclConnector layouts)
    # ------------------------------------------------------------------

    @staticmethod
    def _gather_blocks(layer, block_ids):
        if layer.shape[1] == 2:               # MLA / FlashInfer
            return layer[block_ids, ...].detach().to("cpu")
        if layer.shape[0] == 2:               # FlashAttention
            return layer[:, block_ids, ...].detach().to("cpu")
        return None

    @staticmethod
    def _scatter_blocks(layer, block_ids, data) -> bool:
        src = data.to(layer.device, non_blocking=True)
        if layer.shape[1] == 2:               # MLA / FlashInfer
            layer[block_ids, ...] = src
            return True
        if layer.shape[0] == 2:               # FlashAttention
            layer[:, block_ids, ...] = src
            return True
        return False

    # ------------------------------------------------------------------
    # STORE — engine thread just enqueues (no GPU op, no wait)
    # ------------------------------------------------------------------

    def enqueue_store(self, job_id: str, full_block_ids: list,
                      full_token_ids: list,
                      request_id: Optional[str] = None,
                      protected: bool = False) -> None:
        """Queue an incremental store for `job_id`.  Captures a CUDA event
        on the engine stream so the background gather waits for the
        finishing forward's writes to be visible, then returns
        immediately.  The request's blocks must stay retained (delay-free)
        until `drain_done` reports `request_id` (= gather complete)."""
        if not self.ready or not full_block_ids:
            self._mark_done(request_id)
            return
        self._ensure_pool()
        ev = None
        if self._is_cuda:
            try:
                import torch
                ev = torch.cuda.Event()
                ev.record()       # capture engine (default) stream state
            except Exception:
                ev = None
        try:
            # block=True => high-water back-pressure (rare); never drops.
            self._queue.put(
                (job_id, list(full_block_ids), list(full_token_ids),
                 request_id, ev, protected),
                block=True, timeout=None)
        except Exception as e:  # pragma: no cover
            logger.warning("RoundKVStore.enqueue_store failed job=%s: %s",
                           job_id, e)
            self._mark_done(request_id)

    def release_protected(self, request_id) -> None:
        """修2: 请求 admit/结束 → 释放它 ARENA_SINK/preempt 时 pin 的保护, 让数据回到
        可被 LRU 淘的状态 (降 arena 容量压力). 超时兜底见 _expire_protected_pins."""
        s = self._lru_store
        if s is not None and hasattr(s, "release_protected"):
            try:
                s.release_protected(request_id)
            except Exception:  # pragma: no cover
                pass

    def _store_loop(self, idx: int) -> None:
        stream = None
        if self._is_cuda:
            try:
                import torch
                torch.cuda.set_device(self._device)
                stream = torch.cuda.Stream()
            except Exception:
                stream = None
        while not self._stop.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                break
            job_id, block_ids, token_ids, request_id, ev, protected = task
            try:
                self._do_store(job_id, block_ids, token_ids, request_id,
                               ev, stream, protected)
            except Exception as e:  # pragma: no cover
                logger.warning("RoundKVStore store failed job=%s: %s",
                               job_id, e)
                self._mark_done(request_id)

    def _do_store(self, job_id, block_ids, token_ids, request_id, ev,
                  stream, protected: bool = False) -> None:
        with self._job_lock(job_id):
            last = self._last_stored.get(job_id)
            if last is None:
                last = self._read_total_blocks(job_id)   # cross-restart
            end = align_blocks(len(token_ids), self.block_size)
            # ★ block_ids 可能短于累积 token 推出的块数 (多轮: token_ids 累积, 但
            # 本轮 block_ids 不含全部复用前缀块). 切片 block_ids[last:end] 会静默
            # 截断 → inc 实际块数 < end-last, 而 write_inc 按 end-start 当块数会越界
            # (index out of bounds). 用实际 block_ids 长度夹住 end, 保证 [last,end)
            # 与 gather 出的块数一致, manifest 也不会过度声明.
            end = min(end, len(block_ids))
            if end <= last:
                # No new COMPLETE block this round (e.g., output < 1 block).
                self._mark_done(request_id)
                return
            inc_block_ids = list(block_ids[last:end])
            # ---- store-direct: GPU kernel 直写 paged->arena, 省 gather+memcpy ----
            if self._store_direct_available():
                _t1 = time.time()
                ok = self._store_direct_arena_lru(
                    job_id, last, end, inc_block_ids, token_ids, ev, stream,
                    protected=protected, protect_key=request_id)
                # GPU 写已 sync 完成 -> paged 块可释放
                self._mark_done(request_id)
                if ok is False:
                    logger.warning(
                        "round-kv STORE-DIRECT inc write failed job=%s [%d,%d)"
                        " — 进度不推进", str(job_id)[:32], last, end)
                    return
                # manifest 已由 LruArenaStore.write_inc 写过同一文件同内容
                # (store-direct 必走 LRU), 不重复写 (省 O(token) JSON).
                self._last_stored[job_id] = end
                logger.info(
                    "round-kv STORE-DIRECT: job=%s inc_blocks=%d write_ms=%.0f",
                    str(job_id)[:32], end - last, (time.time() - _t1) * 1000.0)
                return
            # ---- gather the increment (own stream; never touches engine) ----
            _t0 = time.time()
            tensors = self._gather(inc_block_ids, ev, stream)
            gather_ms = (time.time() - _t0) * 1000.0
            # Gather done -> the GPU blocks are now safe to free.
            self._mark_done(request_id)
            if tensors is None:
                return
            # ---- write increment file + update manifest (off critical path) ----
            _t1 = time.time()
            ok = self._write_inc(job_id, last, end, tensors, token_ids)
            if ok is False:
                # 存失败 (块数不符已被上面夹住; 这里兜其余: alloc/evict 失败等).
                # 不推进 _last_stored / 不写 manifest → 下轮 (多轮同 job) 重试这段,
                # 避免 manifest 过度声明 (声称存了实际没存) + 永久丢复用.
                logger.warning(
                    "round-kv STORE inc write failed job=%s [%d,%d) — 进度不推进",
                    str(job_id)[:32], last, end)
                return
            # LRU arena 路径下 manifest 已由 LruArenaStore.write_inc 写过 (同一
            # 文件同内容), 不重复写; 仅非 LRU fallback (raw/safetensors) 才写.
            if not (self._arena and self._arena_mapped
                    and self._lru_store is not None):
                self._write_manifest(
                    job_id, end, token_ids[:end * self.block_size])
            write_ms = (time.time() - _t1) * 1000.0
            self._last_stored[job_id] = end
            logger.info(
                "round-kv STORE: job=%s inc_blocks=%d gather_ms=%.0f "
                "write_ms=%.0f", str(job_id)[:32], end - last,
                gather_ms, write_ms)
            # ---- coalesce many small increments into one (bg, off engine) ----
            # (safetensors-only; raw .bin / arena don't accumulate DATA files,
            # only tiny .slot indices, so no coalesce needed there)
            if (self._coalesce_threshold > 0 and not self._raw
                    and not (self._arena and self._arena_mapped)):
                try:
                    self._maybe_coalesce(job_id, end)
                except Exception as e:  # pragma: no cover
                    logger.debug("round-kv coalesce failed job=%s: %s",
                                 job_id, e)

    def _maybe_coalesce(self, job_id: str, end: int) -> None:
        """If a job has accrued > threshold increment files, merge them into
        a single inc_0_end file (read all + write one + delete olds).  Runs
        in the bg store thread (already holding the job lock), so it never
        touches the engine.  Bounds the file count for super-multi-turn so
        the prefill load opens few files."""
        incs = self._list_increments(job_id)
        if len(incs) <= self._coalesce_threshold:
            return
        _t0 = time.time()
        try:
            from safetensors import safe_open
            from safetensors.torch import save_file
        except Exception:
            return
        merged: dict = {}
        # Concatenate every layer's blocks across all increments in order.
        for (s, e, path) in incs:
            with safe_open(path, framework="pt", device="cpu") as f:
                for layer_name in f.keys():
                    t = f.get_tensor(layer_name)
                    merged.setdefault(layer_name, []).append(t)
        import torch
        cat_dim = None
        out = {}
        for layer_name, parts in merged.items():
            # block dim: 1 for FlashAttention [2,blk,...], 0 for MLA [blk,2,...]
            if parts[0].shape[1] == 2:
                out[layer_name] = torch.cat(parts, dim=0).contiguous()
            else:
                out[layer_name] = torch.cat(parts, dim=1).contiguous()
        new_path = self._inc_path(job_id, 0, end)
        fd, tmp = tempfile.mkstemp(dir=self._job_dir(job_id),
                                   prefix=".merge_", suffix=".tmp")
        os.close(fd)
        save_file(out, tmp)
        os.replace(tmp, new_path)         # atomic: merged file now present
        # Delete the old increments (all except the merged-in-place one).
        for (s, e, path) in incs:
            if path != new_path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        logger.info(
            "round-kv COALESCE: job=%s files=%d->1 end_blocks=%d ms=%.0f",
            str(job_id)[:32], len(incs), end, (time.time() - _t0) * 1000.0)

    def _gather(self, inc_block_ids, ev, stream):
        try:
            import torch
        except Exception:
            return None
        tensors = {}
        ctx = (torch.cuda.stream(stream)
               if (stream is not None) else _nullctx())
        if stream is not None and ev is not None:
            try:
                stream.wait_event(ev)
            except Exception:
                pass
        with ctx:
            for layer_name, kv in self._kv_caches.items():
                layer = kv[0] if isinstance(kv, (list, tuple)) else kv
                g = self._gather_blocks(layer, inc_block_ids)
                if g is None:
                    logger.warning(
                        "RoundKVStore: unsupported layout layer=%s shape=%s",
                        layer_name, tuple(layer.shape))
                    return None
                tensors[layer_name] = g.contiguous()
        if stream is not None:
            try:
                stream.synchronize()   # ensure D2H complete before freeing
            except Exception:
                pass
        return tensors

    def _write_inc(self, job_id, start, end, tensors, token_ids=None) -> None:
        if self._arena and self._arena_mapped:
            return self._write_inc_arena(job_id, start, end, tensors,
                                         token_ids)
        if self._raw:
            return self._write_inc_raw(job_id, start, end, tensors)
        from safetensors.torch import save_file
        os.makedirs(self._job_dir(job_id), exist_ok=True)
        final = self._inc_path(job_id, start, end)
        fd, tmp = tempfile.mkstemp(dir=self._job_dir(job_id),
                                   prefix=".inc_", suffix=".tmp")
        os.close(fd)
        try:
            save_file(tensors, tmp)
            os.replace(tmp, final)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _write_inc_arena(self, job_id, start, end, tensors,
                         token_ids=None) -> None:
        """Store the increment into the shared pinned ARENA (decode side).
        Build the block-major [nbc, nL, 2, *rest] (one slot per block), bump-
        allocate a contiguous slot run, memcpy into the shared pages, then
        write the tiny inc_*.slot index (bump_base) so prefill can find +
        validate it.  No data file, no per-load read on the prefill side."""
        # Stage 2 LRU dispatch
        if self._lru_store is not None:
            return self._write_inc_arena_lru(job_id, start, end, tensors,
                                             token_ids)
        import torch
        layers = []
        for ln in self._kv_caches:
            t = tensors.get(ln)
            if t is None:
                return                           # incomplete -> skip
            layers.append(t.contiguous())
        stk = torch.stack(layers)                # FA:[nL,2,nbc,*r] MLA:[nL,nbc,2,*r]
        if stk.shape[1] == 2:                    # FA per-layer [2, nbc, *rest]
            perm = [2, 0, 1] + list(range(3, stk.dim()))
        else:                                    # MLA per-layer [nbc, 2, *rest]
            perm = [1, 0, 2] + list(range(3, stk.dim()))
        bm = stk.permute(*perm).contiguous()     # [nbc, nL, 2, *rest]
        nbc = bm.shape[0]
        base = self._arena_alloc(nbc)
        if base is None:
            logger.warning("round-kv ARENA: increment %d blocks > arena "
                           "%d slots; not stored (job=%s)", nbc,
                           self._num_slots, str(job_id)[:32])
            return
        off = base % self._num_slots
        # memcpy CPU(gathered) -> shared pinned pages (one contiguous run).
        self._arena_view[off:off + nbc].copy_(bm)
        self._write_slot_index(job_id, start, end, base)
        # Phase A: register this run in the reverse index so future
        # _arena_alloc's see "slots [off, off+nbc) belong to (job_id,
        # start, end)" and decide whether to protect them.  Sweep out
        # the previous owners (if any) of the same physical slots so
        # _job_to_slots stays consistent.
        jid = str(job_id)
        with self._arena_lock:
            slots_set = self._job_to_slots.setdefault(jid, set())
            for k in range(nbc):
                sid = off + k
                prev = self._slot_to_inc.get(sid)
                if prev is not None:
                    prev_jid = prev[0]
                    if prev_jid != jid:
                        prev_slots = self._job_to_slots.get(prev_jid)
                        if prev_slots is not None:
                            prev_slots.discard(sid)
                            if not prev_slots:
                                self._job_to_slots.pop(prev_jid, None)
                self._slot_to_inc[sid] = (jid, int(start), int(end))
                slots_set.add(sid)

    def _write_inc_raw(self, job_id, start, end, tensors) -> None:
        """Write the increment as ONE raw contiguous .bin, BLOCK-MAJOR
        [nbc, nL, 2, *rest].  Block-major puts the block dim OUTERMOST, so a
        load of a block sub-range [a:b] is a contiguous slice -> the load can
        read/H2D ONLY the gap blocks (no strided get_slice, no over-H2D of the
        whole chunk for a tiny gap).  Shape derivable at load from kv_caches."""
        import torch
        os.makedirs(self._job_dir(job_id), exist_ok=True)
        final = self._inc_path_raw(job_id, start, end)
        layers = []
        for ln in self._kv_caches:
            t = tensors.get(ln)
            if t is None:
                return                       # incomplete -> skip
            layers.append(t.contiguous())
        stk = torch.stack(layers)            # FA:[nL,2,nbc,*r] MLA:[nL,nbc,2,*r]
        if stk.shape[1] == 2:                # FA per-layer [2, nbc, *rest]
            perm = [2, 0, 1] + list(range(3, stk.dim()))
        else:                                # MLA per-layer [nbc, 2, *rest]
            perm = [1, 0, 2] + list(range(3, stk.dim()))
        bm = stk.permute(*perm).contiguous()         # [nbc, nL, 2, *rest]
        flat = bm.reshape(-1).view(torch.uint8)
        fd, tmp = tempfile.mkstemp(dir=self._job_dir(job_id),
                                   prefix=".inc_", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(flat.numpy().tobytes())
            os.replace(tmp, final)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Manifest (token_ids + total_blocks) — read by prefill process
    # ------------------------------------------------------------------

    def _write_manifest(self, job_id, total_blocks, token_ids) -> None:
        path = self._manifest_path(job_id)
        d = self._job_dir(job_id)
        os.makedirs(d, exist_ok=True)
        payload = {"total_blocks": int(total_blocks),
                   "token_ids": [int(t) for t in token_ids]}
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".man_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read_manifest(self, job_id) -> Optional[dict]:
        try:
            with open(self._manifest_path(job_id), "r") as f:
                return json.load(f)
        except (FileNotFoundError, ValueError):
            return None
        except Exception:  # pragma: no cover
            return None

    def _read_total_blocks(self, job_id) -> int:
        m = self._read_manifest(job_id)
        return int(m["total_blocks"]) if m else 0

    # ------------------------------------------------------------------
    # Completion bookkeeping (drain_done -> free GPU blocks)
    # ------------------------------------------------------------------

    def set_done_hook(self, hook) -> None:
        """Phase 2: install a callback invoked at the END of _mark_done
        for any normal-store completion (not preempt-sync, which short-
        circuits earlier).  Used by p2p_nccl_connector to learn when an
        ARENA_SINK D2H has finished so it can push the req_id onto the
        producer-side fast-release queue.  Callback signature:
            hook(request_id: str) -> None
        Errors in the hook are swallowed (logged) so a buggy callback
        can't break the store pool."""
        self._done_hook = hook

    def _mark_done(self, request_id: Optional[str]) -> None:
        if request_id is None:
            return
        # Phase 1: if this is a preempt-save (sync caller is waiting), signal
        # the event but DO NOT route through _done -- the scheduler owns the
        # free in the preempt path, so adding to _done would cause delay-free
        # release to also fire.  Only normal-store completions feed _done.
        with self._preempt_lock:
            ev = self._preempt_events.get(request_id)
        if ev is not None:
            ev.set()
            return
        with self._done_lock:
            self._done.add(request_id)
        # Phase 2: notify the done-hook (if installed).  Runs in the
        # store-pool thread; keep handler cheap.
        hook = getattr(self, "_done_hook", None)
        if hook is not None:
            try:
                hook(request_id)
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "RoundKVStore done-hook raised for req=%s: %s",
                    request_id, e)

    def save_preempted_sync(self, job_id: str, block_ids: list,
                            token_ids: list, request_id: str,
                            timeout_s: float = 30.0) -> bool:
        """Phase 1: synchronously save a preempted decode request's KV
        increment to arena.  Reuses the background store pool (gather +
        write_inc_arena + advance _last_stored) and waits for completion
        via a per-request Event signalled in _mark_done.

        Returns True iff the increment was successfully persisted; False
        means the caller should fall back to the normal recompute path
        (KV not in arena -> integrity preserved)."""
        if not self.ready or not self._is_cuda:
            return False
        ev = threading.Event()
        with self._preempt_lock:
            self._preempt_events[request_id] = ev
        try:
            # enqueue_store will be picked up by _store_loop, which does the
            # D2H gather (own CUDA stream) + write_inc_arena and finally
            # calls _mark_done(request_id) -> ev.set().
            # ★ 修2: preempt-save 是"在途"KV (请求被抢占,等重新 admit 拉回),
            # protected=True → pin 住不被淘. 同 ARENA_SINK.
            self.enqueue_store(job_id, block_ids, token_ids, request_id,
                               protected=True)
            ok = ev.wait(timeout=timeout_s)
            return ok
        except Exception:  # pragma: no cover
            return False
        finally:
            with self._preempt_lock:
                self._preempt_events.pop(request_id, None)

    def drain_done(self) -> set:
        with self._done_lock:
            if not self._done:
                return set()
            d = self._done
            self._done = set()
            return d

    # ------------------------------------------------------------------
    # LOOKUP (prefill, scheduler thread, filesystem-only)
    # ------------------------------------------------------------------

    def lookup_resolve(self, job_id: str, cur_token_ids: list) -> Optional[tuple]:
        """★ Stage 6d 跨 job 表驱动 lookup.

        返回 (matched_tokens, matched_blocks, slot_gen_list) 或 None.
          - content_addr 开 (LRU): 走 LruArenaStore.lookup_resolve, job 无关,
            命中 own/cross-job; slot_gen_list 是匹配前缀逐块 (slot,gen).
          - content_addr 关 / 非 LRU: 退回 own-job lookup, slot_gen_list=None
            (调用方据此走原 own-.slot load 路径).
        """
        # scheduler 侧 lazy 开只读表, 否则 _lru_store=None 会落到文件路径 (~32ms).
        self._ensure_lookup_store()
        if (self._lru_store is not None
                and getattr(self._lru_store, "content_addr", False)):
            return self._lru_store.lookup_resolve(list(cur_token_ids))
        res = self.lookup(job_id, cur_token_ids)
        if res is None:
            return None
        mt, mb = res
        return mt, mb, None

    def lookup(self, job_id: str, cur_token_ids: list) -> Optional[tuple]:
        """Return (matched_tokens, matched_blocks) for the longest
        block-aligned prefix of `cur_token_ids` covered by the stored
        increments of `job_id`, else None.  Cheap: reads only the
        manifest JSON, not the KV blobs."""
        # ★ scheduler 侧 lazy 开只读表 (同 lookup_resolve), 否则 _lru_store=None 会
        # 落到读 manifest token_ids 的 fallback —— content_addr 下 token_ids 已不存
        # (空) → 永远 None → ARENA_SINK/recovery admit 全 miss → 退回 NCCL (回归).
        # ensure 后走 _lookup_lru → lru_store.lookup → content_addr 时哈希表 resolve.
        self._ensure_lookup_store()
        # Stage 2 LRU dispatch
        if self._lru_store is not None:
            return self._lookup_lru(job_id, cur_token_ids)
        m = self._read_manifest(job_id)
        if not m:
            return None
        stored = m.get("token_ids") or []
        if not stored:
            return None
        n = min(len(stored), len(cur_token_ids))
        lcp = 0
        for i in range(n):
            if stored[i] != cur_token_ids[i]:
                break
            lcp += 1
        matched_blocks = align_blocks(lcp, self.block_size)
        if matched_blocks <= 0:
            return None
        # cap by what is actually stored
        matched_blocks = min(matched_blocks, int(m.get("total_blocks", 0)))
        if matched_blocks <= 0:
            return None
        # ARENA: cap to the longest prefix whose increments are still resident
        # (not overwritten by the ring).  This keeps the connector from asking
        # to load an evicted block — it'll just recompute past valid_end.
        if self._arena and self._arena_mapped and self._hdr is not None:
            valid_end = self._arena_valid_prefix_blocks(job_id)
            matched_blocks = min(matched_blocks, valid_end)
            if matched_blocks <= 0:
                return None
        return matched_blocks * self.block_size, matched_blocks

    def _arena_valid_prefix_blocks(self, job_id: str) -> int:
        """Longest contiguous-from-0 block prefix of `job_id` whose backing
        slots are still valid in the ring (bump_base >= next_slot - num_slots).
        Reads only the tiny inc_*.slot files + the shared counter."""
        try:
            next_slot = int(self._hdr[0])
        except Exception:
            return 0
        oldest = next_slot - self._num_slots
        valid_end = 0
        for (s, e, path) in self._list_increments(job_id):
            if s != valid_end:
                break                            # gap in coverage
            if not path.endswith(".slot"):
                break                            # non-arena chunk
            base = self._read_slot_index_cached(path)
            if base is None or base < oldest:
                break                            # evicted / missing
            valid_end = e
        return valid_end

    # ------------------------------------------------------------------
    # LOAD (prefill, worker forward thread, before compute)
    # ------------------------------------------------------------------

    def _list_increments(self, job_id) -> list:
        """Return sorted [(start, end, path)] of a job's increment files.
        TTL-cached per job (C): decode bumps the dir mtime every round so an
        mtime cache never hits; within `_inc_ttl` seconds reuse the cached list
        (a load wave then does at most one listdir per job)."""
        now = time.monotonic()
        c = self._inc_cache.get(job_id)
        if c is not None and (now - c[0]) < self._inc_ttl:
            return c[1]
        d = self._job_dir(job_id)
        out = []
        try:
            names = os.listdir(d)
        except OSError:
            self._inc_cache[job_id] = (now, out)
            return out
        for name in names:
            if not name.startswith("inc_"):
                continue
            for ext in (".slot", ".bin", ".safetensors"):
                if name.endswith(ext):
                    try:
                        core = name[len("inc_"):-len(ext)]
                        s_str, e_str = core.split("_")
                        out.append((int(s_str), int(e_str),
                                    os.path.join(d, name)))
                    except Exception:
                        pass
                    break
        out.sort()
        self._inc_cache[job_id] = (now, out)
        return out

    def _read_for_load(self, job_id: str, dst_block_ids: list,
                       src_block_offset: int):
        """CPU-only: read the saved blocks
        [src_block_offset : src_block_offset+len(dst_block_ids)] for
        `job_id` from its increment files, and CONCATENATE per layer so each
        layer becomes ONE contiguous CPU tensor covering the whole gap
        (instead of one small piece per increment file).  Returns
        (dst_block_ids, {layer_name: cpu_tensor}) or None on miss/gap.

        Concatenating here (K increments -> 1 tensor/layer) is what lets the
        scatter do 1 batched H2D per layer instead of K×32 tiny ones.  No
        GPU ops -> safe to run on a load-pool thread in parallel."""
        n = len(dst_block_ids)
        if n <= 0:
            return None
        lo = int(src_block_offset)
        hi = lo + n
        if lo < 0:
            return None
        incs = self._list_increments(job_id)
        if not incs:
            return None
        try:
            import torch
            from safetensors import safe_open
        except Exception:
            return None
        parts: dict = {}                       # layer_name -> [pieces in order]
        block_dim: dict = {}                   # layer_name -> 0 (MLA) or 1 (FA)
        covered = lo
        for (s, e, path) in incs:
            if e <= lo or s >= hi:
                continue                       # no overlap with [lo,hi)
            a = max(s, lo)
            b = min(e, hi)
            if a > covered:
                return None                    # gap in coverage -> miss
            try:
                with safe_open(path, framework="pt", device="cpu") as f:
                    keys = set(f.keys())
                    for layer_name, kv in self._kv_caches.items():
                        if layer_name not in keys:
                            return None
                        layer = kv[0] if isinstance(kv, (list, tuple)) else kv
                        sl = f.get_slice(layer_name)
                        shp = sl.get_shape()
                        fl, fh = a - s, b - s      # file-local block range
                        if layer.shape[1] == 2:    # MLA / FlashInfer
                            if shp[0] < fh:
                                return None
                            piece = sl[fl:fh]
                            block_dim[layer_name] = 0
                        elif layer.shape[0] == 2:  # FlashAttention
                            if shp[1] < fh:
                                return None
                            piece = sl[:, fl:fh]
                            block_dim[layer_name] = 1
                        else:
                            return None
                        parts.setdefault(layer_name, []).append(piece)
            except Exception as e:  # pragma: no cover
                logger.warning("RoundKVStore read failed job=%s: %s",
                               job_id, e)
                return None
            covered = b
        if covered < hi:
            return None
        # Concatenate each layer's increment pieces into one contiguous tensor.
        per_layer: dict = {}
        for layer_name, plist in parts.items():
            d = block_dim[layer_name]
            per_layer[layer_name] = (plist[0] if len(plist) == 1
                                     else torch.cat(plist, dim=d)).contiguous()
        return list(dst_block_ids), per_layer

    def _to_device_pinned(self, cpu_t, device):
        """H2D copy through a reusable pinned double-buffer (async on the
        engine stream).  Falls back to a direct pageable copy if pinned
        allocation is unavailable.  The returned GPU tensor is valid on the
        current stream (the subsequent scatter is ordered after it)."""
        import torch
        if not self._use_pinned or not self._is_cuda:
            return cpu_t.to(device)
        n = cpu_t.numel()
        i = self._pin_idx
        self._pin_idx ^= 1
        try:
            buf = self._pin_bufs[i]
            if buf is None or buf.numel() < n:
                buf = torch.empty(n, dtype=cpu_t.dtype, pin_memory=True)
                self._pin_bufs[i] = buf
            # Wait for this buffer's previous H2D before overwriting it.
            ev = self._pin_events[i]
            if ev is not None:
                ev.synchronize()
            buf[:n].copy_(cpu_t.reshape(-1))
            g = buf[:n].to(device, non_blocking=True).reshape(cpu_t.shape)
            nev = torch.cuda.Event()
            nev.record()
            self._pin_events[i] = nev
            return g
        except Exception as e:  # pragma: no cover
            logger.warning("round-kv pinned H2D failed (%s); falling back to "
                           "pageable", e)
            self._use_pinned = False
            return cpu_t.to(device)

    def _scatter_request(self, dst_block_ids, per_layer) -> bool:
        """Batched H2D + scatter for one request: per layer, ONE H2D of the
        contiguous gap tensor (via pinned double-buffer) then ONE indexed
        write into the paged buffer.  32 ops/req instead of K×32."""
        for layer_name, cpu_t in per_layer.items():
            kv = self._kv_caches.get(layer_name)
            if kv is None:
                return False
            layer = kv[0] if isinstance(kv, (list, tuple)) else kv
            src = self._to_device_pinned(cpu_t, layer.device)
            if layer.shape[1] == 2:            # MLA / FlashInfer
                layer[dst_block_ids, ...] = src
            elif layer.shape[0] == 2:          # FlashAttention
                layer[:, dst_block_ids, ...] = src
            else:
                return False
        return True

    def _get_profile_pin(self, n: int, dtype):
        import torch
        b = getattr(self, "_prof_pin", None)
        if b is None or b.numel() < n or b.dtype != dtype:
            b = torch.empty(n, dtype=dtype, pin_memory=True)
            self._prof_pin = b
        return b

    def _profile_load(self, items: list) -> list:
        """DIAGNOSTIC: for each request, time read(CPU) / pin(CPU) / h2d(GPU) /
        scatter(GPU) in ISOLATION — each fenced with cuda.synchronize so no
        stage overlaps/hides another.  Logs 'round-kv PROF2' per big load with
        each stage's ms + GB/s, and CPU-total vs GPU-total.  This pinpoints
        whether the bottleneck is CPU data-prep (read+pin) or GPU (h2d+scatter).
        Slow (serial + syncs) -> diagnostic only."""
        import torch
        results = []
        for (job, dst, off) in items:
            torch.cuda.synchronize()
            t0 = time.time()
            res = self._read_for_load(job, dst, off)     # CPU: file -> pageable
            read_ms = (time.time() - t0) * 1000.0
            if res is None:
                results.append(False)
                continue
            d2, per_layer = res
            idx = None
            pin_ms = h2d_ms = sc_ms = 0.0
            nbytes = 0
            ok = True
            for ln, cpu_t in per_layer.items():
                kv = self._kv_caches.get(ln)
                if kv is None:
                    ok = False
                    break
                layer = kv[0] if isinstance(kv, (list, tuple)) else kv
                if idx is None:
                    idx = torch.as_tensor(d2, device=layer.device,
                                          dtype=torch.long)
                n = cpu_t.numel()
                nbytes += n * cpu_t.element_size()
                buf = self._get_profile_pin(n, cpu_t.dtype)
                tp = time.time()                          # CPU: pageable->pinned
                buf[:n].copy_(cpu_t.reshape(-1))
                pin_ms += (time.time() - tp) * 1000.0
                torch.cuda.synchronize()
                th = time.time()                          # GPU: pinned->device
                g = buf[:n].to(layer.device,
                               non_blocking=True).reshape(cpu_t.shape)
                torch.cuda.synchronize()
                h2d_ms += (time.time() - th) * 1000.0
                ts = time.time()                          # GPU: device->paged
                if layer.shape[0] == 2:
                    layer[:, idx, ...] = g
                elif layer.shape[1] == 2:
                    layer[idx, ...] = g
                torch.cuda.synchronize()
                sc_ms += (time.time() - ts) * 1000.0
            results.append(ok)
            gb = nbytes / 1e9
            if gb >= 0.0:        # log EVERY load (even tiny) for diagnosis
                f = lambda ms: (gb / (ms / 1e3)) if ms > 0 else 0.0
                logger.info(
                    "round-kv PROF2: bytes=%.1fGB | read=%.0fms(%.1fGB/s) "
                    "pin=%.0fms(%.1f) h2d=%.0fms(%.1f) scatter=%.0fms(%.1f) "
                    "| CPU(read+pin)=%.0fms  GPU(h2d+scatter)=%.0fms",
                    gb, read_ms, f(read_ms), pin_ms, f(pin_ms), h2d_ms,
                    f(h2d_ms), sc_ms, f(sc_ms), read_ms + pin_ms,
                    h2d_ms + sc_ms)
        return results

    def _probe_h2d_gbs(self, nbytes: int = 256 * 1024 * 1024) -> float:
        """Measure the H2D bandwidth available RIGHT NOW: a clean, contiguous
        pinned copy with sync.  A low value means the GPU/HBM is already
        saturated (by the forward / KV migration) at the moment a load
        fires -> contention, independent of the load code.  Profiling only."""
        try:
            import torch
            if not self._is_cuda:
                return 0.0
            n = nbytes // 2
            if self._probe_pin is None or self._probe_pin.numel() < n:
                self._probe_pin = torch.empty(n, dtype=torch.float16,
                                              pin_memory=True)
                self._probe_gpu = torch.empty(n, dtype=torch.float16,
                                              device=self._device)
            torch.cuda.synchronize()
            t = time.time()
            self._probe_gpu[:n].copy_(self._probe_pin[:n], non_blocking=True)
            torch.cuda.synchronize()
            dt = time.time() - t
            return nbytes / 1e9 / dt if dt > 0 else 0.0
        except Exception:
            return 0.0

    def _scatter_request_profiled(self, dst_block_ids, per_layer,
                                  acc: list) -> bool:
        """Profiling variant of _scatter_request: attributes pin_copy / h2d /
        index_write time (each fenced with cuda sync) into
        acc=[pin_ms, h2d_ms, idx_ms, bytes]."""
        import torch
        for layer_name, cpu_t in per_layer.items():
            kv = self._kv_caches.get(layer_name)
            if kv is None:
                return False
            layer = kv[0] if isinstance(kv, (list, tuple)) else kv
            nbytes = cpu_t.numel() * cpu_t.element_size()
            t0 = time.time()
            pin = cpu_t if cpu_t.is_pinned() else cpu_t.pin_memory()
            acc[0] += (time.time() - t0) * 1000.0
            torch.cuda.synchronize()
            t1 = time.time()
            src = pin.to(layer.device, non_blocking=True)
            torch.cuda.synchronize()
            t2 = time.time()
            acc[1] += (t2 - t1) * 1000.0
            if layer.shape[1] == 2:
                layer[dst_block_ids, ...] = src
            elif layer.shape[0] == 2:
                layer[:, dst_block_ids, ...] = src
            else:
                return False
            torch.cuda.synchronize()
            acc[2] += (time.time() - t2) * 1000.0
            acc[3] += nbytes
        return True

    def _get_stage_gpu(self, n: int, dtype, device):
        import torch
        if (self._stage_gpu is None or self._stage_gpu.numel() < n
                or self._stage_gpu.dtype != dtype
                or self._stage_gpu.device != device):
            self._stage_gpu = torch.empty(n, dtype=dtype, device=device)
        return self._stage_gpu

    def _next_stage_pin(self, n: int, dtype):
        """Double-buffered pinned staging: alternate buffers so the CPU fill
        of the next chunk overlaps the H2D of the current one.  Waits the
        buffer's previous H2D event before overwriting it."""
        import torch
        i = self._stage_idx
        self._stage_idx ^= 1
        self._stage_cur = i
        ev = self._stage_events[i]
        if ev is not None:
            ev.synchronize()
        buf = self._stage_pins[i]
        if buf is None or buf.numel() < n or buf.dtype != dtype:
            buf = torch.empty(n, dtype=dtype, pin_memory=True)
            self._stage_pins[i] = buf
        return buf[:n]

    def _stage_pin_record(self):
        import torch
        try:
            ev = torch.cuda.Event()
            ev.record()
            self._stage_events[self._stage_cur] = ev
        except Exception:
            pass

    def _scatter_request_batched(self, dst_block_ids, per_layer) -> bool:
        """Batched H2D + GPU-side scatter for one request.  Instead of 32
        small per-layer H2Ds, copy a chunk of layers into ONE contiguous
        pinned buffer, do ONE big H2D into a reused GPU staging buffer, then
        scatter each layer GPU->GPU with a GPU-tensor index.  Chunked by
        _stage_cap_bytes to bound memory.  This turns ~32*reqs tiny serial
        H2Ds (≈0.85 GB/s) into a handful of big ones (≈probe ~20 GB/s)."""
        import torch
        layers = list(per_layer.items())
        if not layers:
            return False
        kv0 = self._kv_caches.get(layers[0][0])
        if kv0 is None:
            return False
        dev = (kv0[0] if isinstance(kv0, (list, tuple)) else kv0).device
        dtype = layers[0][1].dtype
        idx = torch.as_tensor(dst_block_ids, device=dev, dtype=torch.long)
        cap = max(self._stage_cap_bytes // max(layers[0][1].element_size(), 1),
                  1)
        j = 0
        while j < len(layers):
            # accumulate a chunk of whole layers up to the staging cap
            chunk = []
            nel = 0
            while j < len(layers):
                tn = layers[j][1].numel()
                if chunk and nel + tn > cap:
                    break
                chunk.append(layers[j])
                nel += tn
                j += 1
            pin = self._next_stage_pin(nel, dtype)
            off = 0
            plan = []
            for name, t in chunk:
                tn = t.numel()
                _tp = time.time()
                pin[off:off + tn].copy_(t.reshape(-1))
                self._acc_pin_ms += (time.time() - _tp) * 1000.0
                plan.append((name, off, tn, t.shape))
                off += tn
            gpu = self._get_stage_gpu(nel, dtype, dev)
            gpu[:nel].copy_(pin[:nel], non_blocking=True)   # ONE big H2D
            self._stage_pin_record()
            for name, o, tn, shp in plan:
                kv = self._kv_caches.get(name)
                if kv is None:
                    return False
                layer = kv[0] if isinstance(kv, (list, tuple)) else kv
                src = gpu[o:o + tn].view(shp)
                if layer.shape[1] == 2:            # MLA / FlashInfer
                    layer[idx, ...] = src
                elif layer.shape[0] == 2:          # FlashAttention
                    layer[:, idx, ...] = src
                else:
                    return False
        return True

    def _load_batch_pipelined(self, items: list) -> list:
        """Overlap reads with scatters.  Pool threads do PURE CPU file reads
        (pageable, NO CUDA -> no cudaHostAlloc driver-lock contention); the
        engine scatters each request as soon as its read completes, on the
        copy stream (reusable pinned double-buffer, no per-request alloc).
        read(i+1) thus overlaps scatter(i)."""
        import torch
        from concurrent.futures import as_completed
        # Deferred read of the PREVIOUS load's GPU-scatter time (events are
        # complete by now -> no sync, no stall).  This is the GPU work that
        # ran ASYNC on the copy stream and did NOT block the engine.
        if self._pipe_ev is not None:
            try:
                _e0, _e1, _gbp = self._pipe_ev
                _gpu = _e0.elapsed_time(_e1)
                logger.info(
                    "round-kv LOAD gpu(prev): %.1fGB scatter_gpu_ms=%.0f "
                    "(%.1f GB/s, async — does NOT block engine)", _gbp, _gpu,
                    (_gbp / (_gpu / 1e3)) if _gpu else 0.0)
            except Exception:  # pragma: no cover
                pass
            self._pipe_ev = None
        pool = self._ensure_load_pool()
        results = [False] * len(items)
        nblk = sum(len(d) for (_, d, _) in items)
        gb = nblk * 2.097152 / 1e3
        _t0 = time.time()
        futs = {pool.submit(self._read_for_load, j, d, o): k
                for k, (j, d, o) in enumerate(items)}
        lstream = (self._get_load_stream() if self._load_stream_scatter
                   else None)
        sctx = (torch.cuda.stream(lstream) if lstream is not None
                else _nullctx())
        e0 = e1 = None
        with sctx:
            if lstream is not None:
                e0 = torch.cuda.Event(enable_timing=True)
                e0.record(lstream)
            for fut in as_completed(futs):
                k = futs[fut]
                try:
                    res = fut.result()
                except Exception:  # pragma: no cover
                    res = None
                if res is None:
                    continue
                dst, per_layer = res
                try:
                    results[k] = self._scatter_request_batched(dst, per_layer)
                except Exception:  # pragma: no cover
                    results[k] = False
            if lstream is not None:
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record(lstream)
        if lstream is not None:
            ev = torch.cuda.Event()
            ev.record(lstream)
            torch.cuda.current_stream().wait_event(ev)
            if e0 is not None and e1 is not None:
                self._pipe_ev = (e0, e1, gb)   # read next call (no sync)
        # engine_block_ms = wall time the ENGINE was stuck here.  Scatter is
        # enqueued async on the copy stream, so this ≈ the READ wait (the real,
        # undistorted engine stall — no sync added, unlike the profile path).
        logger.info(
            "round-kv LOAD pipe: reqs=%d blocks=%d fail=%d GB=%.1f "
            "engine_block_ms=%.0f (≈read; blocks engine)",
            len(items), nblk, sum(1 for r in results if not r), gb,
            (time.time() - _t0) * 1000.0)
        return results

    # ------------------------------------------------------------------
    # RAW load — contiguous .bin chunks, bulk H2D, no strided read / no cat
    # ------------------------------------------------------------------

    def _get_raw_pin(self, n: int, dtype):
        import torch
        b = self._raw_pin
        if b is None or b.numel() < n or b.dtype != dtype:
            b = torch.empty(n, dtype=dtype, pin_memory=True)
            self._raw_pin = b
        return b

    def _load_request_raw(self, job_id: str, dst_block_ids: list,
                          src_block_offset: int) -> bool:
        """Load a request's gap from raw BLOCK-MAJOR .bin chunks.  The chunk is
        [nbc, nL, 2, *rest] so the gap [a:b] is a contiguous slice of dim 0 ->
        we mmap + copy + H2D ONLY the gap blocks (no over-H2D of the whole
        chunk for a tiny gap, no strided read).  Then per layer a GPU permute
        (block-major -> [2, blk]) + scatter.  Runs inside the copy-stream."""
        import numpy as np
        import torch
        n = len(dst_block_ids)
        if n <= 0:
            return False
        lo = int(src_block_offset)
        hi = lo + n
        incs = self._list_increments(job_id)
        if not incs:
            return False
        layer_names = list(self._kv_caches.keys())
        if not layer_names:
            return False
        kv0 = self._kv_caches[layer_names[0]]
        layer0 = kv0[0] if isinstance(kv0, (list, tuple)) else kv0
        rest = tuple(layer0.shape[2:])
        if layer0.shape[0] == 2:
            dim = 1                           # FlashAttention
        elif layer0.shape[1] == 2:
            dim = 0                           # MLA / FlashInfer
        else:
            return False
        dtype = layer0.dtype
        dev = layer0.device
        nL = len(layer_names)
        covered = lo
        for (s, e, path) in incs:
            if not path.endswith(".bin"):
                return False                 # raw mode but found non-raw chunk
            if e <= lo or s >= hi:
                continue
            if s > covered:
                return False                 # gap in coverage
            a = max(s, lo)
            b = min(e, hi)
            nbc = e - s
            try:
                arr = np.memmap(path, dtype=np.uint8, mode="r")
                t = torch.from_numpy(arr).view(dtype).view((nbc, nL, 2) + rest)
                sub = t[a - s:b - s]          # [b-a, nL, 2, *rest] CONTIGUOUS
                flat = sub.reshape(-1)        # gap-only (no over-H2D)
                fn = flat.numel()
                # mmap-gap -> double-buffered PINNED -> ASYNC H2D on the copy
                # stream (FIFO-ordered with the scatters; event-guarded pinned
                # reuse).  Only the GAP blocks are moved.
                pin = self._next_stage_pin(fn, dtype)
                pin.copy_(flat)
                gstage = self._get_stage_gpu(fn, dtype, dev)
                gstage[:fn].copy_(pin, non_blocking=True)
                self._stage_pin_record()
                gpu = gstage[:fn].view((b - a, nL, 2) + rest)
                del arr, t, sub, flat
            except Exception as ex:  # pragma: no cover
                logger.warning("round-kv raw load failed job=%s %s: %s",
                               job_id, path, ex)
                return False
            idx = torch.as_tensor(dst_block_ids[a - lo:b - lo], device=dev,
                                  dtype=torch.long)
            for i, ln in enumerate(layer_names):
                kv = self._kv_caches[ln]
                layer = kv[0] if isinstance(kv, (list, tuple)) else kv
                src = gpu[:, i]               # [b-a, 2, *rest]
                if dim == 1:
                    # FA: kv[L][:, idx] wants [2, b-a, *rest] -> permute
                    layer[:, idx, ...] = src.permute(
                        1, 0, *range(2, src.dim()))
                else:
                    # MLA: kv[L][idx] wants [b-a, 2, *rest] == src
                    layer[idx, ...] = src
            covered = b
        return covered >= hi

    def _load_batch_raw(self, items: list) -> list:
        """RAW load for an admit wave: per request, mmap chunks + bulk H2D +
        scatter on the copy stream.  No CPU read pool needed (the strided
        get_slice + cat is gone)."""
        import torch
        results = [False] * len(items)
        if self._pipe_ev is not None:
            try:
                _e0, _e1, _gbp = self._pipe_ev
                _gpu = _e0.elapsed_time(_e1)
                logger.info(
                    "round-kv LOAD gpu(prev): %.1fGB h2d+scatter_ms=%.0f "
                    "(%.1f GB/s)", _gbp, _gpu,
                    (_gbp / (_gpu / 1e3)) if _gpu else 0.0)
            except Exception:  # pragma: no cover
                pass
            self._pipe_ev = None
        nblk = sum(len(d) for (_, d, _) in items)
        gb = nblk * 2.097152 / 1e3
        _t0 = time.time()
        lstream = (self._get_load_stream() if self._load_stream_scatter
                   else None)
        sctx = (torch.cuda.stream(lstream) if lstream is not None
                else _nullctx())
        e0 = e1 = None
        with sctx:
            if lstream is not None:
                e0 = torch.cuda.Event(enable_timing=True)
                e0.record(lstream)
            for k, (j, d, o) in enumerate(items):
                try:
                    results[k] = self._load_request_raw(j, d, o)
                except Exception:  # pragma: no cover
                    results[k] = False
            if lstream is not None:
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record(lstream)
        if lstream is not None:
            ev = torch.cuda.Event()
            ev.record(lstream)
            torch.cuda.current_stream().wait_event(ev)
            if e0 is not None and e1 is not None:
                self._pipe_ev = (e0, e1, gb)
        logger.info(
            "round-kv LOAD raw: reqs=%d blocks=%d fail=%d GB=%.1f "
            "engine_block_ms=%.0f (mmap+H2D+scatter)",
            len(items), nblk, sum(1 for r in results if not r), gb,
            (time.time() - _t0) * 1000.0)
        return results

    def _load_request_arena(self, job_id: str, dst_block_ids: list,
                            src_block_offset: int, acc=None) -> bool:
        """Load a request's gap DIRECTLY from the resident pinned ARENA.  The
        increment's blocks live in a contiguous slot run [base_off, base_off+n)
        (block-major [n, nL, 2, *rest]); the gap [a:b] is a contiguous slice ->
        ONE async H2D straight from the registered shared pages (no mmap, no
        CPU pinned copy, no per-load alloc).  Then per-layer GPU permute +
        scatter, exactly like the .bin path.  Runs inside the copy stream.
        `acc` (optional [meta,idx,h2d,scat] float list) accumulates per-phase
        CPU-side time (NO cuda sync added) to pin down per-request overhead."""
        # Stage 2 LRU dispatch
        if self._lru_store is not None:
            return self._load_request_arena_lru(
                job_id, dst_block_ids, src_block_offset)
        import torch
        n = len(dst_block_ids)
        if n <= 0:
            return False
        lo = int(src_block_offset)
        hi = lo + n
        _tm = time.time() if acc is not None else 0.0
        incs = self._list_increments(job_id)
        if not incs:
            return False
        layer_names = list(self._kv_caches.keys())
        if not layer_names:
            return False
        dim = self._arena_dim
        rest = self._arena_rest
        nL = self._arena_nL
        dtype = self._arena_dtype
        kv0 = self._kv_caches[layer_names[0]]
        dev = (kv0[0] if isinstance(kv0, (list, tuple)) else kv0).device
        next_slot = int(self._hdr[0])
        oldest = next_slot - self._num_slots
        covered = lo
        for (s, e, path) in incs:
            if not path.endswith(".slot"):
                return False                     # arena mode, non-arena chunk
            if e <= lo or s >= hi:
                continue
            if s > covered:
                return False                     # gap in coverage
            base = self._read_slot_index_cached(path)
            if base is None or base < oldest:
                return False                     # evicted by the ring / missing
            a = max(s, lo)
            b = min(e, hi)
            cnt = b - a
            base_off = base % self._num_slots
            slot_a = base_off + (a - s)          # contiguous (alloc skips wrap)
            if acc is not None:
                acc[0] += time.time() - _tm      # meta: listdir + read .slot
            try:
                src = self._arena_view[slot_a:slot_a + cnt]   # [cnt,nL,2,*rest]
                fn = src.numel()
                gstage = self._get_stage_gpu(fn, dtype, dev)
                _th = time.time() if acc is not None else 0.0
                # DIRECT async H2D from the registered (pinned) arena.
                gstage[:fn].copy_(src.reshape(-1), non_blocking=True)
                gpu = gstage[:fn].view((cnt, nL, 2) + rest)
                if acc is not None:
                    acc[2] += time.time() - _th   # h2d enqueue (CPU-side)
            except Exception as ex:  # pragma: no cover
                logger.warning("round-kv arena load failed job=%s: %s",
                               job_id, ex)
                return False
            _ti = time.time() if acc is not None else 0.0
            idx = torch.as_tensor(dst_block_ids[a - lo:b - lo], device=dev,
                                  dtype=torch.long)
            if acc is not None:
                acc[1] += time.time() - _ti       # build idx (incl. tiny H2D)
            _ts = time.time() if acc is not None else 0.0
            for i, ln in enumerate(layer_names):
                kv = self._kv_caches[ln]
                layer = kv[0] if isinstance(kv, (list, tuple)) else kv
                srcl = gpu[:, i]                  # [cnt, 2, *rest]
                if dim == 1:
                    layer[:, idx, ...] = srcl.permute(
                        1, 0, *range(2, srcl.dim()))
                else:
                    layer[idx, ...] = srcl
            if acc is not None:
                acc[3] += time.time() - _ts       # 32-layer scatter enqueue
            covered = b
            if acc is not None:
                _tm = time.time()
        return covered >= hi

    def _resolve_arena_runs(self, job_id: str, dst_block_ids: list,
                            src_block_offset: int, acc=None):
        """Resolve a request's gap to a list of CONTIGUOUS arena runs
        [(slot_a, cnt, dst_sub), ...] (one per increment it spans), validated
        against the ring.  Returns None if any block is missing/evicted (the
        whole request then falls back to recompute, same as before).  Does NO
        GPU work — just the cheap index lookup — so the batched scatter can
        plan across all requests."""
        n = len(dst_block_ids)
        if n <= 0:
            return None
        lo = int(src_block_offset)
        hi = lo + n
        _tm = time.time() if acc is not None else 0.0
        incs = self._list_increments(job_id)
        if not incs:
            return None
        next_slot = int(self._hdr[0])
        oldest = next_slot - self._num_slots
        runs = []
        covered = lo
        for (s, e, path) in incs:
            if not path.endswith(".slot"):
                return None                      # arena mode, non-arena chunk
            if e <= lo or s >= hi:
                continue
            if s > covered:
                return None                      # gap in coverage
            base = self._read_slot_index_cached(path)
            if base is None or base < oldest:
                return None                      # evicted by the ring / missing
            a = max(s, lo)
            b = min(e, hi)
            cnt = b - a
            base_off = base % self._num_slots
            slot_a = base_off + (a - s)          # contiguous (alloc skips wrap)
            runs.append((slot_a, cnt, list(dst_block_ids[a - lo:b - lo])))
            covered = b
        if acc is not None:
            acc[0] += time.time() - _tm          # meta: listdir + read .slot
        if covered < hi:
            return None
        return runs

    def _load_batch_arena(self, items: list) -> list:
        """ARENA load for an admit wave with BATCHED scatter.  Instead of
        reqs×nL `index_put` launches (which dominated serving — ~9ms/launch),
        we (1) resolve every request to contiguous arena runs, (2) H2D the runs
        into ONE staging buffer [nb, nL, 2, *rest] (chunked by stage cap), and
        (3) do exactly nL `index_put`s per chunk — ONE per layer across ALL
        requests' blocks.  Launches drop from reqs×nL to (chunks×nL)."""
        # Stage 2 LRU dispatch
        if self._lru_store is not None:
            return self._load_batch_arena_lru(items)
        import torch
        results = [False] * len(items)
        if self._pipe_ev is not None:
            try:
                _e0, _e1, _gbp = self._pipe_ev
                _gpu = _e0.elapsed_time(_e1)
                logger.info(
                    "round-kv LOAD gpu(prev): %.1fGB h2d+scatter_ms=%.0f "
                    "(%.1f GB/s)", _gbp, _gpu,
                    (_gbp / (_gpu / 1e3)) if _gpu else 0.0)
            except Exception:  # pragma: no cover
                pass
            self._pipe_ev = None
        nblk = sum(len(d) for (_, d, _) in items)
        gb = nblk * self._slot_bytes / 1e9
        _t0 = time.time()
        drain_ms = -1.0
        if self._sync_first:
            # DIAGNOSTIC: drain prior-step GPU work, then time the load on an
            # idle GPU.  drain_ms = how long the prior forward was still
            # running; the post-drain engine_block then = contention-free load.
            torch.cuda.synchronize()
            drain_ms = (time.time() - _t0) * 1000.0
            _t0 = time.time()
        acc = [0.0, 0.0, 0.0, 0.0]    # CPU-side: meta, idx, h2d-enq, scat-enq
        layer_names = list(self._kv_caches.keys())
        nL = self._arena_nL
        rest = self._arena_rest
        dim = self._arena_dim
        dtype = self._arena_dtype
        kv0 = self._kv_caches[layer_names[0]]
        dev = (kv0[0] if isinstance(kv0, (list, tuple)) else kv0).device
        per_blk_elems = nL * 2
        for x in rest:
            per_blk_elems *= x
        cap_blocks = max(self._stage_cap_bytes // self._slot_bytes, 1)
        # ---- plan: resolve every request to runs, pre-split to <= cap_blocks.
        # all_runs holds only (slot_a, cnt); the dst block ids are flattened in
        # the SAME order into dst_flat and turned into ONE GPU index tensor for
        # the whole wave (vs a per-chunk as_tensor -> many tiny slow H2Ds).
        all_runs = []                 # (slot_a, cnt)  -- arena-source order
        dst_flat = []                 # all dst block ids, same order
        for k, (j, d, o) in enumerate(items):
            try:
                runs = self._resolve_arena_runs(j, d, o, acc)
            except Exception:  # pragma: no cover
                runs = None
            if runs is None:
                results[k] = False
                continue
            results[k] = True
            for (slot_a, cnt, dst_sub) in runs:
                p = 0
                while p < cnt:                   # split big runs to fit a chunk
                    take = min(cap_blocks, cnt - p)
                    all_runs.append((slot_a + p, take))
                    dst_flat.extend(dst_sub[p:p + take])
                    p += take
        # ONE host->device copy for the whole wave's index (then each chunk just
        # slices a contiguous view -- no per-chunk H2D).
        _ti = time.time()
        idx_all = (torch.as_tensor(dst_flat, device=dev, dtype=torch.long)
                   if dst_flat else None)
        acc[1] += time.time() - _ti
        chunks = 0
        nruns = len(all_runs)
        ev_h2d = []          # per-chunk (start,end) GPU events for H2D phase
        ev_scat = []         # ... for scatter phase  (only under sync_first)
        lstream = (self._get_load_stream() if self._load_stream_scatter
                   else None)
        sctx = (torch.cuda.stream(lstream) if lstream is not None
                else _nullctx())
        e0 = e1 = None
        g0 = 0                       # running block offset into idx_all
        with sctx:
            if lstream is not None:
                e0 = torch.cuda.Event(enable_timing=True)
                e0.record(lstream)
            i = 0
            while i < len(all_runs):
                # accumulate runs up to the staging cap
                chunk = []
                nb = 0
                while i < len(all_runs) and (nb == 0
                                             or nb + all_runs[i][1]
                                             <= cap_blocks):
                    chunk.append(all_runs[i])
                    nb += all_runs[i][1]
                    i += 1
                chunks += 1
                _diag = self._sync_first and lstream is not None
                # ---- Phase 1: H2D each run into one staging buffer ----
                _th = time.time()
                gstage = self._get_stage_gpu(nb * per_blk_elems, dtype, dev)
                staging = gstage[:nb * per_blk_elems].view(
                    (nb, nL, 2) + rest)
                if _diag:
                    eh0 = torch.cuda.Event(enable_timing=True)
                    eh1 = torch.cuda.Event(enable_timing=True)
                    eh0.record(lstream)
                # B: coalesce arena-contiguous runs into ONE H2D.
                pos = 0
                ci = 0
                while ci < len(chunk):
                    slot_a, cnt = chunk[ci]
                    span = cnt
                    nj = ci + 1
                    while nj < len(chunk) and chunk[nj][0] == slot_a + span:
                        span += chunk[nj][1]
                        nj += 1
                    staging[pos:pos + span].copy_(
                        self._arena_view[slot_a:slot_a + span],
                        non_blocking=True)
                    pos += span
                    ci = nj
                if _diag:
                    eh1.record(lstream)
                    ev_h2d.append((eh0, eh1))
                acc[2] += time.time() - _th
                # idx for this chunk = a contiguous slice of the wave's index
                # tensor (built once above) -- no per-chunk H2D.
                all_idx = idx_all[g0:g0 + nb]
                g0 += nb
                # ---- Phase 2: scatter staging -> paged KV ----
                _ts = time.time()
                if _diag:
                    es0 = torch.cuda.Event(enable_timing=True)
                    es1 = torch.cuda.Event(enable_timing=True)
                    es0.record(lstream)
                if self._fused_fn is not None:
                    # ONE fused launch does all nL layers (vs nL index_puts).
                    self._fused_fn(staging, all_idx, self._layer_ptrs, nb, nL,
                                   dim, self._fused_NBLK, self._fused_P)
                else:
                    for li, ln in enumerate(layer_names):
                        kv = self._kv_caches[ln]
                        layer = kv[0] if isinstance(kv, (list, tuple)) else kv
                        srcl = staging[:, li]    # [nb, 2, *rest]
                        if dim == 1:
                            layer[:, all_idx, ...] = srcl.permute(
                                1, 0, *range(2, srcl.dim()))
                        else:
                            layer[all_idx, ...] = srcl
                if _diag:
                    es1.record(lstream)
                    ev_scat.append((es0, es1))
                acc[3] += time.time() - _ts
            if lstream is not None:
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record(lstream)
        gpu_ms = -1.0
        if lstream is not None:
            ev = torch.cuda.Event()
            ev.record(lstream)
            if self._sync_first and e0 is not None and e1 is not None:
                # drained GPU: read the load's own GPU time now (no deferral),
                # so this line is self-contained.
                torch.cuda.synchronize()
                try:
                    gpu_ms = e0.elapsed_time(e1)
                except Exception:  # pragma: no cover
                    gpu_ms = -1.0
            else:
                torch.cuda.current_stream().wait_event(ev)
                if e0 is not None and e1 is not None:
                    self._pipe_ev = (e0, e1, gb)
        eng_ms = (time.time() - _t0) * 1000.0
        # sum the per-phase GPU times (events already complete after the sync)
        gh = gs = -1.0
        if ev_h2d:
            try:
                gh = sum(a.elapsed_time(b) for a, b in ev_h2d)
                gs = sum(a.elapsed_time(b) for a, b in ev_scat)
            except Exception:  # pragma: no cover
                gh = gs = -1.0
        logger.info(
            "round-kv LOAD arena: reqs=%d blocks=%d fail=%d GB=%.2f "
            "engine_block_ms=%.0f | drain_ms=%.0f gpu_ms=%.0f "
            "load_GBps=%.1f | chunks=%d runs=%d launches=%d | "
            "gpu_h2d_ms=%.0f gpu_scat_ms=%.0f | CPU: meta=%.0f "
            "idx=%.0f h2d_enq=%.0f scat_enq=%.0f ms (batched per-layer)",
            len(items), nblk, sum(1 for r in results if not r), gb,
            eng_ms, drain_ms, gpu_ms,
            (gb / (eng_ms / 1e3)) if eng_ms > 0 else 0.0,
            chunks, nruns, chunks * nL, gh, gs,
            acc[0] * 1e3, acc[1] * 1e3, acc[2] * 1e3, acc[3] * 1e3)
        return results

    # ------------------------------------------------------------------
    # ASYNC load — background thread, engine never blocks
    # ------------------------------------------------------------------

    def enqueue_load(self, items: list) -> None:
        """Queue async loads.  `items` = (request_id, job_id, dst_block_ids,
        src_block_offset).  Returns immediately; a bg thread does read+scatter
        and reports completion via drain_loaded().  If not ready, mark done
        right away so the request still unblocks."""
        if not self.ready:
            with self._load_done_lock:
                for it in items:
                    self._load_done.add(it[0])
            return
        self._ensure_async_load()
        for it in items:
            self._load_q.put(it)

    def _ensure_async_load(self) -> None:
        if self._async_load_started:
            return
        self._async_load_started = True
        t = threading.Thread(target=self._async_load_loop, daemon=True,
                             name="RoundKVAsyncLoad")
        t.start()

    def _async_load_loop(self) -> None:
        stream = None
        try:
            import torch
            if self._is_cuda:
                torch.cuda.set_device(self._device)
                stream = torch.cuda.Stream()
        except Exception:  # pragma: no cover
            stream = None
        while not self._stop.is_set():
            try:
                it = self._load_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if it is None:
                break
            rid, job_id, dst, off = it[0], it[1], it[2], it[3]
            slot_gen = it[4] if len(it) > 4 else None   # ★ 跨 job 显式 slot
            _t0 = time.time()
            _nblk = len(dst)
            try:
                # Stage 2 LRU dispatch (在 _arena_registered 检查之前, 否则
                # consumer 端会 fall through 到 FIFO _read_for_load).
                if self._lru_store is not None:
                    import torch
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            self._load_request_arena_lru(
                                job_id, dst, off, slot_gen)
                        stream.synchronize()
                    else:
                        self._load_request_arena_lru(
                            job_id, dst, off, slot_gen)
                        torch.cuda.current_stream().synchronize()
                elif self._arena and self._arena_registered and self._is_cuda:
                    # ARENA: direct H2D from the resident pinned region.
                    import torch
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            self._load_request_arena(job_id, dst, off)
                        stream.synchronize()   # KV present before report-done
                    else:
                        self._load_request_arena(job_id, dst, off)
                        torch.cuda.current_stream().synchronize()
                else:
                    res = self._read_for_load(job_id, dst, off)
                    if res is not None:
                        d2, per_layer = res
                        if stream is not None:
                            import torch
                            with torch.cuda.stream(stream):
                                self._scatter_request_batched(d2, per_layer)
                            stream.synchronize()  # KV present before done
                        else:
                            self._scatter_request_batched(d2, per_layer)
                # One concise line per async load (low frequency).
                logger.info(
                    "round-kv ASYNC-LOAD: blocks=%d ~%.1fGB ms=%.0f "
                    "(bg, engine not blocked)", _nblk,
                    _nblk * 2.097152 / 1e3, (time.time() - _t0) * 1000.0)
            except Exception as e:  # pragma: no cover
                logger.warning("round-kv async load failed req=%s job=%s: %s",
                               rid, job_id, e)
            finally:
                # Always report done -> the request unblocks (even on miss;
                # it then prefills from whatever is present, same risk as the
                # old sync path on a read miss).
                with self._load_done_lock:
                    self._load_done.add(rid)

    def drain_loaded(self) -> set:
        with self._load_done_lock:
            if not self._load_done:
                return set()
            d = self._load_done
            self._load_done = set()
            return d

    def load_into(self, job_id: str, dst_block_ids: list,
                  src_block_offset: int = 0) -> bool:
        """Single-request load (read + scatter).  Kept for tests / fallback;
        production prefill uses load_batch."""
        if not self.ready:
            return False
        # Stage 2 LRU dispatch (绕过 _arena_registered: consumer 端不 register
        # 但仍需要走 LRU 路径读 LRU 格式的 .slot 文件)
        if self._lru_store is not None:
            ok = self._load_request_arena_lru(
                job_id, dst_block_ids, src_block_offset)
            if self._is_cuda:
                import torch
                torch.cuda.current_stream().synchronize()
            return ok
        if self._arena and self._arena_registered and self._is_cuda:
            ok = self._load_request_arena(job_id, dst_block_ids,
                                          src_block_offset)
            if self._is_cuda:
                import torch
                torch.cuda.current_stream().synchronize()
            return ok
        res = self._read_for_load(job_id, dst_block_ids, src_block_offset)
        if res is None:
            return False
        dst, per_layer = res
        if self._batched:
            return self._scatter_request_batched(dst, per_layer)
        return self._scatter_request(dst, per_layer)

    def _ensure_load_pool(self):
        if self._load_pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._load_pool = ThreadPoolExecutor(
                max_workers=max(self._load_workers, 1),
                thread_name_prefix="RoundKVLoad")
        return self._load_pool

    def load_batch(self, items: list) -> list:
        """Parallel load for a whole admit wave.  `items` is a list of
        (job_id, dst_block_ids, src_block_offset).  The CPU file reads run
        in parallel on the load pool (aggregate RAM bandwidth); the H2D
        scatter runs serially on this (engine/worker) thread to avoid
        multi-thread CUDA.  Turns the wave's load cost from sum(load_i) into
        ~max(read_i) + sum(scatter_i).  Returns per-item success bool."""
        if not self.ready or not items:
            return [False] * len(items)
        # Stage 2 LRU dispatch (在所有 FIFO 路径分支前 — 必须最优先, 否则
        # consumer 端 _arena_registered=False 会 fall through 到 _load_batch_pipelined
        # 走 FIFO safetensors 读路径, 但 LRU 写的是新 .slot 格式, 必崩 "header too large").
        if self._lru_store is not None:
            return self._load_batch_arena_lru(items)
        if self._profile and self._is_cuda:
            # Clean per-stage diagnostic (read/pin/h2d/scatter isolated).
            return self._profile_load(items)
        if self._arena and self._arena_registered and self._is_cuda:
            # ARENA: direct H2D from the resident pinned region (no file read,
            # no mmap->pinned copy, no per-load alloc).
            return self._load_batch_arena(items)
        if self._raw and self._is_cuda:
            # RAW contiguous .bin: mmap + bulk H2D + scatter (no strided read).
            return self._load_batch_raw(items)
        if self._pipelined and self._is_cuda:
            # Overlap read (file->pinned, pool) with scatter (copy stream).
            return self._load_batch_pipelined(items)
        pool = self._ensure_load_pool()
        # Phase 1: parallel CPU reads (file IO).
        _t0 = time.time()
        futs = [pool.submit(self._read_for_load, j, d, o)
                for (j, d, o) in items]
        reads = []
        for f in futs:
            try:
                reads.append(f.result())
            except Exception:  # pragma: no cover
                reads.append(None)
        read_ms = (time.time() - _t0) * 1000.0
        # Phase 2 (PROFILE): contention probe + per-segment timing.
        if self._profile:
            probe = self._probe_h2d_gbs()
            _t1 = time.time()
            acc = [0.0, 0.0, 0.0, 0]   # pin_ms, h2d_ms, idx_ms, bytes
            results = []
            for res in reads:
                if res is None:
                    results.append(False)
                    continue
                try:
                    dst, per_layer = res
                    results.append(
                        self._scatter_request_profiled(dst, per_layer, acc))
                except Exception:  # pragma: no cover
                    results.append(False)
            scatter_ms = (time.time() - _t1) * 1000.0
            gbv = acc[3] / 1e9
            logger.info(
                "round-kv PROFILE: reqs=%d bytes=%.2fGB read_ms=%.0f "
                "probe=%.2fGB/s | pin_copy=%.0fms h2d=%.0fms(%.2fGB/s) "
                "index=%.0fms(%.2fGB/s) | total_ms=%.0f",
                len(items), gbv, read_ms, probe, acc[0],
                acc[1], (gbv / (acc[1] / 1000.0)) if acc[1] else 0.0,
                acc[2], (gbv / (acc[2] / 1000.0)) if acc[2] else 0.0,
                read_ms + scatter_ms)
            return results
        # Phase 2: serial, batched H2D scatter (single CUDA thread; pinned
        # double-buffer inside _scatter_request).
        _t1 = time.time()
        results = []
        import torch
        self._acc_pin_ms = 0.0
        lstream = (self._get_load_stream()
                   if (self._batched and self._is_cuda
                       and self._load_stream_scatter) else None)
        sctx = (torch.cuda.stream(lstream) if lstream is not None
                else _nullctx())
        nblk_tot = sum(len(d) for (_, d, _) in items)
        big = nblk_tot > 1000   # diagnose only the slow (huge-prefix) loads
        gpu_ms = -1.0
        e0 = e1 = None
        with sctx:
            if big and lstream is not None:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record(lstream)
            for res in reads:
                if res is None:
                    results.append(False)
                    continue
                try:
                    dst, per_layer = res
                    if self._batched:
                        results.append(
                            self._scatter_request_batched(dst, per_layer))
                    else:
                        results.append(self._scatter_request(dst, per_layer))
                except Exception:  # pragma: no cover
                    results.append(False)
            if big and lstream is not None:
                e1.record(lstream)
        if lstream is not None:
            # Make the forward (default/current stream) wait for the
            # copy-stream load before it reads the KV.  The load thus overlaps
            # the PREVIOUS forward instead of serialising behind it.
            ev = torch.cuda.Event()
            ev.record(lstream)
            torch.cuda.current_stream().wait_event(ev)
        scatter_ms = (time.time() - _t1) * 1000.0
        if e1 is not None:
            # GPU-exec time of H2D+index on the copy stream (one sync; only on
            # the rare big loads).  pin = CPU pin-fill, gpu = actual GPU work.
            try:
                e1.synchronize()
                gpu_ms = e0.elapsed_time(e1)
            except Exception:
                gpu_ms = -1.0
            logger.info(
                "round-kv LOADBIG: reqs=%d blocks=%d bytes~%.1fGB read_ms=%.0f "
                "pin_fill_ms=%.0f gpu_exec_ms=%.0f scatter_ms=%.0f",
                len(items), nblk_tot, nblk_tot * 2.097152 / 1e3, read_ms,
                self._acc_pin_ms, gpu_ms, scatter_ms)
        nblk = sum(len(d) for (_, d, _) in items)
        logger.info(
            "round-kv LOAD batch: reqs=%d blocks=%d fail=%d read_ms=%.0f "
            "scatter_ms=%.0f total_ms=%.0f", len(items), nblk,
            sum(1 for r in results if not r), read_ms, scatter_ms,
            read_ms + scatter_ms)
        return results

    # ------------------------------------------------------------------
    # LOAD (pipelined) — non-blocking; load(layer i+1) || compute(layer i)
    # ------------------------------------------------------------------

    def _get_load_stream(self):
        if self._load_stream is None and self._is_cuda:
            try:
                import torch
                self._load_stream = torch.cuda.Stream()
            except Exception:
                self._load_stream = None
        return self._load_stream

    def _plan_request(self, job_id: str, dst_block_ids: list,
                      src_block_offset: int):
        """Validate coverage and return (dst_block_ids, [(path, fl, fh)])
        for the gap [src_off, src_off+n), contiguous across increments, or
        None on miss.  Computed ONCE per request so the per-layer driver
        loop only does get_slice + H2D."""
        n = len(dst_block_ids)
        if n <= 0:
            return None
        lo = int(src_block_offset)
        hi = lo + n
        if lo < 0:
            return None
        incs = self._list_increments(job_id)
        if not incs:
            return None
        pieces = []
        covered = lo
        for (s, e, path) in incs:
            if e <= lo or s >= hi:
                continue
            a = max(s, lo)
            b = min(e, hi)
            if a > covered:
                return None
            pieces.append((path, a - s, b - s))
            covered = b
        if covered < hi:
            return None
        return list(dst_block_ids), pieces

    def start_load_pipelined(self, items: list) -> None:
        """★ 逐层流水加载: 在 copy stream 上把整波各层的 arena 直读 scatter 一次性
        发出 (各记一个 CUDA event), 立即返回; forward 每层前 wait_layer(layer) 让
        compute stream 等该层 event → 第 i 层算时 copy stream 已在搬第 i+1 层, 重叠.

        每个 producer forward 都调: 空 items 只 deactivate (wait_layer 变 no-op).
        前提: LRU + per-layer 直读 kernel 可用; 否则回退批量 load_batch (同步, 不流水).
        """
        # 上一波先收尾 (unpin), 防 pin 泄漏
        if self._plr_active:
            self.finish_pipelined()
        if not self.ready or not items:
            self._plr_active = False
            self._plr_events = {}
            self._plr_handle = None
            return
        # 回退: 非 CUDA / 无 LRU / 无 per-layer kernel / 直读未就绪
        if (not self._is_cuda or self._lru_store is None
                or self._arena_direct_layer_fn is None
                or self._arena_direct_fn is None):
            self._plr_active = False
            self.load_batch(items)
            return
        try:
            self._start_load_arena_pipelined(items)
        except Exception as e:  # pragma: no cover
            logger.warning("round-kv PIPELINE start failed (%s); fallback "
                           "batched", e)
            self._plr_active = False
            self._plr_events = {}
            self._plr_handle = None
            self.load_batch(items)

    def _start_load_arena_pipelined(self, items: list) -> None:
        import torch
        # 上一波逐层总耗时 (event 已完成, 不阻塞)
        if self._plr_pipe_ev is not None:
            try:
                _e0, _e1, _gb = self._plr_pipe_ev
                _ms = _e0.elapsed_time(_e1)
                logger.info(
                    "round-kv PIPELINE gpu(prev): %.2fGB layer_scatter_ms=%.1f "
                    "(%.1f GB/s, 与 forward 重叠)", _gb, _ms,
                    (_gb / (_ms / 1e3)) if _ms else 0.0)
            except Exception:
                pass
            self._plr_pipe_ev = None

        _t0 = time.time()
        # 1) resolve + pin 整波 (含跨 job 4 元组), 一次
        bh = self._lru_store.load_batch_pin(items)
        nblk = len(bh.slot_ids)
        if nblk == 0:
            bh.release()                 # 无可复用块 -> 不流水, forward 全重算
            self._plr_active = False
            self._plr_events = {}
            self._plr_handle = None
            return
        src_slots = torch.tensor(bh.slot_ids, dtype=torch.int64,
                                 device=self._device)
        dst_idx = torch.tensor(bh.dst_block_ids, dtype=torch.int64,
                               device=self._device)
        gb = nblk * self._slot_bytes / 1e9
        lstream = self._get_load_stream()
        events = {}
        e0 = e1 = None
        # 2) 逐层在 copy stream launch per-layer scatter + 记 event
        with torch.cuda.stream(lstream):
            e0 = torch.cuda.Event(enable_timing=True)
            e0.record(lstream)
            for i, ln in enumerate(self._arena_direct_layer_names):
                layer_ptr = int(self._arena_direct_layers[i].data_ptr())
                self._arena_direct_layer_fn(
                    int(self._arena_addr), src_slots, dst_idx, layer_ptr,
                    nblk, self._arena_nL, i, self._arena_dim,
                    self._arena_direct_NBLK, self._arena_direct_P)
                ev = torch.cuda.Event()
                ev.record(lstream)
                events[ln] = ev
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record(lstream)
        # ★ 关键: 索引 tensor 被 copy stream 跨 32 层引用, 必须 (1) record_stream
        # 告诉 allocator copy stream 在用, (2) 留活到 finish_pipelined (forward 后).
        # 否则函数返回即被复用覆写 → scatter 读乱 slot 越界 → 下个 gemm 爆 CUBLAS.
        try:
            src_slots.record_stream(lstream)
            dst_idx.record_stream(lstream)
        except Exception:
            pass
        self._plr_src = src_slots
        self._plr_dst = dst_idx
        self._plr_events = events
        self._plr_handle = bh
        self._plr_active = True
        self._plr_pipe_ev = (e0, e1, gb)
        logger.info(
            "round-kv PIPELINE start: reqs=%d blocks=%d GB=%.2f layers=%d "
            "issue_ms=%.0f (copy stream, 逐层 event)",
            len(items), nblk, gb, len(events), (time.time() - _t0) * 1000.0)

    def wait_layer(self, layer_name: str) -> None:
        """forward 每层前调: 让 compute stream 等该层 copy event (GPU 侧等, 不阻塞
        CPU). 第 i 层算时 copy stream 已在搬后面的层, 实现重叠. 无流水时 no-op."""
        if not self._plr_active:
            return
        ev = self._plr_events.get(layer_name)
        if ev is None:
            return                          # 该层不在本波 load 内
        try:
            import torch
            torch.cuda.current_stream().wait_event(ev)
        except Exception:
            try:
                ev.synchronize()
            except Exception:
                pass

    def finish_pipelined(self) -> None:
        """forward 后调 (wait_for_save): 确保各层 copy 真完成 → post-load gen 校验
        → unpin. 必须等 copy 完成再 unpin: 否则 pin 早释放, slot 可能被淘+复用,
        而 copy stream 还在读它 → 脏数据. 此时 copy(~百 ms) 早被 forward 盖掉, 同步
        最后一层 event 通常 ~0 等待."""
        if not self._plr_active:
            return
        bh = self._plr_handle
        events = self._plr_events
        self._plr_active = False
        self._plr_events = {}
        self._plr_handle = None
        if events:
            try:
                # 同步最后一层 copy event = 等所有 scatter 落地 (copy stream 顺序)
                next(reversed(events.values())).synchronize()
            except Exception:
                pass
        # copy 已全部落地, 索引 tensor 可释放 (留活到此防 forward 中被复用覆写)
        self._plr_src = None
        self._plr_dst = None
        if bh is None:
            return
        try:
            if not bh.post_load_validate():
                self._lru_postload_fail_count += 1
                logger.error(
                    "round-kv PIPELINE post-load gen MISMATCH (canary,total=%d):"
                    " pin/evict 不变量被破坏 — 查 evict 是否绕过 can_evict.",
                    self._lru_postload_fail_count)
        finally:
            bh.release()

    # ------------------------------------------------------------------
    # Lifecycle / cleanup
    # ------------------------------------------------------------------

    def has(self, job_id: str) -> bool:
        return os.path.exists(self._manifest_path(job_id))

    def delete(self, job_id: str) -> None:
        """Remove a job's whole increment history (trajectory end /
        eviction).  Best-effort."""
        # Stage 2 LRU dispatch
        if self._lru_store is not None:
            try:
                self._delete_lru(job_id)
            except Exception as e:  # pragma: no cover
                logger.debug("LRU delete failed job=%s: %s", job_id, e)
            self._last_stored.pop(job_id, None)
            self._inc_cache.pop(job_id, None)
            return
        with self._job_lock(job_id):
            self._last_stored.pop(job_id, None)
            self._inc_cache.pop(job_id, None)
            try:
                shutil.rmtree(self._job_dir(job_id))
            except FileNotFoundError:
                pass
            except OSError as e:  # pragma: no cover
                logger.debug("RoundKVStore.delete failed job=%s: %s",
                             job_id, e)
        # Phase A: clear reverse-index so this job's old slot mappings
        # no longer protect them in _arena_alloc.  Physical slots are
        # still occupied with this job's bytes (until bump overwrites),
        # but lookup() will miss (manifest gone) so they're effectively
        # free.
        with self._arena_lock:
            slots = self._job_to_slots.pop(str(job_id), None)
            if slots:
                for sid in slots:
                    cur = self._slot_to_inc.get(sid)
                    if cur is not None and cur[0] == str(job_id):
                        del self._slot_to_inc[sid]
            self._finished_jobs.discard(str(job_id))

    def mark_finished(self, job_id: str) -> None:
        """Phase A: scheduler signals the trajectory's last step has
        finished — the job's KV history is no longer needed.  We:
        (1) mark the job as finished so its slots are immediately
            considered overwritable in _arena_alloc (no longer
            protected even if inc_start==0),
        (2) asynchronously delete the manifest + reverse-index so
            lookup() stops returning hits.
        Best-effort: any errors are logged and swallowed so a buggy
        scheduler call can't kill the store."""
        if not job_id:
            return
        jid = str(job_id)
        with self._arena_lock:
            self._finished_jobs.add(jid)
        # Run the actual delete on a daemon thread so we don't block
        # the scheduler.  delete() takes per-job lock + rmtree -> ms
        # range, not worth blocking on.
        try:
            t = threading.Thread(target=self.delete, args=(jid,),
                                 daemon=True, name=f"RKVDel-{jid[:16]}")
            t.start()
        except Exception as e:  # pragma: no cover
            logger.debug("mark_finished thread launch failed job=%s: %s",
                         jid, e)

    def _is_protected(self, slot_id: int) -> bool:
        """Phase A: slot is protected iff it currently belongs to an
        ACTIVE (non-finished) job's HEAD increment (inc_start == 0).
        Head increment is the prompt prefix -> shared across rounds of
        the same job -> don't let bump overwrite it while the job is
        still running.  Caller must hold _arena_lock."""
        e = self._slot_to_inc.get(slot_id)
        if e is None:
            return False
        job_id, inc_start, _ = e
        if job_id in self._finished_jobs:
            return False
        return inc_start == 0

    def shutdown(self) -> None:
        self._stop.set()
        for _ in self._threads:
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        if self._load_pool is not None:
            try:
                self._load_pool.shutdown(wait=False)
            except Exception:
                pass


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
