# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import regex as re
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

# ★ LICHT_PROBE=1: master switch for stall-investigation probes (SLK-SLOW,
# MIG-LOOP-SLOW here). Default off → zero overhead. See vllm/v1/engine/core.py.
# (Module has a global `import os`; methods still use local `import os as _os`.)
_LICHT_PROBE = os.environ.get("LICHT_PROBE") == "1"


@dataclass
class ReqMeta:
    # Request Id
    request_id: str
    # Request block ids
    block_ids: torch.Tensor
    # Request num tokens
    num_tokens: int
    # Optional explicit remote endpoints.
    remote_prefill_address: Optional[str] = None
    remote_decode_address: Optional[str] = None

    @staticmethod
    def make_meta(request_id: str, token_ids: list[int], block_ids: list[int],
                  block_size: int,
                  remote_prefill_address: Optional[str] = None,
                  remote_decode_address: Optional[str] = None) -> "ReqMeta":
        block_ids_tensor = torch.tensor(block_ids)
        return ReqMeta(
            request_id=request_id,
            block_ids=block_ids_tensor,
            num_tokens=len(token_ids),
            remote_prefill_address=remote_prefill_address,
            remote_decode_address=remote_decode_address,
        )


@dataclass
class RoundReqMeta:
    """LICHT cross-round KV reuse entry passed scheduler→worker.

    `block_ids` are GPU blocks: on prefill `round_load` the freshly
    allocated prefix blocks to fill from the store; on decode
    `round_store` the finished request's blocks to persist.  `token_ids`
    is the full sequence (store only; empty for load)."""
    request_id: str
    job_id: str
    block_ids: list[int]
    token_ids: list[int]
    num_blocks: int = 0
    # round_load only: index of the first SAVED block to read.  Skips the
    # leading prefix blocks already present in GPU via the local prefix
    # cache, so block_ids (the destination) and the saved source stay
    # aligned even when a local hit and a round-kv hit coexist.
    src_block_offset: int = 0
    # ★ Stage 6d cross-job: resolved (slot, gen) per dst block, from the
    # scheduler-side lookup_resolve (content-addressing).  Two parallel int
    # lists, aligned with block_ids.  None/empty = own-job load (worker reads
    # its own .slot via src_block_offset).  Carries the table-resolved slots to
    # the worker so a brand-new job can load another job's shared prefix.
    src_slots: Optional[list] = None
    src_gens: Optional[list] = None
    # ★ P2 SSD 升级段: scheduler 侧 lookup_resolve_tiered 续探到的 SSD 块.
    # worker 在 load 前把它们 promote 回 CPU (pread), 然后统一走 CPU->GPU.
    # 三个平行 int 列表逐块对齐, 覆盖绝对块区间 [ssd_start, ssd_start+len).
    # None/空 = 无 SSD 段.
    ssd_start: int = 0
    ssd_slots: Optional[list] = None
    ssd_gens: Optional[list] = None
    ssd_hashes: Optional[list] = None


def _rl_slot_gen(rl: "RoundReqMeta"):
    """★ Stage 6d: 从 RoundReqMeta 取跨 job 显式 (slot,gen) 列表 (与 block_ids
    对齐); None = own-job (worker 走自己的 .slot)."""
    if rl.src_slots and rl.src_gens:
        return list(zip(rl.src_slots, rl.src_gens))
    return None


def _rl_item(rl: "RoundReqMeta"):
    """构建 load_batch/pipelined 的 item. 无跨 job slot 时返回 3 元组 (与旧路径
    完全一致, FIFO/raw/pipelined 消费者不受影响); 有则 4 元组带 (slot,gen) 列表
    (仅 LRU content-addr 路径会消费)."""
    sg = _rl_slot_gen(rl)
    if sg is None:
        return (rl.job_id, rl.block_ids, rl.src_block_offset)
    return (rl.job_id, rl.block_ids, rl.src_block_offset, sg)


def _rl_item_async(rl: "RoundReqMeta"):
    """enqueue_load 的 item (带 request_id). 同样按需 4→5 元组."""
    sg = _rl_slot_gen(rl)
    if sg is None:
        return (rl.request_id, rl.job_id, rl.block_ids, rl.src_block_offset)
    return (rl.request_id, rl.job_id, rl.block_ids, rl.src_block_offset, sg)


@dataclass
class P2pNcclConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]
    # 回传 REMOVED (2026-05-21): no v3 push-back/offload metadata fields.

    def __init__(self):
        self.requests = []
        # LICHT round-kv cross-round reuse (empty unless enabled).
        self.round_load: list[RoundReqMeta] = []
        self.round_store: list[RoundReqMeta] = []
        # Phase 1 (save-on-preempt): per-step preempt-victim saves the
        # worker must execute synchronously in start_load_kv (the gather
        # must complete before forward overwrites the victim's slots).
        # Empty unless LICHT_PHASE1_SAVE_ON_PREEMPT=1.
        self.preempt_store: list[RoundReqMeta] = []
        # Phase 2 (PD path selector): per-step arena-sink requests the
        # decode worker fires as ARENA_SINK RPCs to prefill.  Each
        # entry = (request_id, job_id, prompt_token_ids,
        # remote_prefill_address).  Prefill side D2H's the KV and
        # releases its GPU blocks; decode side admits via arena-load
        # next pass.  Empty unless LICHT_PHASE2_ADMISSION_GATE=1.
        self.arena_sink: list[tuple] = []
        # ★ P2 修 (2026-07-05 定案): Phase2 defer 期的后台修复搬运 —— 探测
        # 发现 "CPU 有洞、洞在 SSD" (冷期被降级的历史段) 时, 让 worker 后台
        # 把该段 SSD->CPU 搬回, CPU 真齐后按【纯 CPU】判定 admit. SSD 不参与
        # "齐" 的承诺 (decode 永不重算的铁律: 失败模式只能是"多等", 不能是
        # "错"). 每 entry = (request_id, job_id, ssd_start, slots, gens,
        # hashes). 幂等 (内容寻址, 重复搬零成本).
        self.arena_promote: list[tuple] = []
        # Phase A: job_ids whose trace has finished (is_last_step on a
        # request just completed).  Worker iterates and calls
        # _round_store_obj.mark_finished so its in-memory eviction
        # bookkeeping updates.  Schedule-only drain — broadcast every
        # step; usually empty.
        self.finished_jobs: list[str] = []

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        remote_prefill_address: Optional[str] = None,
        remote_decode_address: Optional[str] = None,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(
                request_id,
                token_ids,
                block_ids,
                block_size,
                remote_prefill_address,
                remote_decode_address,
            ))


class P2pNcclConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer
        self.send_type = str(
            self.config.get_from_extra_config("send_type",
                                              "PUT_ASYNC")).upper()
        self.direct_block_mode = self.send_type in {
            "BLOCK_MIGRATE", "BLOCK_DIRECT", "DISTSERVE"
        }
        self.chunked_prefill: dict[str, Any] = {}
        self._pending_bridge_reqs: list[tuple[str, list[int]]] = []
        # Per prefill-step phase timing (producer): confirms whether the
        # SYNCHRONOUS round-kv load blocks the engine.  idle=gap since prev
        # step, load=time stuck in start_load_kv (round-kv read+scatter),
        # fwd=forward, bridge=stage_bridge_request.
        self._step_t_load_start = None
        self._step_t_load_end = None
        self._step_prev_save_end = None
        self._pending_failed_block_migrations: dict[
            str, tuple[list[int], list[int], str]
        ] = {}
        # 回传 REMOVED (2026-05-21): no v3 push-back/offload queues, bg
        # thread, tier handles, or push bridge.

        # LICHT cross-round KV reuse store (CPU/SSD).  Enabled when
        # extra_config["round_kv_reuse_path"] is set (e.g. /dev/shm/...).
        # prefill (producer) LOADs a returning request's prior-round KV
        # before forward; decode (consumer) STOREs a finished request's
        # full-sequence KV via the delay-free + get_finished protocol.
        self._round_kv_path = str(
            self.config.get_from_extra_config("round_kv_reuse_path", "")
            or "")
        self._round_kv_enabled = bool(self._round_kv_path)
        # ASYNC load (default OFF): async parks requests in
        # WAITING_FOR_REMOTE_KVS for a variable (load-dependent) time, which
        # breaks the LICHT-V3 admit predictor (it assumes the deterministic
        # sync pattern: admit -> load this step -> run this step).  So default
        # to SYNCHRONOUS (admit a batch -> load all their KV -> run together).
        # LICHT_ROUND_KV_ASYNC=1 re-enables async (engine non-blocking) for
        # experiments where the predictor isn't in play.
        import os as _os
        self._round_async = (
            _os.environ.get("LICHT_ROUND_KV_ASYNC", "0") == "1")
        self._round_store_obj = None  # RoundKVStore (lazy)
        # [Stage 6d perf] 按 request_id 缓存 lookup_resolve 结果: 同一请求在
        # get_num + update_state (每步 ×2) 及跨步反复 probe 时只算一次. 每步
        # build_connector_meta 丢弃本步未再 probe 的条目 (_rk_lk_seen), 不泄漏.
        self._rk_lk_cache: dict[str, Any] = {}
        self._rk_lk_seen: set = set()
        # ★ P2: claim 了 SSD 段的请求 (req_id -> job_id). worker promote 后
        # 清 SSD inflight; 请求没走到 load 就挂掉时由 request_finished 兜底清.
        self._ssd_marked: dict[str, str] = {}
        # ★ P2 修 (当步现探): SSD 段探测结果只在【本步】有效 —— rid ->
        # (step_no, seg). 步内 get_num/update_state 共用同一探测 (防两次
        # 现探结果不一致导致 claim/记账错位), 跨步必重探 (防陈旧地图).
        self._rk_step_no = 0
        self._rk_ssd_seg: dict[str, Any] = {}
        # ★ Phase2 专用查询的缓存 (2026-07-04 18:17 复盘: Phase2 覆盖判定
        # 误用了跨步缓存的 _rk_lookup_cached -> sink 写齐后 matched 永远停在
        # 首次探测值 -> 5 个请求 PRESENT 却永远 SHORT/defer). Phase2 是轮询
        # "写齐了没", CPU 部分必须每次现算 (C 快路径 ~1ms); 贵的 Python
        # tail hash 按 (matched 边界) 缓存, 边界变才重算.
        self._rk_ph2_tail: dict[str, Any] = {}   # rid -> (mb, tail_hashes)
        self._rk_ph2_res: dict[str, Any] = {}    # rid -> (step_no, res) 步内共用
        # ★ Phase2 defer 期后台修复搬运 (2026-07-05): rid -> 上次入队 step
        # (节流, 25 步重试一次; 搬运幂等); _q 每步由 build_connector_meta
        # 排空进 metadata.arena_promote 下发 worker.
        self._ph2_promote_pending: dict[str, int] = {}
        self._ph2_promote_q: list = []
        if self._round_kv_enabled:
            try:
                from vllm.v1.core.sched.licht_v3.round_kv_store import (
                    RoundKVStore)
                self._round_store_obj = RoundKVStore(
                    self._round_kv_path, self._block_size)
                logger.info(
                    "LICHT round-kv reuse enabled (role=%s, is_producer=%s, "
                    "path=%s) | Phase1_save_on_preempt=%s | "
                    "Phase2_admission_gate=%s threshold=%.2f",
                    role, self.is_producer, self._round_kv_path,
                    _os.environ.get(
                        "LICHT_PHASE1_SAVE_ON_PREEMPT", "0") == "1",
                    _os.environ.get(
                        "LICHT_PHASE2_ADMISSION_GATE", "0") == "1",
                    float(_os.environ.get(
                        "LICHT_PHASE2_GATE_THRESHOLD", "0.80")))
            except Exception as e:
                logger.warning("LICHT round-kv init failed: %s; disabling.", e)
                self._round_kv_enabled = False
        # Scheduler-side bookkeeping: producer prefix-load / decode store.
        self._round_load_reqs: dict[str, tuple] = {}
        self._pending_round_store: dict[str, tuple] = {}
        # ---- Phase 1: save-on-preempt (decode side) ----
        # When scheduler preempts a running decode request, instead of
        # discarding its KV (recompute path), we synchronously D2H-save
        # the increment to arena.  On re-admit, the consumer's
        # get_num_new_matched_tokens looks the request up in arena and
        # loads from there (skipping the catastrophic re-prefill of the
        # entire accumulated context).  request_id -> job_id of saved.
        self._phase1_save_on_preempt = (
            _os.environ.get("LICHT_PHASE1_SAVE_ON_PREEMPT", "0") == "1")
        # Scheduler-side: req_id -> job_id of requests whose KV was
        # handed off to the connector for save-on-preempt.  Consumes by
        # the consumer recovery path (get_num_new_matched_tokens /
        # update_state_after_alloc) to route to arena instead of NCCL.
        self._preempt_saved: dict[str, str] = {}
        # Scheduler-side: drained by build_connector_meta into
        # meta.preempt_store so the worker can do the actual save.
        # req_id -> (job_id, block_ids, all_token_ids).
        self._pending_preempt_store: dict[str, tuple] = {}

        # ---- Phase A: trace-finished job IDs to evict on the worker
        # ---- (mark_finished_job is called by the scheduler from
        # ---- _free_request when request.is_last_step is True).
        # Drained by build_connector_meta into meta.finished_jobs.  The
        # worker iterates this list at the top of start_load_kv and
        # calls _round_store_obj.mark_finished so its in-memory
        # _finished_jobs + reverse index update (the scheduler-side
        # instance only handles manifest deletion).
        self._pending_finish_jobs: set[str] = set()

        # ---- Phase 2: PD admission path selector (80% by default) ----
        # When the projected decode KV occupancy after admitting this
        # request (FCFS view: current occupancy already reflects the
        # admit decisions earlier in this same scheduler pass) would
        # cross the threshold, route the PD handoff through the CPU
        # arena instead of NCCL GPU-GPU.  Prefill side D2H'es to arena
        # and releases its GPU blocks; decode side loads from arena
        # when it eventually gets admitted.  Both paths self-release
        # symmetrically (NCCL: RELEASE RPC; arena: D2H done -> mark).
        self._phase2_admission_gate = (
            _os.environ.get("LICHT_PHASE2_ADMISSION_GATE", "0") == "1")
        try:
            self._phase2_gate_threshold = float(
                _os.environ.get("LICHT_PHASE2_GATE_THRESHOLD", "0.80"))
        except ValueError:
            self._phase2_gate_threshold = 0.80
        self._admission_kv_usage: float = 0.0
        self._admission_kv_total_blocks: int = 0
        # Scheduler-side: req_ids the connector has decided to route
        # through arena this pass — drained by build_connector_meta
        # into meta.arena_sink so the worker can fire the ARENA_SINK
        # RPC to prefill.  Tuple = (job_id, prompt_token_ids,
        # remote_prefill_address).
        self._pending_arena_sink: dict[str, tuple] = {}
        # Scheduler-side: req_ids already routed to arena (RPC sent
        # this step OR earlier).  Cleared when the request is
        # admitted from arena (in update_state_after_alloc).  Until
        # then, every get_num_new_matched_tokens attempt for this
        # req checks arena instead of falling into NCCL.
        self._arena_sinked: set[str] = set()
        # ★ 每个 sink 请求的发起时刻 (deadline 兜底用). ARENA_SINK RPC 偶发被
        # prefill declined (bridge 已 RELEASE / 竞态), 此时 arena 永远没数据,
        # 该 req 在 get_num_new_matched_tokens 会永远 lookup miss → return None
        # 永久 defer (update_state 不被调 → 标记不清) → 挂到 600s 超时. 用 deadline:
        # 超时未就绪就放弃 arena、退回 NCCL, 并标 failed 防再次 sink.
        self._arena_sink_ts: dict[str, float] = {}
        self._arena_sink_failed: set[str] = set()
        # ★ 诊断(节流): 记每个 sink 请求上次打 arena-probe 日志的时刻, 用于区分
        #   "数据被淘(matched<need)" vs "数据在却没 admit(matched>=need)"。
        self._arena_probe_log_ts: dict[str, float] = {}
        self._arena_sink_deadline_s = float(
            _os.environ.get("LICHT_ARENA_SINK_DEADLINE_S", "30"))
        # Worker-side store-completion is tracked inside RoundKVStore
        # (drain_done) now that the write is async; get_finished reads it.

        self._rank = get_world_group().rank \
            if role == KVConnectorRole.WORKER else 0
        self._local_rank = get_world_group().local_rank \
            if role == KVConnectorRole.WORKER else 0

        # Bind ZMQ ROUTER to config.kv_ip when set (typically localhost
        # for single-host deployments).  Default `hostname=""` would
        # fall back to `get_ip()` which returns the machine's outward
        # interface IP — ZMQ then refuses `127.0.0.1` connections,
        # which breaks LICHTV3's decode→prefill RPC.  Honouring kv_ip
        # keeps both directions on the same address.
        _bind_host = getattr(self.config, "kv_ip", None) or ""
        self.p2p_nccl_engine = P2pNcclEngine(
            local_rank=self._local_rank,
            config=self.config,
            hostname=_bind_host,
            port_offset=self._rank,
        ) if role == KVConnectorRole.WORKER else None

        # Phase 2: producer-side worker wires ARENA_SINK plumbing so
        # the engine's router thread can call back into the connector
        # to do D2H and the round-kv-store can call back when D2H is
        # done.  Consumer worker / scheduler don't need these hooks
        # (they only SEND ARENA_SINK; receiving + handling is producer).
        # Worker-side _arena_sink_pending tracks req_ids waiting for
        # D2H completion so we know which _done events to forward
        # onto the producer's fast-release queue.
        self._arena_sink_pending: set[str] = set()
        self._arena_sink_decode_addr: dict[str, str] = {}
        if (role == KVConnectorRole.WORKER
                and self.p2p_nccl_engine is not None
                and self.is_producer):
            try:
                self.p2p_nccl_engine.set_arena_sink_handler(
                    self._handle_arena_sink_rpc)
            except AttributeError:
                pass
            if self._round_store_obj is not None:
                try:
                    self._round_store_obj.set_done_hook(
                        self._on_round_store_done)
                except AttributeError:
                    pass

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if self.p2p_nccl_engine is not None:
            self.p2p_nccl_engine.register_kv_caches(kv_caches)
        if self._round_store_obj is not None:
            self._round_store_obj.bind_kv_caches(
                kv_caches, is_producer=self.is_producer)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """

        assert self.p2p_nccl_engine is not None

        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)

        # 回传 REMOVED (2026-05-21): no v3 offload/pushback bg work here.

        # Phase A: process finished-job notifications from scheduler.
        # Both producer and consumer workers do this so each side's
        # _round_store_obj updates its in-memory _finished_jobs (the
        # scheduler-side handles manifest deletion).  Idempotent +
        # cheap (set membership + bg-thread delete).
        if (self._round_kv_enabled and self._round_store_obj is not None
                and metadata.finished_jobs):
            for _jid in metadata.finished_jobs:
                try:
                    self._round_store_obj.mark_finished(_jid)
                except Exception as e:  # pragma: no cover
                    logger.debug(
                        "worker mark_finished failed job=%s: %s", _jid, e)

        if self.direct_block_mode:
            if self.is_producer:
                self._step_t_load_start = time.time()
                # LICHT round-kv: load the reused prior-round prefix into
                # the GPU paged buffer BEFORE the forward, so attention
                # sees it.  Delete-on-load (consume-once).
                if self._round_kv_enabled:
                    # ★ P2: load 分发前把各请求的 SSD 段搬回 CPU (同步基线,
                    # 卡引擎 = pread 时长, claim 期 MAX_MB 已封顶). 之后
                    # 三种分发模式 (async/pipelined/batch) 均不感知 SSD.
                    if metadata.round_load:
                        self._promote_ssd_segments(metadata.round_load)
                    # NO delete-on-load (incremental keeps history; later
                    # rounds append only the delta).
                    if self._round_async:
                        # ASYNC: enqueue to a bg thread and RETURN immediately
                        # — the engine never blocks.  These requests are parked
                        # in WAITING_FOR_REMOTE_KVS; get_finished reports them
                        # done (recving) once the bg load fills their blocks.
                        if metadata.round_load:
                            self._round_store_obj.enqueue_load(
                                [_rl_item_async(rl)
                                 for rl in metadata.round_load])
                    elif self._round_store_obj.pipeline_enabled:
                        # (sync) Layer-wise pipelined load.
                        _items = [_rl_item(rl) for rl in metadata.round_load]
                        self._round_store_obj.start_load_pipelined(_items)
                    elif metadata.round_load:
                        # (sync) Batched load: BLOCKS the engine until done.
                        _items = [_rl_item(rl) for rl in metadata.round_load]
                        self._round_store_obj.load_batch(_items)
                self._step_t_load_end = time.time()
                # Bridge publication must happen after prefill forward.
                self._pending_bridge_reqs.extend((
                    req.request_id,
                    [int(x) for x in req.block_ids.tolist()],
                ) for req in metadata.requests)
                return

            # Phase 1 (save-on-preempt) SAVE side: scheduler preempted
            # one or more victims this step and asked us to persist their
            # KV to arena.  Run synchronously here so the gather reads
            # the slots BEFORE the new tenant's forward (this same step)
            # overwrites them.  Per-victim wait is bounded by
            # save_preempted_sync's timeout.
            if (self._phase1_save_on_preempt
                    and self._round_kv_enabled and metadata.preempt_store
                    and self._round_store_obj is not None):
                _ps_t = time.time()
                for ps in metadata.preempt_store:
                    try:
                        ok = self._round_store_obj.save_preempted_sync(
                            ps.job_id, list(ps.block_ids),
                            list(ps.token_ids), ps.request_id)
                        if not ok:
                            logger.warning(
                                "Phase1 preempt-save timed out req=%s "
                                "job=%s nblk=%d (recompute fallback)",
                                ps.request_id, ps.job_id, len(ps.block_ids))
                    except Exception as e:  # pragma: no cover
                        logger.warning(
                            "Phase1 preempt-save failed req=%s: %s",
                            ps.request_id, e)
                try:
                    _ps_ms = (time.time() - _ps_t) * 1000.0
                    if _LICHT_PROBE and _ps_ms > 500.0:
                        logger.warning("SLK-SLOW preempt_save=%.0fms n=%d",
                                       _ps_ms, len(metadata.preempt_store))
                except Exception:
                    pass

            # Phase 2 (PD path selector) ARENA_SINK fires: tell each
            # prefill side to D2H its KV for this request and release
            # its prefill GPU blocks.  Fire-and-(near-)forget — the
            # RPC just enqueues the D2H on the prefill side.  Decode
            # admits this request from arena in a later pass via
            # get_num_new_matched_tokens (which waits for arena
            # lookup to succeed).
            if (self._phase2_admission_gate and metadata.arena_sink
                    and self.p2p_nccl_engine is not None):
                _as_t = time.time()
                for (req_id, job_id, prompt_tids, remote_addr) in \
                        metadata.arena_sink:
                    if not remote_addr:
                        logger.warning(
                            "ARENA_SINK skip req=%s: no remote addr",
                            req_id)
                        continue
                    try:
                        ok = self.p2p_nccl_engine.send_arena_sink_request(
                            req_id, job_id, prompt_tids, remote_addr)
                        if ok:
                            logger.info(
                                "ARENA_SINK sent req=%s job=%s "
                                "remote=%s ntoks=%d",
                                req_id, job_id, remote_addr,
                                len(prompt_tids))
                        else:
                            logger.warning(
                                "ARENA_SINK declined req=%s remote=%s "
                                "(prefill miss / no handler)",
                                req_id, remote_addr)
                    except Exception as e:  # pragma: no cover
                        logger.warning(
                            "ARENA_SINK send failed req=%s remote=%s: %s",
                            req_id, remote_addr, e)
                try:
                    _as_ms = (time.time() - _as_t) * 1000.0
                    if _LICHT_PROBE and _as_ms > 500.0:
                        logger.warning("SLK-SLOW arena_sink_send=%.0fms n=%d",
                                       _as_ms, len(metadata.arena_sink))
                except Exception:
                    pass

            # Consumer recovery (Phase 1 save-on-preempt AND Phase 2
            # admission-gate): arena → GPU paged buffer for re-admitted
            # requests whose KV lives in arena.  Both phases register the
            # request in _round_load_reqs (drained into metadata.round_load),
            # so this load must fire if EITHER phase is on — not just phase 1.
            # (Bug fix: previously gated on _phase1_save_on_preempt only, so
            # Phase-2-only deployments never read the sunk KV back.)
            # Done BEFORE the NCCL pull loop so attention sees the prefix this
            # step.  Sync load (small per-request increment).
            # ★ P2 修 (2026-07-05): Phase2 defer 期修复搬运 —— 后台线程把
            # "CPU 有洞、洞在 SSD" 的历史段搬回 CPU (不占引擎), CPU 真齐后
            # 调度器按纯 CPU 判定 admit. consumer 的 load 永远只读 CPU.
            if (self._phase2_admission_gate and metadata.arena_promote
                    and self._round_store_obj is not None):
                for (_rid, _job, _a, _slots, _gens, _hashes) in \
                        metadata.arena_promote:
                    try:
                        self._round_store_obj.enqueue_promote(
                            _job, int(_a),
                            list(zip(_slots, _gens, _hashes)))
                    except Exception as e:  # pragma: no cover
                        logger.warning(
                            "arena_promote enqueue failed req=%s: %s",
                            _rid, e)
            if ((self._phase1_save_on_preempt or self._phase2_admission_gate)
                    and self._round_kv_enabled and metadata.round_load
                    and self._round_store_obj is not None):
                _items = [_rl_item(rl) for rl in metadata.round_load]
                _lb_t = time.time()
                try:
                    _res = self._round_store_obj.load_batch(_items)
                    _lb_ms = (time.time() - _lb_t) * 1000.0
                    if _LICHT_PROBE and _lb_ms > 500.0:
                        logger.warning("SLK-SLOW load_batch=%.0fms n=%d",
                                       _lb_ms, len(_items))
                    logger.info(
                        "consumer arena load_batch: reqs=%d ok=%d "
                        "(Phase1/2 recovery, decode reads arena)",
                        len(_items), sum(1 for r in _res if r))
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "consumer arena load_batch failed: %s "
                        "(falling through to recompute path)", e)

            # LICHT round-kv: decode STOREs finished requests' full-seq KV
            # from their delay-free-retained blocks.  Completions are
            # reported via get_finished so the scheduler frees them.
            if self._round_kv_enabled and metadata.round_store:
                for rs in metadata.round_store:
                    # Engine ONLY enqueues (no GPU op, no wait).  The
                    # background pool does the incremental gather (own CUDA
                    # stream) + write, and marks the request done once its
                    # gather completes, releasing the delay-free blocks.
                    self._round_store_obj.enqueue_store(
                        rs.job_id, rs.block_ids, rs.token_ids, rs.request_id)

            _mig_t = time.time()
            _mig_n = len(metadata.requests)
            for req_meta in metadata.requests:
                remote_address = req_meta.remote_prefill_address
                if remote_address is None:
                    ip, port = self.parse_request_id(req_meta.request_id,
                                                     is_prefill=False)
                    remote_address = f"{ip}:{port + self._rank}"

                decoding_block_ids = [int(x) for x in req_meta.block_ids.tolist()]
                pending_migration = self._pending_failed_block_migrations.get(
                    req_meta.request_id)

                if pending_migration is not None:
                    context_block_ids, pending_decoding_block_ids, \
                        pending_remote_address = pending_migration
                    if pending_remote_address != remote_address:
                        logger.warning(
                            "⚠️[BLOCK]Remote address changed while retrying "
                            "migration for req:%s, old:%s, new:%s",
                            req_meta.request_id,
                            pending_remote_address,
                            remote_address,
                        )
                    remote_address = pending_remote_address
                    decoding_block_ids = pending_decoding_block_ids
                else:
                    # Pure decode-pull mode: fetch bridge metadata via a
                    # single non-blocking BRIDGE_POP probe.  If the
                    # producer has not staged yet, stay in
                    # WAITING_FOR_REMOTE_KVS and retry next forward step.
                    context_block_ids = \
                        self.p2p_nccl_engine.pop_bridge_request(
                            req_meta.request_id,
                            remote_address,
                            timeout_s=0.0,
                        )
                    if context_block_ids is None:
                        continue

                migrated = self.p2p_nccl_engine.launch_block_migration(
                    req_meta.request_id,
                    context_block_ids,
                    decoding_block_ids,
                    remote_address,
                )
                if migrated:
                    self._pending_failed_block_migrations.pop(
                        req_meta.request_id, None)
                else:
                    # Keep request in waiting-for-remote-kv and retry next step
                    # with the same bridge metadata.
                    self._pending_failed_block_migrations[req_meta.request_id] = (
                        context_block_ids,
                        decoding_block_ids,
                        remote_address,
                    )
            try:
                _mig_ms = (time.time() - _mig_t) * 1000.0
                if _LICHT_PROBE and _mig_ms > 500.0:
                    logger.warning(
                        "MIG-LOOP-SLOW total=%.0fms nreq=%d — direct_block "
                        "迁移循环(pop_bridge_request + launch_block_migration)"
                        "整体阻塞, 结合 BRIDGE-POP-SLOW/LBM-SLOW 定位",
                        _mig_ms, _mig_n)
            except Exception:
                pass
            return

        # Legacy layer-wise GET/PUT path.
        # Only consumer/decode loads KV Cache.
        if self.is_producer:
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        def inject_kv_into_layer(
            layer: torch.Tensor,
            kv_cache: torch.Tensor,
            block_ids: torch.Tensor,
            request_id: str,
        ) -> None:
            """
            Inject KV cache data into a given attention layer tensor.

            This function updates `layer` in-place with values from `kv_cache`,
            handling different backend layouts:
              - MLA (Multi-Linear Attention) or FlashInfer: KV tensors are
                indexed along the first dimension.
              - FlashAttention: KV tensors are indexed along the second
                dimension.

            If the number of provided block IDs does not match the number of KV
            blocks, only the overlapping portion is updated, and a warning is
            logged.

            Args:
                layer (torch.Tensor): The attention layer KV tensor to update.
                kv_cache (torch.Tensor): The KV cache tensor to inject.
                block_ids (torch.Tensor): Indices of the blocks to update.
                request_id (str): Request identifier used for logging.

            Returns:
                None. The function modifies `layer` in-place.
            """
            if (isinstance(attn_metadata, MLACommonMetadata)
                    or layer.shape[1] == 2):  # MLA or FlashInfer
                num_block = kv_cache.shape[0]
                self.check_tensors_except_dim(layer, kv_cache, 0)
                if len(block_ids) == num_block:
                    layer[block_ids, ...] = kv_cache
                else:
                    layer[block_ids[:num_block], ...] = kv_cache
                    logger.warning(
                        "🚧kv_cache does not match, block_ids:%d, "
                        "num_block:%d, request_id:%s", len(block_ids),
                        num_block, request_id)

            elif layer.shape[0] == 2:  # FlashAttention
                num_block = kv_cache.shape[1]
                self.check_tensors_except_dim(layer, kv_cache, 1)
                if len(block_ids) == num_block:
                    layer[:, block_ids, ...] = kv_cache
                else:
                    layer[:, block_ids[:num_block], ...] = kv_cache
                    logger.warning(
                        "🚧kv_cache does not match, block_ids:%d, "
                        "num_block:%d, request_id:%s", len(block_ids),
                        num_block, request_id)

        # Load the KV for each request each layer
        for request in metadata.requests:
            ip, port = self.parse_request_id(request.request_id,
                                             is_prefill=False)
            remote_address = ip + ":" + str(port + self._rank)
            _rt0 = time.time()
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                # Only process layers that have kv_cache
                # attribute (attention layers) Skip non-attention
                # layers like FusedMoE
                kv_cache = getattr(layer, 'kv_cache', None)
                if kv_cache is None:
                    continue

                layer = kv_cache[forward_context.virtual_engine]

                kv_cache = self.p2p_nccl_engine.recv_tensor(
                    request.request_id + "#" + layer_name,
                    remote_address=remote_address)

                if kv_cache is None:
                    raise RuntimeError(
                        "Missing remote KV cache for request "
                        f"{request.request_id}, layer {layer_name} from "
                        f"{remote_address}")

                inject_kv_into_layer(layer, kv_cache, request.block_ids,
                                     request.request_id)
            # ★ RECV-BLOCK: 这个请求逐层 NCCL 收 KV 用了多久。>1s = decode 在
            # recv_tensor 上阻塞等 prefill 送 KV (exec stall 的真凶)。
            _rt = (time.time() - _rt0) * 1000.0
            if _rt > 1000.0:
                logger.warning(
                    "RECV-BLOCK req=%s recv_kv=%.0fms (decode 在 NCCL 收 KV 上"
                    "阻塞, 等 prefill 送)", request.request_id, _rt)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        Used for LICHT round-kv layer-wise pipelining: on the prefill
        (producer) side, this waits until the background driver has loaded
        this layer's reused prefix, so attention reads valid KV while later
        layers keep loading.  Cheap no-op when round-kv pipelining is off or
        this forward has nothing to load.

        Args:
            layer_name: the name of that layer
        """
        if (self._round_kv_enabled and self.is_producer
                and self._round_store_obj is not None):
            self._round_store_obj.wait_layer(layer_name)
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        if self.direct_block_mode:
            return

        # Only producer/prefill saves KV Cache
        if not self.is_producer:
            return

        assert self.p2p_nccl_engine is not None

        def extract_kv_from_layer(
            layer: torch.Tensor,
            block_ids: torch.Tensor,
        ) -> torch.Tensor:
            """
            Extract KV cache slices from a given attention layer tensor.

            This function handles multiple backend layouts:
              - MLA (Multi-Linear Attention) or FlashInfer: KV tensors are
                indexed along the first dimension.
              - FlashAttention: KV tensors are indexed along the second
                dimension.

            Args:
                layer (torch.Tensor): The KV cache from the attention layer.
                block_ids (torch.Tensor): Indices of blocks to extract.

            Returns:
                torch.Tensor: A tensor containing the extracted KV slices.
                Returns None if the layout is unsupported.
            """
            if (isinstance(attn_metadata, MLACommonMetadata)
                    or layer.shape[1] == 2):  # MLA or FlashInfer
                return layer[block_ids, ...]

            if layer.shape[0] == 2:  # FlashAttention
                return layer[:, block_ids, ...]

            return None

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, P2pNcclConnectorMetadata)
        for request in connector_metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, True)
            remote_address = ip + ":" + str(port + self._rank)

            kv_cache = extract_kv_from_layer(kv_layer, request.block_ids)
            ok = self.p2p_nccl_engine.send_tensor(request_id + "#" +
                                                  layer_name, kv_cache,
                                                  remote_address)
            if not ok:
                raise RuntimeError(
                    "Failed to stage KV cache for request "
                    f"{request_id}, layer {layer_name}")

    def wait_for_save(self):
        if self.is_producer:
            assert self.p2p_nccl_engine is not None
            if self.direct_block_mode:
                # ★ 逐层流水: forward 各层已 wait 过各自 copy event, 此处收尾——
                # post-load gen 校验 + unpin (no-op if 本步没流水加载).
                if (self._round_kv_enabled
                        and self._round_store_obj is not None):
                    try:
                        self._round_store_obj.finish_pipelined()
                    except Exception as e:  # pragma: no cover
                        logger.warning("round-kv finish_pipelined failed: %s", e)
                # Pure decode-pull mode: only stage bridge metadata
                # locally.  Decode will fetch via BRIDGE_POP RPC in its
                # own forward step.
                _nreq = len(self._pending_bridge_reqs)
                t_save_start = time.time()
                for request_id, context_block_ids in self._pending_bridge_reqs:
                    self.p2p_nccl_engine.stage_bridge_request(
                        request_id, context_block_ids)
                self._pending_bridge_reqs.clear()
                # Per-step phase breakdown (producer): idle | load | fwd |
                # bridge.  load = time the engine was STUCK in start_load_kv
                # (round-kv synchronous load).  Confirms the stall source.
                t_save_end = time.time()
                ls, le = self._step_t_load_start, self._step_t_load_end
                # Only log steps that actually forwarded requests — skip the
                # frequent idle steps (reqs=0) that spam the log.
                import os as _os
                if (ls is not None and le is not None and _nreq > 0
                        and _os.environ.get("LICHT_STEP_PROFILE") == "1"):
                    idle = ((ls - self._step_prev_save_end) * 1000.0
                            if self._step_prev_save_end is not None else 0.0)
                    logger.info(
                        "round-kv STEP: reqs=%d idle_ms=%.0f load_ms=%.0f "
                        "fwd_ms=%.0f bridge_ms=%.0f",
                        _nreq, idle, (le - ls) * 1000.0,
                        (t_save_start - le) * 1000.0,
                        (t_save_end - t_save_start) * 1000.0)
                self._step_prev_save_end = t_save_end
                return
            self.p2p_nccl_engine.wait_for_sent()

    def _handle_arena_sink_rpc(self, request_id: str, job_id: str,
                               prompt_token_ids: list,
                               decode_zmq_address: str) -> bool:
        """Phase 2: producer-side router-thread callback for the
        ARENA_SINK RPC.  Pulls the request's block_ids out of
        bridge_queue (set by stage_bridge_request earlier), kicks off
        the async D2H to the round-kv arena, and remembers the
        request so the _round_store_obj done-hook can fast-release it
        once D2H finishes.  Runs in the engine's router thread —
        must not block."""
        if not self.is_producer:
            return False
        if not (self._round_kv_enabled
                and self._round_store_obj is not None):
            return False
        engine = self.p2p_nccl_engine
        if engine is None:
            return False
        with engine.state_lock:
            block_ids = engine.bridge_queue.pop(request_id, None)
        if not block_ids:
            # Bridge metadata wasn't staged for this req on us — race
            # (RELEASE already happened) or wrong target.  Tell the
            # decode side to fall back.
            logger.warning(
                "ARENA_SINK miss in bridge_queue req=%s (already "
                "released or wrong target)", request_id)
            return False
        try:
            self._arena_sink_pending.add(request_id)
            self._arena_sink_decode_addr[request_id] = decode_zmq_address
            # ★ 在途保护: sink 即给整个 job 打 .inflight 标记 → 淘汰跳过整 job,
            #   这一轮的 KV(及前缀)在 decode admit 拉走前不会被淘。admit /
            #   request_finished 时 clear。幂等/跨进程/不碰 pin 字段。
            #   必须在 enqueue_store 之【前】mark: 写线程一开工 job 就已在途,
            #   不给淘汰留 "已开始写但还没 mark" 的窗口。(decode 在 break->sink
            #   时也已本地 mark 过 — 这里是幂等加固。)
            self._round_store_obj.mark_inflight(str(job_id))
            # sink=True: 写失败/复核有洞 → 保留 GPU 块带退避重试, 直到整份
            # 可见才 fast-release; 请求挂掉 (.inflight 被 decode 清) 才放弃。
            self._round_store_obj.enqueue_store(
                str(job_id), list(block_ids),
                list(prompt_token_ids), request_id, sink=True)
            logger.info(
                "ARENA_SINK enqueued req=%s job=%s nblk=%d decode=%s",
                request_id, job_id, len(block_ids), decode_zmq_address)
            return True
        except Exception as e:  # pragma: no cover
            logger.warning(
                "ARENA_SINK enqueue_store failed req=%s: %s",
                request_id, e)
            self._arena_sink_pending.discard(request_id)
            self._arena_sink_decode_addr.pop(request_id, None)
            return False

    def _on_round_store_done(self, request_id: str) -> None:
        """Phase 2: round-kv-store done-hook.  When an ARENA_SINK
        request's D2H finishes, mark its prefill GPU blocks for
        release via the producer fast-release queue (mirrors the
        RELEASE-RPC path at p2p_nccl_engine.py:984-1000).  Runs in
        the store-pool thread — must be cheap and exception-safe.

        Must also set `release_received_ts` on the engine's
        `_delay_free_ts[req_id]` ledger: the engine's RELEASE-timeout
        detector (engine.py:1248-1280) reads that field to decide
        whether to clear `pending_release_deadlines`.  Without it
        every ARENA_SINK req trips the 600s timeout warning + an
        unnecessary force-free (functionally a no-op since the
        blocks are already released via fast-release-queue drain,
        but ledger goes inconsistent and the WARNING is alarming)."""
        if request_id not in self._arena_sink_pending:
            return
        self._arena_sink_pending.discard(request_id)
        decode_addr = self._arena_sink_decode_addr.pop(request_id, "")
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.p2p.\
p2p_nccl_engine import get_fast_release_queue
            q = get_fast_release_queue()
            if q is None:
                return
            now = time.time()
            engine = self.p2p_nccl_engine
            if engine is not None:
                # Mirror RELEASE-RPC handler: set release_received_ts
                # so pending_release_deadlines clears + push the
                # ts_entry (NOT a fresh dict) onto the queue so the
                # scheduler's drain sees the consistent record.
                ts_entry = engine._delay_free_ts.setdefault(
                    request_id, {})
                ts_entry["release_received_ts"] = now
                ts_entry["arena_sink_done_ts"] = now
                q.put_nowait((request_id, ts_entry.copy()))
            else:
                q.put_nowait(
                    (request_id, {"arena_sink_done_ts": now}))
            logger.info(
                "ARENA_SINK D2H done req=%s decode=%s -> "
                "fast-release", request_id, decode_addr)
        except Exception as e:  # pragma: no cover
            logger.warning(
                "ARENA_SINK done-hook fast-release failed req=%s: %s",
                request_id, e)

    def get_finished(
            self, finished_req_ids: set[str],
            **kwargs) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        assert self.p2p_nccl_engine is not None

        no_compile_layers = (
            self._vllm_config.compilation_config.static_forward_context)
        sending, recving = self.p2p_nccl_engine.get_finished(
            finished_req_ids, no_compile_layers)
        # LICHT round-kv: report decode-store completions as finished
        # sends so the scheduler frees the delay-free retained blocks.
        if self._round_kv_enabled and self._round_store_obj is not None:
            done = self._round_store_obj.drain_done()
            if done:
                sending = (sending or set()) | done
            # ASYNC load completions -> recving, so the scheduler moves the
            # request from WAITING_FOR_REMOTE_KVS to running (KV now filled).
            if self._round_async:
                loaded = self._round_store_obj.drain_loaded()
                if loaded:
                    recving = (recving or set()) | loaded
        return sending, recving

    def pop_delay_free_timestamps(
            self, req_ids: set[str]) -> dict[str, dict[str, float]]:
        if self.p2p_nccl_engine is not None:
            return self.p2p_nccl_engine.pop_delay_free_timestamps(req_ids)
        return {}

    @staticmethod
    def poll_fast_releases(
            timeout: float = 0.0) -> list[tuple[str, dict[str, float]]]:
        """Drain the fast-release queue (scheduler-side, Change 3).

        Returns a list of (request_id, timestamps) for requests whose
        RELEASE has been received by the listener thread.

        timeout=0.0 (default): non-blocking drain — original behaviour, used
        by the main-thread fallback path (_poll_fast_releases Path B).
        timeout>0: block up to `timeout` seconds for the FIRST item (true OS
        sleep, GIL fully released while waiting), then drain the rest
        non-blocking.  Used by _bg_free_loop so the bg thread sleeps instead
        of busy-polling when idle (Fix B: replaces Fix A's poll+1ms-sleep,
        eliminating the ~1000/s idle wakeups that still briefly grabbed the
        GIL).
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
            get_fast_release_queue)
        q = get_fast_release_queue()
        if q is None:
            return []
        released: list[tuple[str, dict[str, float]]] = []
        if timeout > 0:
            # Block for the first item so the thread truly sleeps when idle
            # (SimpleQueue.get raises queue.Empty on timeout → return empty).
            try:
                released.append(q.get(timeout=timeout))
            except Exception:
                return []
        while True:
            try:
                item = q.get_nowait()
                released.append(item)
            except Exception:
                break
        return released

    # ==============================
    # Scheduler-side methods
    # ==============================

    def mark_finished_job(self, job_id: str) -> None:
        """Phase A: scheduler tells us a trajectory's last step has
        finished — the job's KV history is no longer needed.  We:
        (1) call our scheduler-side _round_store_obj.mark_finished
            (deletes manifest on shared FS — also stops lookup() from
            returning hits across all processes), and
        (2) record the job_id so build_connector_meta can broadcast it
            via meta.finished_jobs.  Each worker then updates its own
            in-memory _finished_jobs + reverse index so _arena_alloc
            no longer protects this job's head increment."""
        if not job_id:
            return
        jid = str(job_id)
        if self._round_store_obj is not None:
            try:
                self._round_store_obj.mark_finished(jid)
            except Exception as e:  # pragma: no cover
                logger.debug("mark_finished sched-side failed job=%s: %s",
                             jid, e)
        self._pending_finish_jobs.add(jid)
        logger.info("PhaseA mark_finished_job job=%s (manifest deleted, "
                    "broadcast to worker via meta.finished_jobs)", jid)

    def set_admission_kv_usage(self, usage: float,
                               num_total_blocks: int = 0) -> None:
        """Phase 2: scheduler publishes (occupancy, total GPU blocks)
        once per admission pass.  The path-selector in
        get_num_new_matched_tokens uses them to project post-admit
        occupancy (FCFS: each admit in this pass already shows up in
        `usage` next time around).  No-op when the gate env is off.
        Cheap: two assignments; safe to call every step.
        """
        try:
            self._admission_kv_usage = float(usage)
        except (TypeError, ValueError):
            pass
        try:
            self._admission_kv_total_blocks = int(num_total_blocks)
        except (TypeError, ValueError):
            pass

    def save_preempt(self, request: "Request", block_ids: list,
                     all_token_ids: list) -> bool:
        """Phase 1: register a preempted decode request's KV for an
        arena save.  Called by the scheduler from the preempt path
        BEFORE kv_cache_manager.free().  The actual gather + D2H runs
        on the worker in start_load_kv THIS SAME STEP (synchronously,
        so the gather finishes before forward writes the slots).

        Returns True iff the save was scheduled (caller should remember
        the request as "kv lives in arena").  False -> caller proceeds
        with normal recompute path (no functional change vs current
        behaviour, just no save attempted).

        Process-boundary note: the connector's `_round_store_obj` is
        only `bind_kv_caches`-ed on the worker process — the gather
        cannot run here on the scheduler.  We only record state; the
        worker drains `meta.preempt_store` in start_load_kv.
        """
        if not self._phase1_save_on_preempt:
            return False
        # Only decode side preempts (producer prefill doesn't run decode).
        if self.is_producer:
            return False
        if not self._round_kv_enabled:
            return False
        job_id = getattr(request, "job_id", None)
        if not job_id:
            return False
        try:
            self._pending_preempt_store[request.request_id] = (
                str(job_id), list(block_ids), list(all_token_ids))
            self._preempt_saved[request.request_id] = str(job_id)
            # ★ 在途保护: preempt 存档即给整个 job 打 .inflight → 淘汰跳过整 job,
            #   存档(prompt+已生成 output)在下次 admit 拉走前不被淘。re-admit /
            #   request_finished 时 clear。和 ARENA_SINK 共用一套标记。
            if self._round_store_obj is not None:
                self._round_store_obj.mark_inflight(str(job_id))
        except Exception as e:  # pragma: no cover
            logger.warning("Phase1 save_preempt record failed req=%s: %s",
                           request.request_id, e)
            return False
        logger.info("Phase1 save_preempt enqueued req=%s job=%s nblk=%d",
                    request.request_id, job_id, len(block_ids))
        return True

    def _rk_lookup_cached(self, request: "Request"):
        """[Stage 6d perf] 按 request_id 缓存 lookup_resolve 结果.

        同一请求在一步内被 get_num + update_state 各查一次 (2×), 且未准入时跨步
        反复 probe —— 这里首次算, 之后全 O(1) 复用 (含 None=miss 也缓存).

        正确性: result 跨步复用即使 slot 后被淘, load 末尾 try_pin 用 entry.gen
        校验 fail-closed → miss 重算, 绝不读错 (就是日志里的 fail=N). prompt 对
        request_id 固定, 故按 request_id 缓存正确. 每步 build_connector_meta 丢弃
        本步未再 probe 的条目, 不泄漏.
        """
        rid = request.request_id
        self._rk_lk_seen.add(rid)
        job_id = getattr(request, "job_id", None)
        if rid in self._rk_lk_cache:
            entry = self._rk_lk_cache[rid]
        else:
            if not job_id or self._round_store_obj is None:
                self._rk_lk_cache[rid] = None
                return None
            _t = time.time()
            # ★ P2 修 (当步现探): 缓存只存【贵且稳定】的部分 —— CPU resolve
            # 结果 + SSD 尾段 hash (token 不变 hash 永远有效). SSD 的探测
            # 结果【绝不】跨步缓存 (2026-07-03 复盘: 旧版整包缓存 -> admit
            # 时拿几分钟前的 SSD 地图 -> 74 次 promote 秒败).
            base = self._round_store_obj.lookup_resolve(
                str(job_id), request.prompt_token_ids)
            tail = None
            try:
                _mb = base[1] if base is not None else 0
                tail = self._round_store_obj.ssd_tail_hashes(
                    request.prompt_token_ids, _mb)
            except Exception:
                tail = None
            # [PROBE] 只在真正算 (cache miss) 时累计 lookup 耗时/次数.
            self._sched_lk_ms = getattr(self, "_sched_lk_ms", 0.0) \
                + (time.time() - _t) * 1000.0
            self._sched_lk_n = getattr(self, "_sched_lk_n", 0) + 1
            entry = ((base, tail)
                     if (base is not None or tail) else None)
            self._rk_lk_cache[rid] = entry
        if entry is None:
            return None
        base, tail = entry
        # SSD 段: 每步现探一次 (µs 级 C 原子读), 步内共用; 探中即挂 inflight
        # (在 ssd_probe_fresh 内) + 记入 _ssd_marked 供 finish 兜底清.
        seg = None
        if tail:
            _c = self._rk_ssd_seg.get(rid)
            if _c is not None and _c[0] == self._rk_step_no:
                seg = _c[1]
            else:
                try:
                    _mb = base[1] if base is not None else 0
                    # ★ 闸门按角色: producer(prefill) 走诚实收益闸门 (搬 vs
                    # 重算); consumer(decode Phase2 admission) 旁路 —— 那边
                    # 等不到数据的下场是客户端超时, 搬永远比死等划算.
                    # ★ 2026-07-05 (prefill inflight 久占修): claim 期【不挂】
                    # SSD inflight —— admit 前根本不搬, 挂着只是把 SSD 槽白锁
                    # 一整个排队期, 深队列下饿死 SSD 容量. 改为只在 admit 挂
                    # (update_state_after_alloc). 等待期腾出的槽让别的请求 KV
                    # 复用. 正确性: promote 时 pin+gen fail-closed 才防读错,
                    # inflight 只是保住复用的优化; 等待中槽被淘 → admit 现探返
                    # None → 该段退回重算 (安全, 只丢复用). Phase2(decode)不走
                    # 这条路, 它的 claim 挂保留 (defer 期后台真在搬).
                    seg = self._round_store_obj.ssd_probe_fresh(
                        str(job_id), _mb, tail,
                        apply_gate=self.is_producer,
                        mark_inflight=False)
                except Exception:
                    seg = None
                self._rk_ssd_seg[rid] = (self._rk_step_no, seg)
                # NB: 不在此记 _ssd_marked —— producer 只在 admit 挂 inflight,
                # 由 update_state_after_alloc 记 (供 request_finished 兜底清).
        if base is None and seg is None:
            return None
        mt, mb, sg = base if base is not None else (0, 0, None)
        if seg is not None:
            _a, _recs = seg
            _end = _a + len(_recs)
            return _end * self._block_size, _end, sg, seg
        return mt, mb, sg, None

    def _promote_ssd_segments(self, round_load) -> None:
        """★ P2 (worker, 引擎线程): load 分发前把各请求的 SSD 段搬回 CPU.

        成功: 跨 job 显式路径把 promote 返回的 (cpu_slot, gen) 接到
        src_slots/src_gens 尾部 (own-job 路径无需接 —— promote 以同 job 写入
        CPU 账本, own .slot 已自然覆盖 gap 全段).
        失败: 清空该请求的 ssd + src 字段 → 整个 gap 退回 own-.slot 路径并
        整体 fail-closed (per_item_ok=False, 同现有 fail=N 类; promote 侧
        已 error 级报警). 处理后抹掉 ssd 字段, 下游分发模式不感知 SSD."""
        for rl in round_load:
            if not rl.ssd_slots:
                continue
            recs = list(zip(rl.ssd_slots, rl.ssd_gens or [],
                            rl.ssd_hashes or []))
            sg = None
            if (len(recs) == len(rl.ssd_slots)
                    and self._round_store_obj is not None):
                try:
                    sg = self._round_store_obj.promote_from_ssd(
                        rl.job_id, int(rl.ssd_start), recs)
                except Exception as e:  # pragma: no cover
                    logger.error(
                        "promote_ssd_segments error req=%s job=%s: %s",
                        rl.request_id, str(rl.job_id)[:32], e)
            if sg is not None:
                if rl.src_slots:
                    rl.src_slots = (list(rl.src_slots)
                                    + [int(s) for (s, _g) in sg])
                    rl.src_gens = (list(rl.src_gens)
                                   + [int(g) for (_s, g) in sg])
                # src_slots 空/None = own-job: promote 已写进本 job 账本,
                # own .slot 覆盖, 无需显式列表.
            else:
                # 失败: gap 整体退 own-.slot (own incs 只到 SSD 段前 →
                # load_request 返 None → 整请求 fail-closed, 不留半截).
                rl.src_slots = None
                rl.src_gens = None
            rl.ssd_slots = None
            rl.ssd_gens = None
            rl.ssd_hashes = None

    def _rk_lookup_fresh_tiered(self, request):
        """★ Phase2 (decode admission) 专用查询: CPU 覆盖每次现算.

        与 _rk_lookup_cached 的区别 (2026-07-04 复盘): Phase2 在轮询 "sink
        写齐了没", CPU matched 必须反映当刻 (跨步缓存会把 matched 冻在首次
        探测值 -> PRESENT 却永远 SHORT). CPU resolve 走 C 快路径 (~1ms);
        贵的 Python tail hash 按 matched 边界缓存 (边界变才重算 O(prompt)).

        ★ 2026-07-05 定案: 返回的 matched 是【纯 CPU】口径 (SSD 不参与
        "齐" 的承诺 —— decode 永不重算, 失败模式只能是"多等"不能是"错").
        ssd_seg 仅作情报: 非 None = "CPU 的洞在 SSD 里" (冷期被降级的历史
        段), 调用方据此触发 defer 期后台修复搬运, CPU 真齐后才 admit.
        SSD 探测不走收益闸门 (死等无经济学), 探中挂 SSD inflight 保护
        修复窗口. 步内 get_num/update_state 共用同一结果.
        返回 4 元组 (mt_cpu, mb_cpu, sg, ssd_seg) 或 None."""
        rid = request.request_id
        _c = self._rk_ph2_res.get(rid)
        if _c is not None and _c[0] == self._rk_step_no:
            return _c[1]
        job_id = getattr(request, "job_id", None)
        if not job_id or self._round_store_obj is None:
            return None
        base = self._round_store_obj.lookup_resolve(
            str(job_id), request.prompt_token_ids)
        mb = base[1] if base is not None else 0
        seg = None
        try:
            _t = self._rk_ph2_tail.get(rid)
            if _t is None or _t[0] != mb:
                tail = self._round_store_obj.ssd_tail_hashes(
                    request.prompt_token_ids, mb)
                self._rk_ph2_tail[rid] = (mb, tail)
            else:
                tail = _t[1]
            if tail:
                seg = self._round_store_obj.ssd_probe_fresh(
                    str(job_id), mb, tail, apply_gate=False)
                if seg is not None:
                    self._ssd_marked[rid] = str(job_id)
        except Exception:
            seg = None
        if base is None and seg is None:
            res = None
        else:
            mt, _mb2, sg = base if base is not None else (0, 0, None)
            res = (mt, mb, sg, seg)   # matched = 纯 CPU; seg 仅情报
        self._rk_ph2_res[rid] = (self._rk_step_no, res)
        return res

    def _ph2_schedule_promote(self, rid: str, job_id: str, seg) -> None:
        """把 Phase2 defer 期的修复搬运排进本步元数据 (幂等 + 节流).

        节流 25 步重试一次: 搬运本身幂等 (内容寻址, 重复搬 = dedup HIT 零
        I/O), 节流只为省元数据带宽. rid 在 admit / request_finished 时清."""
        _p = self._ph2_promote_pending.get(rid)
        if _p is not None and self._rk_step_no - _p < 25:
            return
        self._ph2_promote_pending[rid] = self._rk_step_no
        _a, recs = seg
        self._ph2_promote_q.append(
            (rid, str(job_id), int(_a),
             [int(s) for (s, _g, _h) in recs],
             [int(g) for (_s, g, _h) in recs],
             [int(h) for (_s, _g, h) in recs]))

    def mark_arena_sink(self, request) -> bool:
        """★ 改动1/2: scheduler 在 waiting 循环【80% break】后, 对 break 那个 + 它后面
        没扫到的【还没 sink 的】请求调这个 → 标记 sink + 排进 _pending_arena_sink
        (build_connector_meta 会发 RPC 让 prefill 把这个请求的 KV 下沉 arena + 放掉
        prefill GPU)。已 sink / preempt-saved / 已 pending / producer 的跳过。
        返回是否【新】标记 (供日志统计)。"""
        if self.is_producer:
            return False
        if not (self._phase2_admission_gate and self._round_kv_enabled
                and self._round_store_obj is not None):
            return False
        rid = request.request_id
        if (rid in self._arena_sinked or rid in self._preempt_saved
                or rid in self._pending_arena_sink):
            return False
        _remote = None
        _kvp = getattr(request, "kv_transfer_params", None)
        if isinstance(_kvp, dict):
            _remote = _kvp.get("prefill_zmq_address")
        if _remote is None:
            try:
                _ip, _port = self.parse_request_id(
                    request.request_id, is_prefill=False)
                _remote = f"{_ip}:{_port}"
            except Exception:
                _remote = None
        _job = getattr(request, "job_id", None) or ""
        self._pending_arena_sink[rid] = (
            str(_job), list(request.prompt_token_ids), _remote or "")
        self._arena_sinked.add(rid)
        self._arena_sink_ts[rid] = time.time()
        # ★ 竞态修复 (moto8386 死法): 在 decode【决定 sink 的当下】就本地打
        #   .inflight (共享 /dev/shm, 两侧 evictor 都看得到), 不等 ARENA_SINK
        #   RPC 到 prefill 再 mark —— 原来这段空窗里 decode 自己的 evictor 可
        #   把该 job 已存的前缀淘掉, 而 prefill 按 manifest 只补尾段 → 永久洞。
        #   清除路径不变 (admit 拉走 / 请求挂掉时 clear), 幂等。
        if (_job and self._round_kv_enabled
                and self._round_store_obj is not None):
            try:
                self._round_store_obj.mark_inflight(str(_job))
            except Exception:  # pragma: no cover
                pass
        logger.info(
            "Phase2 break->sink req=%s job=%s remote=%s ntoks=%d",
            rid, _job, _remote or "<none>", len(request.prompt_token_ids))
        return True

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.is_producer:
            # LICHT round-kv: a returning request can reuse its prior-round
            # KV from the CPU/SSD store.  Synchronous load (load_async=
            # False) → blocks allocated + filled this step before forward.
            if self._round_kv_enabled:
                job_id = getattr(request, "job_id", None)
                if job_id:
                    # ★ Stage 6d: lookup_resolve (content-addr 表驱动, job 无关)
                    # 算可复用 token 数. 与 update_state_after_alloc 用同一口径
                    # (同一缓存), 否则全新 job own-job lookup 返 0 → 不分配 → 跨 job
                    # 命中白费. [perf] 走 _rk_lookup_cached: 同请求只算一次.
                    res = self._rk_lookup_cached(request)
                    if res is not None:
                        matched_tokens, _mb, _sg, _ssd = res
                        ext = matched_tokens - num_computed_tokens
                        if ext > 0:
                            # async (True): park in WAITING_FOR_REMOTE_KVS,
                            # bg load, engine free.  sync (False): old path.
                            return ext, self._round_async
            return 0, False

        # Phase 2 (PD path selector): if this request was previously
        # routed through the arena (RPC fired in an earlier pass), the
        # connector waits here for the prefill side to finish D2H
        # writing.  ★ 改动3: 只有 arena 把它【这一轮的完整 KV】都写齐了才 admit;
        # 部分命中【不】admit (否则缺的块在 decode 上重算、吃满 256 token budget,
        # 拖死生成). 没齐就一直 defer 等下轮; 删除原 30s deadline → 重算兜底:
        # decode 永不重算, 等不到就让客户端 300s 超时把请求挂掉 (用户决定).
        if (self._phase2_admission_gate
                and request.request_id in self._arena_sinked
                and self._round_kv_enabled
                and self._round_store_obj is not None):
            job_id = getattr(request, "job_id", None)
            if job_id:
                # ★ P2 修 (2026-07-04 复盘, decode3 卡死请求): 覆盖判定改两层
                # 拼接 —— CPU 淘掉但已降级进 SSD 的块算"在" (admit 后 worker
                # 在 load 前 promote 回 CPU). 原 CPU-only lookup 会对
                # "CPU 有洞 + 洞在 SSD" 的请求永远 defer 到客户端超时.
                # ★★ 必须用 fresh 版 (18:17 复盘): Phase2 在轮询 sink 进度,
                # _rk_lookup_cached 的跨步缓存会把 matched 冻在首次值 ->
                # PRESENT 却永远 SHORT. fresh 版 CPU 每步现算 + SSD 不走
                # 收益闸门 + 探中挂 SSD inflight.
                res = self._rk_lookup_fresh_tiered(request)
                # 整份齐 = 两层命中块覆盖整个 prompt 的完整块数.
                _need_blk = (len(request.prompt_token_ids)
                             // self._block_size)
                _mb = res[1] if res is not None else -1
                _ssd_blk = (len(res[3][1])
                            if (res is not None and res[3] is not None)
                            else 0)
                # ★ 诊断(每请求每 5s 一条): 区分 sink 请求死法 —
                #   一直 matched<need → 数据缺(被淘/没写齐) = pin/淘汰问题;
                #   一直 matched>=need 却没 admit → 数据在却 admit 不上 = 调度问题。
                _now = time.time()
                if _now - self._arena_probe_log_ts.get(
                        request.request_id, 0.0) > 5.0:
                    self._arena_probe_log_ts[request.request_id] = _now
                    _verdict = ("MISS" if res is None else
                                ("FULL" if _mb >= _need_blk else "SHORT"))
                    logger.info(
                        "Phase2 arena-probe req=%s job=%s matched_blk=%d "
                        "(ssd=%d) need=%d verdict=%s", request.request_id,
                        str(job_id)[:40], _mb, _ssd_blk, _need_blk, _verdict)
                    # ★ 诊断: SHORT 时查清 block[matched] 到底为啥 miss
                    #   (ht_miss/unpublished/gen_mismatch/refcnt0/PRESENT + 这个 job
                    #    自己覆盖没覆盖这块) → 一眼锁定真因, 不再猜。
                    if _verdict == "SHORT" and _mb >= 0:
                        try:
                            _why = self._round_store_obj.probe_block_reason(
                                str(job_id), request.prompt_token_ids, _mb)
                            logger.info(
                                "Phase2 SHORT-WHY req=%s job=%s break_block=%d "
                                "%s", request.request_id, str(job_id)[:40],
                                _mb, _why)
                        except Exception as _pe:
                            logger.info("Phase2 SHORT-WHY probe failed req=%s: "
                                        "%s", request.request_id, _pe)
                if res is not None:
                    matched_tokens = res[0]
                    if _mb >= _need_blk:   # ★ 纯 CPU 口径的"齐" (2026-07-05)
                        ext = matched_tokens - num_computed_tokens
                        return max(0, ext), False   # 齐 → admit-from-arena
                    if res[3] is not None:
                        # ★ defer 期后台修复: CPU 的洞在 SSD (冷期被降级的
                        # 历史段) → 交给 worker 后台搬回 CPU; 本步照旧 defer,
                        # CPU 真齐后才 admit. 失败 = 继续等 + 自动重试,
                        # 结构上不可能产生垃圾 (decode 永不重算的铁律).
                        self._ph2_schedule_promote(
                            request.request_id, str(job_id), res[3])
            # 没齐 (这一轮的块还没 sink 完 / 被淘) → defer 等下轮, 永不重算.
            return None, False

        # ★ 改动1/2: 原来这里有个 "first-time 80% 投影门"(>80% 就标 sink+defer)。
        # 已挪到 scheduler 的 waiting 循环 (改动2A: 对所有请求投影; 改动1: 第一个超
        # 80% 就 break, break 后调 mark_arena_sink 把没扫到的请求 sink)。所以这里
        # 不再做 sink 决策 —— 没 sink 过的请求直接落到下面 NCCL 路 (从 prefill 拉)。

        # Phase 1 (save-on-preempt): if this request was preempt-saved on
        # this decode instance, its KV lives in the arena — recover from
        # arena instead of trying to NCCL-pull from producer (which has
        # long since freed its KV).  Lookup uses prompt + already-
        # generated outputs since save_preempt persisted that full prefix.
        if (self._phase1_save_on_preempt
                and request.request_id in self._preempt_saved
                and self._round_kv_enabled
                and self._round_store_obj is not None):
            job_id = self._preempt_saved.get(request.request_id)
            try:
                all_tids = list(request.prompt_token_ids) + list(
                    request.output_token_ids)
            except AttributeError:
                all_tids = list(request.prompt_token_ids)
            if job_id:
                res = self._round_store_obj.lookup(str(job_id), all_tids)
                if res is not None:
                    matched_tokens, matched_blocks = res
                    # ★ 改动3(preempt-save 分支, 与 arena-sink 分支一致):
                    # 只有存档【整份齐】(prompt+已生成 output 的完整块都在 arena)
                    # 才 admit; 没齐就 defer 等下轮, 永不 fall-through 重算。pin
                    # (.inflight) 保证存档不被淘 → 终会凑齐。存档还没落完 / 跨步
                    # 被淘时, 这一步就先等。
                    _need_blk = len(all_tids) // self._block_size
                    if matched_blocks >= _need_blk:
                        ext = matched_tokens - num_computed_tokens
                        # Sync arena load on resume (small increment;
                        # avoids parking in WAITING_FOR_REMOTE_KVS).
                        return max(0, ext), False   # 齐 → admit-from-arena
            # 没齐 (存档没落完 / 被淘) → defer 等下轮, 不重算。
            return None, False

        num_external_tokens = (len(request.prompt_token_ids) - 1 -
                               num_computed_tokens)

        if num_external_tokens < 0:
            num_external_tokens = 0

        if self.direct_block_mode and num_external_tokens > 0:
            return num_external_tokens, True

        return num_external_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        if not self.is_producer and num_external_tokens > 0:
            # Phase 2 (PD path selector) consumer recovery: this request
            # was previously routed through arena (RPC fired earlier).
            # get_num_new_matched_tokens just confirmed arena has data
            # (otherwise it would have returned None to defer).  Route
            # to round-load like the prefix-reuse / preempt-recovery
            # paths instead of NCCL pull.
            if (self._phase2_admission_gate
                    and request.request_id in self._arena_sinked
                    and self._round_kv_enabled
                    and self._round_store_obj is not None):
                job_id = getattr(request, "job_id", None)
                if job_id:
                    # ★ P2 修 (2026-07-05 定案): 与 get_num 同一步共用同一
                    # fresh 结果 (步内 memo); matched 是【纯 CPU】口径 ——
                    # admit 只在 CPU 真齐时发生, load 永远只读 CPU (SSD 段
                    # 已在 defer 期由后台修复搬回). sg 显式传 (表驱动匹配
                    # 可能超出 own-.slot 覆盖).
                    res = self._rk_lookup_fresh_tiered(request)
                    if res is not None:
                        _matched, matched_blocks, slot_gen, _ssd_seg = res
                        num_blocks = num_external_tokens // self._block_size
                        local_hit_blocks = matched_blocks - num_blocks
                        block_ids0 = blocks.get_block_ids()[0]
                        if (num_blocks > 0 and local_hit_blocks >= 0
                                and len(block_ids0)
                                >= local_hit_blocks + num_blocks):
                            dst = list(block_ids0)[
                                local_hit_blocks:
                                local_hit_blocks + num_blocks]
                            sg = None
                            if slot_gen is not None:
                                sg = slot_gen[
                                    local_hit_blocks:
                                    local_hit_blocks + num_blocks]
                                if len(sg) != num_blocks:
                                    sg = None   # 解析不足, 退 own-.slot
                            self._round_load_reqs[
                                request.request_id] = (
                                    str(job_id), dst, local_hit_blocks, sg)
                            self._arena_sinked.discard(
                                request.request_id)
                            self._arena_sink_ts.pop(
                                request.request_id, None)
                            self._rk_ph2_tail.pop(request.request_id, None)
                            self._rk_ph2_res.pop(request.request_id, None)
                            self._ph2_promote_pending.pop(
                                request.request_id, None)
                            # ★ 在途保护: 已拉走 → 清 .inflight, job 重新可淘。
                            self._round_store_obj.clear_inflight(str(job_id))
                            logger.info(
                                "Phase2 admit-from-arena req=%s job=%s "
                                "matched_blocks=%d num_blocks=%d "
                                "local_hit=%d (decode reads arena)",
                                request.request_id, str(job_id)[:32],
                                matched_blocks, num_blocks, local_hit_blocks)
                            return
                # Lookup raced / failed -> fall through; drop the
                # marker so a retry doesn't get stuck in the arena
                # branch forever.
                self._arena_sinked.discard(request.request_id)
                self._arena_sink_ts.pop(request.request_id, None)
            # Phase 1 (save-on-preempt) consumer recovery: if this request
            # was preempt-saved, route to the arena-load path (matches the
            # producer's prefix-reuse flow) instead of the NCCL pull queue.
            if (self._phase1_save_on_preempt
                    and request.request_id in self._preempt_saved
                    and self._round_kv_enabled
                    and self._round_store_obj is not None):
                job_id = self._preempt_saved.get(request.request_id)
                try:
                    all_tids = list(request.prompt_token_ids) + list(
                        request.output_token_ids)
                except AttributeError:
                    all_tids = list(request.prompt_token_ids)
                if job_id:
                    res = self._round_store_obj.lookup(
                        str(job_id), all_tids)
                    if res is not None:
                        _matched, matched_blocks = res
                        num_blocks = num_external_tokens // self._block_size
                        local_hit_blocks = matched_blocks - num_blocks
                        block_ids0 = blocks.get_block_ids()[0]
                        if (num_blocks > 0 and local_hit_blocks >= 0
                                and len(block_ids0)
                                >= local_hit_blocks + num_blocks):
                            dst = list(block_ids0)[
                                local_hit_blocks:
                                local_hit_blocks + num_blocks]
                            self._round_load_reqs[
                                request.request_id] = (
                                    str(job_id), dst, local_hit_blocks, None)
                            # One-shot: arena → GPU, drop the marker.
                            self._preempt_saved.pop(
                                request.request_id, None)
                            # ★ 在途保护: 已拉走 → 清 .inflight, job 重新可淘。
                            self._round_store_obj.clear_inflight(str(job_id))
                            return
                # Arena recovery failed → fall through to NCCL path
                # (which will see 0 tokens on producer = recompute).
                self._preempt_saved.pop(request.request_id, None)
            self._requests_need_load[request.request_id] = (
                request, blocks.get_block_ids()[0])
            return
        # LICHT round-kv: producer/prefill reuse — record the EXACT prefix
        # blocks to fill before the forward.  The block list is
        # [local-cache-hit blocks ... | external (gap) blocks | new-token
        # blocks].  We must target only the gap blocks, at the right
        # offset, AND read the saved source from the same offset — else we
        # overwrite shared cached blocks and leave the prefix tail garbage.
        if (self.is_producer and self._round_kv_enabled
                and num_external_tokens > 0):
            job_id = getattr(request, "job_id", None)
            if job_id and self._round_store_obj is not None:
                # [perf] 复用 get_num 这步算过的同一结果 (按 request_id 缓存).
                res = self._rk_lookup_cached(request)
                if res is not None:
                    _matched_tokens, matched_blocks, slot_gen, ssd_seg = res
                    num_blocks = num_external_tokens // self._block_size
                    # local cache hit (in blocks) sits before the gap.
                    local_hit_blocks = matched_blocks - num_blocks
                    block_ids0 = blocks.get_block_ids()[0]
                    if (num_blocks > 0 and local_hit_blocks >= 0
                            and len(block_ids0)
                            >= local_hit_blocks + num_blocks):
                        dst = list(block_ids0)[
                            local_hit_blocks:local_hit_blocks + num_blocks]
                        g0 = local_hit_blocks       # gap 起点 (绝对块号)
                        gend = local_hit_blocks + num_blocks
                        # ★ P2: gap 拆两段 —— CPU 段 [g0, cpu_end) + SSD 段
                        # [max(ssd_a, g0), gend). ssd_a = CPU 匹配终点.
                        ssd_meta = None
                        cpu_end = gend
                        if ssd_seg is not None:
                            ssd_a, ssd_recs = ssd_seg
                            ssd_from = max(int(ssd_a), g0)
                            recs_sliced = list(
                                ssd_recs[ssd_from - int(ssd_a):])
                            # 一致性: SSD 段必须恰好补到 gap 末尾
                            if (recs_sliced
                                    and ssd_from + len(recs_sliced) == gend
                                    and ssd_from >= g0):
                                ssd_meta = (ssd_from, recs_sliced)
                                cpu_end = ssd_from
                                # ★ producer 唯一的 SSD inflight 挂点
                                # (2026-07-05): admit 确定要用这段才挂, 只需
                                # 撑过 admit->worker promote pin 的毫秒竞态窗口
                                # (等待期不挂, 见 _rk_lookup_cached). worker
                                # promote 后清; 请求挂掉由 request_finished 兜底.
                                self._round_store_obj.ssd_mark_inflight(
                                    str(job_id))
                                self._ssd_marked[request.request_id] = (
                                    str(job_id))
                        # ★ 跨 job: 把 lookup_resolve 解析的 slot 切到与 dst
                        # 对齐的 CPU 段 [g0, cpu_end).
                        sg = None
                        if slot_gen is not None:
                            sg = slot_gen[g0:cpu_end]
                            if len(sg) != cpu_end - g0:
                                sg = None   # 解析不足, 退回 own-.slot
                        self._round_load_reqs[request.request_id] = (
                            str(job_id), dst, local_hit_blocks, sg, ssd_meta)

    # 回传 REMOVED (2026-05-21): enqueue_v3_pushback / enqueue_v3_offload /
    # enqueue_v3_offload_release deleted.

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        meta = P2pNcclConnectorMetadata()
        # 回传 REMOVED: no v3 push-back/offload drain.

        # round-kv SCHED 指标: 每步在 arena lookup 上花的总时间/次数 (cache miss
        # 才计). 配合 build_meta 自身耗时, 监控 admission 循环里 lookup 的开销.
        _bcm_t0 = time.time()
        _lk_ms = getattr(self, "_sched_lk_ms", 0.0)
        _lk_n = getattr(self, "_sched_lk_n", 0)
        self._sched_lk_ms = 0.0
        self._sched_lk_n = 0
        # [perf] 丢弃本步未再 probe 的 lookup 缓存条目 (已准入/已结束的请求 →
        # 不在 _rk_lk_seen), 保留仍在反复 probe 的等待请求, 不泄漏.
        if self._rk_lk_cache:
            self._rk_lk_cache = {k: v for k, v in self._rk_lk_cache.items()
                                 if k in self._rk_lk_seen}
            # ★ P2 修: SSD 现探结果同策略清理 + 步号推进 (跨步作废)
            self._rk_ssd_seg = {k: v for k, v in self._rk_ssd_seg.items()
                                if k in self._rk_lk_seen}
        self._rk_step_no += 1
        self._rk_lk_seen.clear()

        # LICHT round-kv: drain scheduler-side reuse bookkeeping into the
        # metadata so the worker connector can load (prefill) / store
        # (decode) during the forward.  Done before the role-specific
        # early returns below so both paths carry it.
        if self._round_kv_enabled:
            for req_id, _v in self._round_load_reqs.items():
                # _v 为 (job_id, dst, src_offset) / (..., slot_gen) 4 元组 /
                # (..., ssd_meta) 5 元组 (★ P2 producer 路径).
                job_id, dst_block_ids, src_offset = _v[0], _v[1], _v[2]
                slot_gen = _v[3] if len(_v) > 3 else None
                src_slots = ([int(s) for (s, _g) in slot_gen]
                             if slot_gen else None)
                src_gens = ([int(g) for (_s, g) in slot_gen]
                            if slot_gen else None)
                # ★ P2: SSD 段 (ssd_from, [(slot,gen,hash),...]) -> 平行列表
                _ssd = _v[4] if len(_v) > 4 else None
                ssd_start = int(_ssd[0]) if _ssd else 0
                ssd_slots = ([int(s) for (s, _g, _h) in _ssd[1]]
                             if _ssd else None)
                ssd_gens = ([int(g) for (_s, g, _h) in _ssd[1]]
                            if _ssd else None)
                ssd_hashes = ([int(h) for (_s, _g, h) in _ssd[1]]
                              if _ssd else None)
                meta.round_load.append(RoundReqMeta(
                    request_id=req_id, job_id=job_id,
                    block_ids=list(dst_block_ids), token_ids=[],
                    num_blocks=len(dst_block_ids),
                    src_block_offset=src_offset,
                    src_slots=src_slots, src_gens=src_gens,
                    ssd_start=ssd_start, ssd_slots=ssd_slots,
                    ssd_gens=ssd_gens, ssd_hashes=ssd_hashes))
            self._round_load_reqs.clear()
            for req_id, (job_id, block_ids0, token_ids) in \
                    self._pending_round_store.items():
                meta.round_store.append(RoundReqMeta(
                    request_id=req_id, job_id=job_id,
                    block_ids=list(block_ids0), token_ids=list(token_ids),
                    num_blocks=0))
            self._pending_round_store.clear()
            # Phase 1 (save-on-preempt): drain pending preempt-victim
            # saves so the worker can run them in start_load_kv this
            # same step (before the new tenant overwrites the slots).
            if self._phase1_save_on_preempt and self._pending_preempt_store:
                for req_id, (job_id, block_ids0, token_ids) in \
                        self._pending_preempt_store.items():
                    meta.preempt_store.append(RoundReqMeta(
                        request_id=req_id, job_id=job_id,
                        block_ids=list(block_ids0),
                        token_ids=list(token_ids),
                        num_blocks=0))
                self._pending_preempt_store.clear()

        # Phase 2 (PD path selector): drain arena-sink RPCs so the
        # decode worker fires them in start_load_kv to the producer
        # (prefill) side via ZMQ.  Drained role-independent so both
        # sides see consistent meta (only consumer worker actually
        # acts).
        if self._phase2_admission_gate and self._pending_arena_sink:
            for req_id, (job_id, prompt_tids, remote_addr) in \
                    self._pending_arena_sink.items():
                meta.arena_sink.append(
                    (req_id, job_id, list(prompt_tids), remote_addr))
            self._pending_arena_sink.clear()
        # ★ P2 修 (2026-07-05): Phase2 defer 期修复搬运下发 worker
        if self._phase2_admission_gate and self._ph2_promote_q:
            meta.arena_promote.extend(self._ph2_promote_q)
            self._ph2_promote_q.clear()

        # Phase A: drain trace-finished job IDs.  Both producer and
        # consumer worker process these (each updates its own
        # _round_store_obj instance); drain is role-independent.
        if self._pending_finish_jobs:
            meta.finished_jobs.extend(self._pending_finish_jobs)
            self._pending_finish_jobs.clear()

        if not self.is_producer and self.direct_block_mode:
            for req_id, (request, local_block_ids) in \
                    self._requests_need_load.items():
                remote_prefill_address: Optional[str] = None
                remote_decode_address: Optional[str] = None

                kv_params = request.kv_transfer_params
                if isinstance(kv_params, dict):
                    remote_prefill_address = kv_params.get(
                        "prefill_zmq_address")
                    remote_decode_address = kv_params.get(
                        "decode_zmq_address")

                if remote_prefill_address is None:
                    try:
                        ip, port = self.parse_request_id(req_id,
                                                         is_prefill=False)
                        remote_prefill_address = f"{ip}:{port + self._rank}"
                    except Exception:
                        remote_prefill_address = None

                meta.add_request(
                    request_id=req_id,
                    token_ids=request.prompt_token_ids,
                    block_ids=local_block_ids,
                    block_size=self._block_size,
                    remote_prefill_address=remote_prefill_address,
                    remote_decode_address=remote_decode_address,
                )
            if _lk_n > 0:
                logger.info(
                    "round-kv SCHED: lookups=%d lookup_ms=%.1f "
                    "build_meta_ms=%.1f", _lk_n, _lk_ms,
                    (time.time() - _bcm_t0) * 1000.0)
            return meta

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[new_req.req_id]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                # the request's prompt is chunked prefill
                if num_tokens < len(new_req.prompt_token_ids):
                    # 'CachedRequestData' has no attribute 'prompt_token_ids'
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids[0], new_req.prompt_token_ids)
                    continue
                # the request's prompt is not chunked prefill
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size)
                continue
            if new_req.req_id in self._requests_need_load:
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size)
                self._requests_need_load.pop(new_req.req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = cached_reqs.resumed_from_preemption[i]

            if self.is_producer:
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[req_id]
                num_tokens = (num_scheduled_tokens + num_computed_tokens)
                if req_id not in self.chunked_prefill:
                    logger.warning(
                        "⚠️[PREFILL]Missing chunked_prefill state, req_id:%s, "
                        "resumed:%s", req_id, resumed_from_preemption)
                    continue

                if new_block_ids is None or not new_block_ids \
                        or new_block_ids[0] is None:
                    # No newly allocated block can be normal when the newly
                    # scheduled tokens still fit into existing blocks.
                    # For resumed requests we still need a full block list;
                    # keep waiting if scheduler cannot provide it.
                    if resumed_from_preemption:
                        logger.warning(
                            "⚠️[PREFILL]Resumed req has no new_block_ids, "
                            "req_id:%s, num_computed_tokens:%d", req_id,
                            num_computed_tokens)
                        continue
                    new_block_ids_0: list[int] = []
                else:
                    new_block_ids_0 = new_block_ids[0]

                block_ids = new_block_ids_0
                if not resumed_from_preemption:
                    block_ids = (self.chunked_prefill[req_id][0] + block_ids)
                prompt_token_ids = self.chunked_prefill[req_id][1]
                # the request's prompt is chunked prefill again
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids,
                                                    prompt_token_ids)
                    continue
                # the request's prompt is all prefilled finally
                meta.add_request(request_id=req_id,
                                 token_ids=prompt_token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size)
                self.chunked_prefill.pop(req_id, None)
                continue

            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                if new_block_ids is None or not new_block_ids \
                        or new_block_ids[0] is None:
                    logger.warning(
                        "⚠️[DECODE]No new_block_ids for resumed request, "
                        "req_id:%s, num_computed_tokens:%d", req_id,
                        num_computed_tokens)
                    continue

                request, _ = self._requests_need_load.pop(req_id)
                total_tokens = num_computed_tokens + 1
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                block_ids = new_block_ids[0]

                meta.add_request(request_id=req_id,
                                 token_ids=token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size)

        self._requests_need_load.clear()
        if _lk_n > 0:
            logger.info(
                "round-kv SCHED: lookups=%d lookup_ms=%.1f "
                "build_meta_ms=%.1f", _lk_n, _lk_ms,
                (time.time() - _bcm_t0) * 1000.0)
        return meta

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        # Keep retrying direct-mode bridge pop until worker reports recving
        # complete for each request.
        if self.is_producer or not self.direct_block_mode:
            return

        for req_id in (connector_output.finished_recving or ()):
            self._requests_need_load.pop(req_id, None)
            self._pending_failed_block_migrations.pop(req_id, None)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """

        self.chunked_prefill.pop(request.request_id, None)
        self._requests_need_load.pop(request.request_id, None)
        # ★ 在途保护(挂掉兜底): 若这个请求是在途的 arena-sink / preempt-save 且
        # 【没 admit 就结束】(还在标记集合里 = 排队 300s 超时挂掉等), 清它的
        # .inflight, 否则这个 job 被永久保护(泄漏)。admit 拉走的已在
        # update_state_after_alloc 清过 + 移出集合 → 这里命不中, 不重复清。
        # 不会误清下一轮: 多轮顺序执行, 下一轮的 mark 发生在本轮 finished 之后。
        if (not self.is_producer and self._round_kv_enabled
                and self._round_store_obj is not None):
            if (request.request_id in self._arena_sinked
                    or request.request_id in self._preempt_saved):
                _jid = (getattr(request, "job_id", None)
                        or self._preempt_saved.get(request.request_id))
                if _jid:
                    self._round_store_obj.clear_inflight(str(_jid))
        self._preempt_saved.pop(request.request_id, None)
        # ★ P2 兜底: claim 了 SSD 段却没走到 load 就结束的请求 (排队超时挂掉
        # 等), 清 SSD 账本 .inflight 防永久保护泄漏. 正常路径 worker promote
        # 后已清, 这里幂等重清无害; 不误清下一轮 (多轮顺序执行, 同 CPU 侧).
        _sjob = self._ssd_marked.pop(request.request_id, None)
        if (_sjob and self._round_kv_enabled
                and self._round_store_obj is not None):
            self._round_store_obj.ssd_clear_inflight(_sjob)
        self._rk_ph2_tail.pop(request.request_id, None)
        self._rk_ph2_res.pop(request.request_id, None)
        self._ph2_promote_pending.pop(request.request_id, None)
        # Phase2 arena-sink markers 生命周期到此 (防 _arena_sink_failed 无界增长)
        self._arena_sink_failed.discard(request.request_id)
        self._arena_sink_ts.pop(request.request_id, None)
        self._arena_sinked.discard(request.request_id)
        self._pending_failed_block_migrations.pop(request.request_id, None)
        if self.is_producer and self.direct_block_mode:
            # Bug 4 fix: only treat this request as delay-free (waiting
            # for decode RELEASE) if its bridge metadata was actually
            # staged.  request_finished fires on two paths:
            #   - natural completion → prefill finished, bridge staged,
            #     decode will migrate KV and send RELEASE → delay-free ✓
            #   - external abort (finish_requests, e.g. client 300s
            #     timeout) → prefill may have been mid-chunk, bridge
            #     never staged → decode never sees this request, no
            #     RELEASE will ever come → must NOT be marked delay-free
            # Before this fix the second case leaked into delay-free
            # and locked KV blocks for request_completion_timeout_s
            # (600s), starving the admission control and triggering
            # engine-wide stalls.
            bridge_staged = (
                self.p2p_nccl_engine is not None
                and self.p2p_nccl_engine.was_bridge_staged(
                    request.request_id))
            if not bridge_staged:
                logger.debug(
                    "[REQUEST_FINISHED] req=%s aborted before bridge "
                    "staged; freeing blocks immediately",
                    request.request_id)
            return (bridge_staged and len(block_ids) > 0), None

        # LICHT round-kv: decode persists a finished round's full-sequence
        # KV.  Stash (job_id, block_ids, token_ids) and return delay-free
        # so the scheduler holds the blocks; the worker saves them next
        # forward and reports completion via get_finished → then freed.
        if (not self.is_producer and self._round_kv_enabled):
            job_id = getattr(request, "job_id", None)
            all_ids = list(getattr(request, "all_token_ids", []) or [])
            if job_id and len(block_ids) > 0 and all_ids:
                self._pending_round_store[request.request_id] = (
                    str(job_id), list(block_ids), all_ids)
                return True, None

        send_type = str(
            self.config.get_from_extra_config("send_type",
                                              "PUT_ASYNC")).upper()
        if self.is_producer and send_type in ("PUT_ASYNC", "GET"):
            return True, None
        return False, None

    # ==============================
    # Static methods
    # ==============================

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
        # Regular expression to match the string hostname and integer port
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            ip = match.group(1)
            port = int(match.group(2))

            return ip, port
        raise ValueError(
            f"Request id {request_id} does not contain hostname and port")

    @staticmethod
    def check_tensors_except_dim(tensor1, tensor2, dim):
        shape1 = tensor1.size()
        shape2 = tensor2.size()

        if len(shape1) != len(shape2) or not all(
                s1 == s2
                for i, (s1, s2) in enumerate(zip(shape1, shape2)) if i != dim):
            raise NotImplementedError(
                "Currently, only symmetric TP is supported. Asymmetric TP, PP,"
                "and others will be supported in future PRs.")
