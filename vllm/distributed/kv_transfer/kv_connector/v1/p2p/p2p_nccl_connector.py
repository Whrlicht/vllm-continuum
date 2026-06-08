# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
        self._arena_sink_deadline_s = float(
            _os.environ.get("LICHT_ARENA_SINK_DEADLINE_S", "30"))
        # ★ 偏差2 修复: 到 deadline 时若 arena 数据【还在】, 按用户模型继续等 decode
        # admit (arena 保管, 不重算); 只有数据【被淘没了】或 decode 堵到这个硬上限才
        # 不得已放弃重算. 防永久挂. (LICHT_ARENA_SINK_ADMIT_CAP_S, 默认 180s)
        self._arena_sink_admit_cap_s = float(
            _os.environ.get("LICHT_ARENA_SINK_ADMIT_CAP_S", "180"))
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

            # Phase 2 (PD path selector) ARENA_SINK fires: tell each
            # prefill side to D2H its KV for this request and release
            # its prefill GPU blocks.  Fire-and-(near-)forget — the
            # RPC just enqueues the D2H on the prefill side.  Decode
            # admits this request from arena in a later pass via
            # get_num_new_matched_tokens (which waits for arena
            # lookup to succeed).
            if (self._phase2_admission_gate and metadata.arena_sink
                    and self.p2p_nccl_engine is not None):
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

            # Consumer recovery (Phase 1 save-on-preempt AND Phase 2
            # admission-gate): arena → GPU paged buffer for re-admitted
            # requests whose KV lives in arena.  Both phases register the
            # request in _round_load_reqs (drained into metadata.round_load),
            # so this load must fire if EITHER phase is on — not just phase 1.
            # (Bug fix: previously gated on _phase1_save_on_preempt only, so
            # Phase-2-only deployments never read the sunk KV back.)
            # Done BEFORE the NCCL pull loop so attention sees the prefix this
            # step.  Sync load (small per-request increment).
            if ((self._phase1_save_on_preempt or self._phase2_admission_gate)
                    and self._round_kv_enabled and metadata.round_load
                    and self._round_store_obj is not None):
                _items = [_rl_item(rl) for rl in metadata.round_load]
                try:
                    _res = self._round_store_obj.load_batch(_items)
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
                if ls is not None and le is not None and _nreq > 0:
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
            self._round_store_obj.enqueue_store(
                str(job_id), list(block_ids),
                list(prompt_token_ids), request_id)
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
    def poll_fast_releases() -> list[tuple[str, dict[str, float]]]:
        """Drain the fast-release queue (scheduler-side, Change 3).

        Returns a list of (request_id, timestamps) for requests whose
        RELEASE has been received by the listener thread.
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
            get_fast_release_queue)
        q = get_fast_release_queue()
        if q is None:
            return []
        released: list[tuple[str, dict[str, float]]] = []
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
        if rid in self._rk_lk_cache:
            return self._rk_lk_cache[rid]
        job_id = getattr(request, "job_id", None)
        if not job_id or self._round_store_obj is None:
            self._rk_lk_cache[rid] = None
            return None
        _t = time.time()
        res = self._round_store_obj.lookup_resolve(
            str(job_id), request.prompt_token_ids)
        # [PROBE] 只在真正算 (cache miss) 时累计 lookup 耗时/次数.
        self._sched_lk_ms = getattr(self, "_sched_lk_ms", 0.0) \
            + (time.time() - _t) * 1000.0
        self._sched_lk_n = getattr(self, "_sched_lk_n", 0) + 1
        self._rk_lk_cache[rid] = res
        return res

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
                        matched_tokens, _mb, _sg = res
                        ext = matched_tokens - num_computed_tokens
                        if ext > 0:
                            # async (True): park in WAITING_FOR_REMOTE_KVS,
                            # bg load, engine free.  sync (False): old path.
                            return ext, self._round_async
            return 0, False

        # Phase 2 (PD path selector): if this request was previously
        # routed through the arena (RPC fired in an earlier pass), the
        # connector waits here for the prefill side to finish D2H
        # writing.  Once arena has the data, lookup succeeds and the
        # admit proceeds via the arena-load path (matches Phase 1
        # preempt-recovery: scheduler → update_state_after_alloc →
        # round_load → worker load_batch).  Miss = still defer.
        if (self._phase2_admission_gate
                and request.request_id in self._arena_sinked
                and self._round_kv_enabled
                and self._round_store_obj is not None):
            job_id = getattr(request, "job_id", None)
            if job_id:
                res = self._round_store_obj.lookup(
                    str(job_id), request.prompt_token_ids)
                if res is not None:
                    matched_tokens, _ = res
                    ext = matched_tokens - num_computed_tokens
                    if ext > 0:
                        return ext, False
            # ★ 偏差2 修复 (按用户模型: arena 保管到 admit, 不重算).
            # 到 deadline 时不再盲目放弃重算, 而是先查 arena 数据【还在不在】:
            #   - 还在 (matched≈expected) → arena 保管着, 继续等 decode admit, 不重算
            #     (只有等到硬上限 admit_cap, decode 真堵死, 才不得已放弃);
            #   - 没了 (被淘) → 数据真没了, 重算不可避免; 打 EVICTED 日志 → 这正好
            #     验证"我的后台 evictor 有没有误删 sink 数据".
            _t0 = self._arena_sink_ts.get(request.request_id)
            if _t0 is not None and (time.time() - _t0) > \
                    self._arena_sink_deadline_s:
                _eblk = max(1, len(request.prompt_token_ids)
                            // self._block_size)
                _mblk = 0
                try:
                    _dbg = self._round_store_obj.lookup_resolve(
                        str(job_id), request.prompt_token_ids)
                    _mblk = int(_dbg[1]) if _dbg else 0
                except Exception:  # pragma: no cover
                    pass
                _present = _mblk >= _eblk * 0.5
                _waited = time.time() - _t0
                if _present and _waited < self._arena_sink_admit_cap_s:
                    # 数据还在 arena → 按用户模型继续等 admit, 不重算
                    return None, False
                # 数据被淘 / decode 堵到硬上限 → 不得已放弃
                self._arena_sinked.discard(request.request_id)
                self._arena_sink_ts.pop(request.request_id, None)
                self._arena_sink_failed.add(request.request_id)
                logger.warning(
                    "Phase2 arena-sink req=%s 放弃: arena剩 %d/%d 块 等了%.0fs "
                    "→ %s", request.request_id, _mblk, _eblk, _waited,
                    "EVICTED-数据被淘(淘汰的锅,验证我的evictor)" if not _present
                    else "decode堵到硬上限admit不进")
                return 0, False
            return None, False

        # Phase 2 (PD path selector): first time we see this request
        # in admission.  Project post-admit occupancy.  > threshold
        # -> route through arena (mark, fire RPC via meta, defer);
        # ≤ threshold -> NCCL direct (current behaviour).
        if (self._phase2_admission_gate
                and request.request_id not in self._preempt_saved
                and request.request_id not in self._arena_sink_failed
                and self._admission_kv_total_blocks > 0):
            try:
                _need_tokens = max(
                    0, len(request.prompt_token_ids) - num_computed_tokens)
                _need_blocks = (_need_tokens + self._block_size - 1) \
                    // self._block_size
                _predicted = (self._admission_kv_usage
                              + _need_blocks
                              / float(self._admission_kv_total_blocks))
            except Exception:
                _predicted = self._admission_kv_usage
            if _predicted > self._phase2_gate_threshold:
                # Route to arena.  Need a remote_prefill_address for
                # the RPC; fall back to parse_request_id (the existing
                # convention).
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
                self._pending_arena_sink[request.request_id] = (
                    str(_job), list(request.prompt_token_ids),
                    _remote or "")
                self._arena_sinked.add(request.request_id)
                self._arena_sink_ts[request.request_id] = time.time()
                logger.info(
                    "Phase2 sink->arena req=%s usage=%.3f pred=%.3f "
                    "thr=%.2f need_blk=%d remote=%s",
                    request.request_id, self._admission_kv_usage,
                    _predicted, self._phase2_gate_threshold,
                    _need_blocks, _remote or "<none>")
                return None, False
            # else: fall through to current NCCL path below

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
                    matched_tokens, _ = res
                    ext = matched_tokens - num_computed_tokens
                    if ext > 0:
                        # Sync arena load on resume (small increment;
                        # avoids parking in WAITING_FOR_REMOTE_KVS).
                        return ext, False
            # Arena lookup failed → fall through to the normal NCCL path
            # (which will return 0 since producer no longer has the KV,
            # leading to recompute — same as today's behaviour).
            self._preempt_saved.pop(request.request_id, None)

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
                            self._arena_sinked.discard(
                                request.request_id)
                            self._arena_sink_ts.pop(
                                request.request_id, None)
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
                    _matched_tokens, matched_blocks, slot_gen = res
                    num_blocks = num_external_tokens // self._block_size
                    # local cache hit (in blocks) sits before the gap.
                    local_hit_blocks = matched_blocks - num_blocks
                    block_ids0 = blocks.get_block_ids()[0]
                    if (num_blocks > 0 and local_hit_blocks >= 0
                            and len(block_ids0)
                            >= local_hit_blocks + num_blocks):
                        dst = list(block_ids0)[
                            local_hit_blocks:local_hit_blocks + num_blocks]
                        # ★ 跨 job: 把 lookup_resolve 解析的 slot 切到与 dst 对齐的
                        # 段 [local_hit_blocks, local_hit_blocks+num_blocks).
                        sg = None
                        if slot_gen is not None:
                            sg = slot_gen[
                                local_hit_blocks:local_hit_blocks + num_blocks]
                            if len(sg) != num_blocks:
                                sg = None   # 解析不足, 退回 own-.slot
                        self._round_load_reqs[request.request_id] = (
                            str(job_id), dst, local_hit_blocks, sg)

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
        self._rk_lk_seen.clear()

        # LICHT round-kv: drain scheduler-side reuse bookkeeping into the
        # metadata so the worker connector can load (prefill) / store
        # (decode) during the forward.  Done before the role-specific
        # early returns below so both paths carry it.
        if self._round_kv_enabled:
            for req_id, _v in self._round_load_reqs.items():
                # _v 为 (job_id, dst, src_offset) 或 (..., slot_gen) 4 元组.
                job_id, dst_block_ids, src_offset = _v[0], _v[1], _v[2]
                slot_gen = _v[3] if len(_v) > 3 else None
                src_slots = ([int(s) for (s, _g) in slot_gen]
                             if slot_gen else None)
                src_gens = ([int(g) for (_s, g) in slot_gen]
                            if slot_gen else None)
                meta.round_load.append(RoundReqMeta(
                    request_id=req_id, job_id=job_id,
                    block_ids=list(dst_block_ids), token_ids=[],
                    num_blocks=len(dst_block_ids),
                    src_block_offset=src_offset,
                    src_slots=src_slots, src_gens=src_gens))
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
