# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import json
import bisect
import math
import os
import queue as queue_mod
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Union, Tuple

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.licht_v3.decode_manager import LichtV3DecodeManager
from vllm.utils import get_hash_fn_by_name
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.metrics.monitoring import monitoring_recorder
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.core.estimate_with_func import ToolCallEstimator, Continuum_Recorder

logger = init_logger(__name__)

# ★ LICHT_PROBE=1: master switch for stall-investigation probes (HITPRED-SLOW
# here). Default off → zero overhead. See vllm/v1/engine/core.py. NOTE: the
# _bg_free_loop Fix B (blocking queue) is NOT gated — it is a real fix, always on.
_LICHT_PROBE = os.environ.get("LICHT_PROBE") == "1"


def _sanitize_output_tag(value: str) -> str:
    sanitized = "".join(
        ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_"
        for ch in value
    )
    sanitized = sanitized.strip("_")
    return sanitized or "instance"


def _resolve_instance_output_dir(vllm_config: VllmConfig) -> str:
    configured_output_dir = os.environ.get("RUN_OUTPUT_DIR")
    if configured_output_dir:
        return configured_output_dir

    base_output_dir = "./continuum_exp"
    explicit_tag = os.environ.get("CONTINUUM_INSTANCE_TAG")
    if explicit_tag:
        return os.path.join(base_output_dir, _sanitize_output_tag(explicit_tag))

    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return base_output_dir

    role_map = {
        "kv_producer": "prefill",
        "kv_consumer": "decode",
        "kv_both": "both",
    }
    role = role_map.get(kv_transfer_config.kv_role,
                        kv_transfer_config.kv_role or "single")
    http_port = kv_transfer_config.get_from_extra_config("http_port", None)
    if http_port is not None:
        tag = f"{role}_{http_port}"
    else:
        engine_id = kv_transfer_config.engine_id or "engine"
        tag = f"{role}_{engine_id}"

    return os.path.join(base_output_dir, _sanitize_output_tag(tag))


def _resolve_instance_role(vllm_config: VllmConfig) -> str:
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return "single"

    role_map = {
        "kv_producer": "prefill",
        "kv_consumer": "decode",
        "kv_both": "both",
    }
    return role_map.get(kv_transfer_config.kv_role,
                        kv_transfer_config.kv_role or "single")


class Scheduler(SchedulerInterface):

    # LICHT prefill dynamic-priority parameters.
    # Update these constants directly if you need to retune the strategy.
    LICHT_PREFILL_SCORE_A = 3.0
    LICHT_PREFILL_SCORE_B = 1.0
    LICHT_PREFILL_SCORE_TMAX_S = 120.0
    # Power-law shape for the round_decay term used in
    # _compute_licht_prefill_score.  alpha=0.5 → decay = 1/sqrt(1+k),
    # which keeps mid-round (k=4–15) requests reachable by hunger
    # compensation.  The previous exp(-k) form created a "death valley"
    # where k>=5 requests effectively had zero recovery slope.
    LICHT_PREFILL_ROUND_DECAY_ALPHA = 0.5
    # Min-run grace for LICHT preempt selector: requests admitted into
    # the running pool within the last GRACE_S seconds are excluded from
    # the eviction candidate set, so a freshly-admitted request gets at
    # least one productive prefill chunk before it can be evicted again.
    LICHT_PREEMPT_MIN_RUN_GRACE_S = 15.0

    # LICHTV2 backfill-window parameters.
    # N = lookahead horizon in scheduler steps.  At each scheduler step
    # we maintain two arrays of length N+1:
    #   future_free[t]  = predicted physical free blocks at end of step t
    #   future_alloc[t] = total prefill block alloc at step t (per-step flow)
    # A waiting candidate j is admitted iff inserting it leaves
    #   future_free[t] + cum_delta(t) >= threshold for all t in [0, N]
    # where threshold = LICHTV2_LONG_TAIL_HEADROOM_RATIO * total_blocks
    # for long-tail (R(j) > N) candidates and 0 otherwise.  Additionally
    # future_alloc[t] + B_j(t) must not exceed max_num_batched_tokens /
    # block_size to respect the per-step token budget.
    LICHTV2_N = 300
    LICHTV2_LONG_TAIL_HEADROOM_RATIO = 0.025
    LICHTV2_MAX_LONG_BRIDGE = 2

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        self.continuum_recorder = Continuum_Recorder()
        # ★ 临时 hitprobe (LICHT_HITPROBE=1, 默认关): 诊断 vLLM prefix-hit 指标为何
        #   低于真实复用(hit_length). gcb=get_computed_blocks 调用次数(=queries事件),
        #   admit=真 admit 数; gcb>>admit => 重复计数. ext_none/zero/pos=arena 命中
        #   分布; ext_zero 多 => arena 瞬时 miss 按只本地命中计低. uncount=撤销数.
        self._hitprobe_on = os.environ.get("LICHT_HITPROBE") == "1"
        self._hp = {"gcb": 0, "q_tok": 0, "h_local": 0, "h_ext": 0,
                    "ext_none": 0, "ext_zero": 0, "ext_pos": 0,
                    "uncount": 0, "admit": 0}
        self._hp_step = 0
        logger.info("HITPROBE enabled=%s (LICHT_HITPROBE=%s)",
                    self._hitprobe_on, os.environ.get("LICHT_HITPROBE"))
        output_dir = _resolve_instance_output_dir(vllm_config)
        self.continuum_recorder.set_output_dir(output_dir)
        monitoring_recorder.set_output_dir(output_dir)
        os.environ["RUN_OUTPUT_DIR"] = output_dir
        logger.info("Continuum timestamps output dir: %s", output_dir)

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        self.dcp_world_size = \
            vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): The scheduler’s block_size must be multiplied
        # by dcp_world_size, since block hashes are computed on the
        # original full token sequence at a granularity of
        # original_block_size × dcp_world_size.
        if self.dcp_world_size > 1:
            self.block_size *= self.dcp_world_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        elif self.scheduler_config.policy == "continuum":
            self.policy = SchedulingPolicy.CONTINUUM
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # LICHT / LICHTV2 / LICHTV3 are mutually-exclusive scheduling switches.
        # LICHTV2 reuses LICHT's score-based picker for waiting requests,
        # so on prefill it implies licht_prefill_sched_enabled.  Decode
        # is unaffected by LICHTV2 (it always falls back to default FCFS).
        # LICHTV3 is a strict superset of LICHTV2: it reuses every v2
        # code path on prefill and only adds opt-in decode-side hooks
        # for round-gap KV management.  Internally we set
        # `licht_v2_enabled = True` when v3 is on, so the existing v2
        # prefill backfill machinery activates without any branching.
        self.instance_role = _resolve_instance_role(vllm_config)
        _v3_flag = bool(getattr(self.scheduler_config, "licht_v3", False))
        _v2_flag = bool(self.scheduler_config.licht_v2)
        _v1_flag = bool(self.scheduler_config.licht)
        if sum([_v1_flag, _v2_flag, _v3_flag]) > 1:
            raise ValueError(
                "--licht, --licht-v2 and --licht-v3 are mutually "
                "exclusive; specify at most one.")
        # licht_v3 implies licht_v2 prefill behaviour.
        self.licht_v3_enabled = _v3_flag
        self.licht_v2_enabled = _v2_flag or _v3_flag
        self.licht_v2_prefill_sched_enabled = (
            self.licht_v2_enabled and self.instance_role != "decode")
        # v3 decode hook flag is set independently of the v2 prefill path
        # and only fires on decode instances.  It is wired here so that
        # later v3 decode-side components can gate themselves on it
        # without touching the licht-v2 derivation above.
        self.licht_v3_decode_enabled = (self.licht_v3_enabled
                                        and self.instance_role == "decode")
        # LICHTV3 decode-side coordinator.  Lazily attached after the
        # KV cache manager finishes init (so we know block_size and
        # the real max_num_seqs / num_gpu_blocks).  See the bottom of
        # __init__.
        self.licht_v3_decode_manager: Optional[LichtV3DecodeManager] = None
        # licht_enabled stays true for LICHTV2 on prefill so the existing
        # score / preempt-victim machinery is reused.  Pure decode
        # instances under LICHTV2 don't get the LICHT decode-FCFS path.
        self.licht_enabled = (self.scheduler_config.licht
                              or self.licht_v2_prefill_sched_enabled)
        self.licht_prefill_sched_enabled = (self.licht_enabled
                                            and self.instance_role != "decode")
        self.licht_decode_fcfs_enabled = (self.scheduler_config.licht
                                          and self.instance_role == "decode")
        self.licht_waiting_round_start_ts: dict[str, float] = {}
        # Wall-clock timestamp of the most recent admission into running.
        # Used by _pick_preempt_victim_licht to enforce a min-run grace
        # so that just-admitted requests are not immediately evicted.
        self.licht_running_admit_ts: dict[str, float] = {}
        # LICHTV2: snapshot num_computed_tokens at admission so the
        # timeline can compute B_i(t) / release_blocks(i) consistently
        # across scheduler steps even as num_computed_tokens advances.
        self.licht_v2_num_computed_at_admit: dict[str, int] = {}
        # LICHTV2: count of prefix-cache blocks that were in the free
        # queue (refcount=0) at admit time and got "touched" by this
        # request, removing them from the free queue.  This count is
        # used to model the t=0 free-queue consumption (in addition to
        # B_at(t=0)) AND the t=R release back to the free queue.
        # Without this, the timeline under-counts admit-time consumption
        # for multi-turn workloads that hit prior-turn prefix cache.
        self.licht_v2_evictable_prefix_at_admit: dict[str, int] = {}
        # ★ 计划钉死 (2026-07-05): 准入那一刻的 dyn-chunk cap 快照. 已准入的
        # 请求运行期用它, 不再随每步全局 S* 漂移. timeline 在准入时按此 cap
        # 预订 R/B 并过 guard; 若运行期 cap 被 S*→0 退化改写成整段/DEGEN_CAP,
        # 在跑请求单步索取远超预订 → 打穿 KV 池 → 驱逐 (2026-07-04 复盘). 钉死
        # 后 S* 变化只影响【新准入】者 → plan==reality → 零驱逐不变式恢复.
        self.licht_v2_dyn_cap_at_admit: dict[str, int] = {}
        self._dyn_pin_cap = (
            os.environ.get("LICHT_DYN_PIN_CAP", "1") == "1")
        # S*=0 退化兜底: 记每个【已被切过】请求上一步的 chunk, 供 S*=0 时复用
        # (存在=running/已排过; 缺失=本step新请求). 见 _dyn_degenerate.
        self._dyn_last_chunk: dict[str, int] = {}
        # 详细 trace (LICHT_TRACE=path): 每请求 arrive/admit/finish + 每步
        # KV利用率/chunk, 供 queue-vs-compute + 长/短分开分析. best-effort。
        self._trace_path = os.environ.get("LICHT_TRACE")
        self._trace_f = open(self._trace_path, "w") if self._trace_path else None
        self._trace_step = 0
        # P5b: last-probed (prefix-hit, evictable) per WAITING request, so
        # the StepEvent can report the real prefix hit for waiting reqs
        # (their request.num_computed_tokens is still 0 pre-admit).  The
        # decode-side simulator uses this to model returning requests'
        # 回传'd prefix.  request_id → (hit_length, evictable_prefix).
        self._v3_waiting_hit: dict[str, tuple[int, int]] = {}
        # LICHT_SCHED_HIT_PRED: per-step predicted cross-tier prefix hit
        # (HBM + arena) for WAITING requests, computed BEFORE scoring so the
        # score/dyn_precompute see the real remaining C (= num_tokens - hit)
        # instead of treating waiting reqs as num_computed=0 (which mis-scores
        # a returning round with a big cached prefix as LONG).  Rebuilt each
        # step.  Empty (env off) -> consumers fall back to num_computed_tokens.
        self._sched_hit_pred: dict[str, int] = {}
        # 回传 REMOVED (2026-05-21): no defer/push-back state on the prefill.
        # Total number of GPU blocks (denominator for headroom ratio).
        self._total_kv_blocks = num_gpu_blocks
        if self.licht_enabled:
            logger.info(
                "LICHT mode enabled (instance_role=%s, v2=%s, v3=%s). "
                "KV transfer strategy currently uses default implementation.",
                self.instance_role,
                self.licht_v2_enabled,
                self.licht_v3_enabled,
            )
            if self.licht_prefill_sched_enabled:
                logger.info(
                    "LICHT prefill scheduler params: a=%.1f, b=%.1f, "
                    "Tmax=%.1fs, round_decay=1/(1+k)^%.2f, "
                    "preempt_grace=%.1fs",
                    self.LICHT_PREFILL_SCORE_A,
                    self.LICHT_PREFILL_SCORE_B,
                    self.LICHT_PREFILL_SCORE_TMAX_S,
                    self.LICHT_PREFILL_ROUND_DECAY_ALPHA,
                    self.LICHT_PREEMPT_MIN_RUN_GRACE_S,
                )
            if self.licht_v2_prefill_sched_enabled:
                _chunk = max(
                    1, self.scheduler_config.long_prefill_token_threshold)
                _max_alloc = (self.scheduler_config.max_num_batched_tokens
                              // self.block_size)
                logger.info(
                    "LICHTV2 prefill backfill params: N=%d, "
                    "long_tail_headroom=%.1f%% of %d blocks, "
                    "max_long_bridge=%d, chunk_size=%d tokens, "
                    "block_size=%d, max_alloc_per_step=%d blocks, "
                    "release_at=t=R (1-step delay after last alloc)",
                    self.LICHTV2_N,
                    self.LICHTV2_LONG_TAIL_HEADROOM_RATIO * 100,
                    self._total_kv_blocks,
                    self.LICHTV2_MAX_LONG_BRIDGE,
                    _chunk,
                    self.block_size,
                    _max_alloc,
                )

        # Initialize ToolCallEstimator with tokenizer config
        self.tool_call_estimator = ToolCallEstimator(
            model_name=vllm_config.model_config.tokenizer,
            tokenizer_mode=vllm_config.model_config.tokenizer_mode,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
            tokenizer_revision=vllm_config.model_config.tokenizer_revision,
        )

        # TODO(Hanchen) This stored the list of pineed requests and the time they need to be removed
        self.pinned_requests: list[Tuple[Request, float]] = []
        # Track the first entry time for each job_id in running queue (for job_id level FCFS)
        self.running_job_id_first_entry_time: dict[str] = {}
        # Track prefill start time for throughput measurement
        self.request_prefill_start_time: dict[str, float] = {}
        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        # Requests already freed via the fast-release side-channel (Change 3).
        self._fast_released_req_ids: set[str] = set()

        # --- Delay-free block tracking (admission control) ---
        # Number of blocks currently held by delay-free requests (waiting
        # for RELEASE from decode).  Used by schedule() to avoid evicting
        # running requests when delay-free blocks will be freed soon.
        self._num_delay_free_blocks: int = 0
        self._delay_free_req_ids: set[str] = set()

        # --- Background block-free thread (Change 5) ---
        # Lock protects kv_cache_manager.free / allocate_slots from
        # concurrent access by the background thread and the main loop.
        self._kv_free_lock = threading.Lock()
        # P9/兜底: held across the ENTIRE schedule() so the install-center's
        # background drain can run safely DURING the forward (execute_model,
        # when block_pool is untouched) yet never overlap schedule()'s
        # allocate_slots/touch (the free_block_queue corruption that crashed
        # the engine).  RLock = re-entrant on the engine thread.
        self._schedule_lock = threading.RLock()
        # Deferred cleanup items produced by the background thread.
        # Each item: (request_id, timestamps_dict)
        # The main thread drains this at schedule()/update_from_output() to
        # do non-block cleanup (del requests, pin logic, monitoring).
        self._deferred_frees: queue_mod.SimpleQueue = queue_mod.SimpleQueue()
        self._bg_free_thread: Optional[threading.Thread] = None


        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

        # Start background free thread if KV connector is active (producer).
        if (self.connector is not None
                and getattr(self.connector, "is_producer", False)):
            self._bg_free_thread = threading.Thread(
                target=self._bg_free_loop, daemon=True)
            self._bg_free_thread.start()

        # 回传 REMOVED (2026-05-21): no prefill install center (it only
        # existed to receive decode→prefill push-backs).

        # LICHTV3 decode-side coordinator: instantiate only on decode
        # instances under --licht-v3.  Construction is cheap; the tool
        # predictor model loads lazily on first use.
        if self.licht_v3_decode_enabled:
            try:
                run_dir = os.environ.get(
                    "LICHT_V3_TOOL_PREDICTOR_DIR") or None
                default_t = float(os.environ.get(
                    "LICHT_V3_DEFAULT_T_TOOL_S", "5.0"))
                mcfg = vllm_config.model_config
                self.licht_v3_decode_manager = LichtV3DecodeManager(
                    max_slots=self.scheduler_config.max_num_seqs,
                    block_size=self.block_size,
                    tool_predictor_run_dir=run_dir,
                    default_t_tool_s=default_t,
                    kv_cache_manager=self.kv_cache_manager,
                    model_name_or_path=(getattr(mcfg, "tokenizer", None)
                                         or getattr(mcfg, "model", None)),
                    tokenizer_mode=getattr(mcfg, "tokenizer_mode", "auto"),
                    trust_remote_code=getattr(mcfg, "trust_remote_code", False),
                    tokenizer_revision=getattr(mcfg, "tokenizer_revision", None),
                )
                self._v3_retained_requests: dict[str, Request] = {}
                self.licht_v3_decode_manager.bind_release_retained_cb(
                    self._v3_release_retained)
                # 回传 (P1): give the decode manager the SCHEDULER-role
                # connector so it can enqueue KV push-backs into connector
                # metadata (executed on the worker during forward).
                if self.connector is not None:
                    self.licht_v3_decode_manager.bind_connector(self.connector)
                logger.info(
                    "LICHTV3 decode manager wired up on instance_role=%s",
                    self.instance_role)
            except Exception as e:
                logger.warning(
                    "LICHTV3 decode manager init failed: %s; v3 will be "
                    "inert on this instance.", e)
                self.licht_v3_decode_manager = None

        # LICHTV3 K_queue ground-truth logger (prefill side).  Counts the
        # number of scheduler steps between a request being added to the
        # waiting queue (add_request) and being moved to scheduled_new_reqs
        # (admitted to RUNNING).  Writes one JSONL line per admission to
        # `LICHT_V3_KQUEUE_ACTUAL_LOG`.  Disabled if env var
        # LICHT_V3_LOG_KQUEUE_ACTUAL=0 or instance is not prefill.
        self._v3_kqueue_log_enabled = (
            self.instance_role != "decode"
            and os.environ.get("LICHT_V3_LOG_KQUEUE_ACTUAL", "1") == "1")
        self._v3_sched_step: int = 0
        self._v3_arrival_step: dict[str, int] = {}
        self._v3_arrival_ts: dict[str, float] = {}
        self._v3_kqueue_log_path = os.environ.get(
            "LICHT_V3_KQUEUE_ACTUAL_LOG",
            "/data/whr/vllm-continuum/output/v3_kqueue_actual.jsonl")
        if self._v3_kqueue_log_enabled:
            try:
                os.makedirs(os.path.dirname(self._v3_kqueue_log_path),
                            exist_ok=True)
                # Truncate at startup so each run is self-contained.
                open(self._v3_kqueue_log_path, "w").close()
                logger.info(
                    "LICHTV3 K_queue actual logger enabled on prefill "
                    "(role=%s) → %s",
                    self.instance_role, self._v3_kqueue_log_path)
            except Exception as e:
                logger.warning(
                    "LICHTV3 K_queue actual logger disable: %s", e)
                self._v3_kqueue_log_enabled = False

        # LICHTV3 StepEvent publisher (prefill side).  Publishes one
        # JSON message per schedule() call so decode-side ShadowScheduler
        # can mirror prefill state in real time.  Gated by env
        # LICHT_V3_STEP_EVENT_PUB_ADDR (e.g. "tcp://*:5559") and only
        # active on prefill instances.
        self._v3_step_event_pub_addr = os.environ.get(
            "LICHT_V3_STEP_EVENT_PUB_ADDR", "")
        self._v3_step_event_pub = None
        self._v3_step_wall_history: "list[float]" = []  # recent step_end ts
        if (self._v3_step_event_pub_addr
                and self.instance_role != "decode"):
            try:
                import zmq
                self._v3_zmq_ctx = zmq.Context()
                self._v3_step_event_pub = self._v3_zmq_ctx.socket(zmq.PUB)
                self._v3_step_event_pub.bind(self._v3_step_event_pub_addr)
                logger.info("LICHTV3 StepEvent pub bound on %s",
                            self._v3_step_event_pub_addr)
            except Exception as e:
                logger.warning("LICHTV3 StepEvent pub bind failed: %s; "
                               "ShadowScheduler will not receive updates",
                               e)
                self._v3_step_event_pub = None

    def pop_running_request_based_on_last_step(self, request: Request) -> tuple[Request, bool]:
        """Pop a request from running queue based on job_id level FCFS and last step."""
        if len(self.running) <= 1:
            #wpop from pinned requests from smallest end_time
            latest_pin_end_request = None
            latest_pin_end_time = -float('inf')
            for req, end_time in self.pinned_requests:
                if end_time > latest_pin_end_time:
                    latest_pin_end_time = end_time
                    latest_pin_end_request = req
            if latest_pin_end_request is not None:
                self.pinned_requests.remove((latest_pin_end_request, latest_pin_end_time))
                return latest_pin_end_request, True

            raise IndexError("pop from empty running queue")
                
        # First, find the request that is not last step
        latest_request = None
        latest_entry_time = -float('inf')
        
        for req in self.running:
            job_entry_time = self.running_job_id_first_entry_time.get(req.job_id)
            if job_entry_time > latest_entry_time and not req.is_last_step:
                latest_entry_time = job_entry_time
                latest_request = req
        
        if latest_request is not None:
            self.running.remove(latest_request)
            return latest_request, False

        # Second, check the other requests
        for req in self.running:
            job_entry_time = self.running_job_id_first_entry_time.get(req.job_id)
            if job_entry_time > latest_entry_time:
                latest_entry_time = job_entry_time
                latest_request = req
        
        if latest_request is not None:
            self.running.remove(latest_request)
            return latest_request, False
    
    # TODO (Hanchen) needs to get current time, add with length of pin to put end time of pin
    def pin_request(self, request: Request, length_of_pin: float) -> None:
        self.continuum_recorder.request_pinned(request)
        self.pinned_requests.append((request, time.time() + length_of_pin))

    def unpin_request(self, request: Request, end_time: float) -> None:
        self.pinned_requests.remove((request, end_time))
        self.continuum_recorder.request_unpinned(request)
        with self._kv_free_lock:
            self.kv_cache_manager.free(request)

    # TODO (Hanchen) this needs to be called at the beginning of each step to clean up pinned request based on system time
    # The LRU is handled by kv cache mangager through a reference counter
    def unpin_requests_regular(self) -> None:
        # Check if job id "1" is in waiting requests
        waiting_job_ids = [req.job_id for req in self.waiting]

        for request, end_time in self.pinned_requests:
            #print("time.time() - end_time:", time.time() - end_time)
            if request.job_id not in waiting_job_ids and time.time() >= end_time:
                #print(f"Unpinning request {request.request_id} with job id {request.job_id}")
                self.unpin_request(request, end_time)

    def is_pinned(self, request: Request) -> bool:
        for req, _ in self.pinned_requests:
            if req.job_id == request.job_id:
                return True
        return False

    def _reset_licht_waiting_state(
        self,
        request: Request,
        now_monotonic: Optional[float] = None,  # kept for back-compat; ignored
    ) -> None:
        # Plan B: wait_start is always the request's arrival_time; it is
        # never reset on preempt.  T_wait therefore accumulates
        # monotonically from arrival, spanning both waiting and running
        # periods.  This prevents a preempted request's hunger compensation
        # from collapsing to zero when it is driven back to waiting.
        if not self.licht_enabled:
            return
        self.licht_waiting_round_start_ts[request.request_id] = (
            request.arrival_time)

    def _drop_licht_waiting_state(self, request_id: str) -> None:
        if not self.licht_enabled:
            return
        self.licht_waiting_round_start_ts.pop(request_id, None)

    def _ensure_licht_waiting_start_timestamps(self) -> None:
        # Plan B: any waiting request without a recorded wait_start should
        # fall back to its arrival_time (not now), so that requests that
        # have been sitting in the waiting queue for a while retain their
        # accumulated T_wait even if the bookkeeping was dropped somewhere.
        if not self.licht_prefill_sched_enabled:
            return
        for req in self.waiting:
            self.licht_waiting_round_start_ts.setdefault(
                req.request_id, req.arrival_time)

    def _compute_licht_prefill_score(
        self,
        request: Request,
        now: float,
    ) -> float:
        # k_i is the request's real agent/dialog round from API metadata.
        # NOTE: `now` must be a wall-clock timestamp (time.time()),
        # because wait_start is stored as request.arrival_time which is
        # also wall-clock.  Mixing in time.monotonic() here would yield
        # garbage T_wait values.
        # Experiment (env-gated by LICHT_SCHED_SCHEME): four admission/priority
        # strategies for the chunk-scheduling study.  Default (unset) keeps the
        # original round-based score below — no effect on normal runs.
        _scheme = os.environ.get("LICHT_SCHED_SCHEME")
        if _scheme:
            # Use the predicted cross-tier hit for WAITING reqs (else
            # num_computed_tokens: real for running, 0 for fresh waiting).
            _hit = self._sched_hit_pred.get(
                request.request_id, request.num_computed_tokens)
            C = max(request.num_tokens - _hit, 1)
            L = max(_hit, 0)
            base = math.log(L * C + C * C + 1.0)        # log(L*C + C^2)
            if _scheme == "sjf":
                return -base                            # shortest-job first
            if _scheme == "fcfs":
                return -request.arrival_time            # first-come first-serve
            if _scheme == "hunger":                     # big-first + anti-starve
                ws = self.licht_waiting_round_start_ts.get(
                    request.request_id, request.arrival_time)
                tmax = float(os.environ.get("LICHT_HUNGER_TMAX_S", "5"))
                return base + max(now - ws - tmax, 0.0)
            if _scheme == "sjf_hunger":                 # SJF + aging floor for big
                ws = self.licht_waiting_round_start_ts.get(
                    request.request_id, request.arrival_time)
                tmax = float(os.environ.get("LICHT_HUNGER_TMAX_S", "5"))
                return -base + max(now - ws - tmax, 0.0)
            if _scheme == "fsp":                        # Fair Sojourn Protocol approx:
                # serve earliest fair-share (PS) completion first = arrival +
                # beta * work.  Near-SRPT mean + no job worse than fair-share
                # (no starvation), and does NOT degenerate to FCFS.
                W = C * request.num_tokens              # work ~ C*(L+C)
                beta = float(os.environ.get("LICHT_FSP_BETA", "1e-7"))
                return -(request.arrival_time + beta * W)
            if _scheme == "sjf_deadline":               # SJF + slowdown-deadline
                W = C * request.num_tokens
                ws = self.licht_waiting_round_start_ts.get(
                    request.request_id, request.arrival_time)
                theta = float(os.environ.get("LICHT_DEADLINE_THETA", "1e-6"))
                if (now - ws) >= theta * W:              # slowdown cap hit -> boost
                    return 1e12 + (now - ws)             # most-overdue first
                return -base                             # else SJF
            if _scheme == "wsjf":                        # HRRN response ratio
                W = C * request.num_tokens
                ws = self.licht_waiting_round_start_ts.get(
                    request.request_id, request.arrival_time)
                gamma = float(os.environ.get("LICHT_WSJF_GAMMA", "1e-7"))
                S = max(gamma * W, 1e-6)
                return (now - ws + S) / S                # (wait+service)/service
            if _scheme in ("longcap_sjf", "longcap_fcfs"):
                thr = int(os.environ.get("LICHT_LONG_C", "5120"))
                # LICHT_LONGCAP_ORDER: "long" (default) = longs-first (longs FCFS
                # admitted first up to theta, shorts fill rest); "short" = the old
                # shorts-first (shorts top band, longs below). Diagnostic toggle.
                _order = os.environ.get("LICHT_LONGCAP_ORDER", "long")
                if _order == "short":
                    if C > thr:                          # long: below shorts
                        return -request.arrival_time
                    if _scheme == "longcap_sjf":         # short: SJF, top band
                        return 1e9 - base
                    return 1e9 - request.arrival_time    # short: FCFS, top band
                if C > thr:                              # LONG: FCFS, scheduled
                    return 2e9 - request.arrival_time    #   FIRST up to theta cap
                if _scheme == "longcap_sjf":             # short: SJF, fills rest
                    return -base
                return -request.arrival_time             # short: FCFS, fills rest
            if _scheme == "sjf_reservation":             # SJF + reserve head big job
                if request.request_id == getattr(self, "_resv_head_id", None):
                    return 1e12                          # reserved head big -> top
                return -base
            return base                                 # "guard"/default: big-first
        ki = max(request.agent_round, 0)
        wait_start = self.licht_waiting_round_start_ts.get(
            request.request_id,
            request.arrival_time,
        )
        twait = max(now - wait_start, 0.0)
        # LICHT score form: A * log(1 + k_i)
        #                 + B * (1 + k_i)^(-alpha) * max(twait - tmax, 0)
        # The previous form used exp(-k_i) for round_decay, which collapsed
        # to ~0.007 by k=5 and effectively flat-lined the hunger
        # compensation for mid-round requests (the "death valley" k=4–15).
        # The power-law form decays much more slowly (e.g. alpha=0.5 gives
        # 0.45 at k=4 and 0.30 at k=10), so a stuck mid-round request can
        # still climb past higher-round neighbours within ~120s of waiting.
        wait_term = max(twait - self.LICHT_PREFILL_SCORE_TMAX_S, 0.0)
        round_decay = (1.0 + ki) ** (-self.LICHT_PREFILL_ROUND_DECAY_ALPHA)
        return (self.LICHT_PREFILL_SCORE_A * math.log1p(ki)
                + self.LICHT_PREFILL_SCORE_B * round_decay * wait_term)

    # ---- Dynamic chunk (env-gated LICHT_DYN_CHUNK in {A,B,C,D}) -------------
    # Picks each long request's per-step chunk by balancing future KV re-read
    # (∝ 1/c) against the drag a longer step puts on shorter/waiting requests.
    #   A: c_i = sqrt(brb * C_i / (N_total-2))          (per-req, lumped victims)
    #   B: c_i = sqrt(brb * C_i / N_short(i))           (per-req, SRPT victims)
    #   C: S   = sqrt(brb * Σ D_iC_i / (W_t * Σ D_i))   (batch S, lumped W_t)
    #   D: S   = sqrt(brb * Σ D_iC_i / Σ N_short(i)D_i)  (batch S, SRPT victims)
    # brb = beta_r/b (tokens).  Short reqs (rem<=C*) never chunked.
    def _licht_build_sched_hit_pred(self) -> None:
        """Predict each WAITING request's real cross-tier prefix hit
        (HBM local + arena external) BEFORE scoring / dyn_precompute /
        admission.  Lets the score & dyn_precompute use C_eff = num_tokens -
        hit instead of treating waiting reqs as num_computed=0 (which mis-
        classifies a returning round with a big cached prefix as LONG).

        SIDE-EFFECT-FREE: find_longest_cache_hit is a pure read (we skip
        get_computed_blocks's prefix_cache_stats update); the connector arena
        lookup is a pure read that caches per request_id in _rk_lk_cache (so
        the value is reused for free at admit).  Rebuilt every step so
        eviction is reflected.  Best-effort: any failure -> no entry ->
        consumers fall back to request.num_computed_tokens.

        Gated by LICHT_SCHED_HIT_PRED=1; off -> empty dict -> current behaviour.
        """
        self._sched_hit_pred = {}
        if os.environ.get("LICHT_SCHED_HIT_PRED") != "1":
            return
        if not (self.licht_prefill_sched_enabled
                and getattr(self.kv_cache_manager, "enable_caching", False)):
            return
        coord = getattr(self.kv_cache_manager, "coordinator", None)
        if coord is None:
            return
        _hp_t0 = time.perf_counter()
        _hp_flc = 0.0  # find_longest_cache_hit 累计
        _hp_ext = 0.0  # arena get_num_new_matched_tokens 累计
        _hp_n = 0
        for req in self.waiting:
            try:
                _hp_n += 1
                # HBM local hit (pure read; no stats touched).
                _fa = time.perf_counter()
                _, local_hit = coord.find_longest_cache_hit(
                    req.block_hashes, max(req.num_tokens - 1, 0))
                _fb = time.perf_counter()
                _hp_flc += (_fb - _fa)
                hit = local_hit
                # arena external hit beyond local (pure read, cached).
                if self.connector is not None:
                    ext, _async = self.connector.get_num_new_matched_tokens(
                        req, local_hit)
                    if ext:
                        hit += ext
                    _hp_ext += (time.perf_counter() - _fb)
                self._sched_hit_pred[req.request_id] = hit
            except Exception:  # pragma: no cover - best-effort, never raise
                pass
        try:
            _hp_ms = (time.perf_counter() - _hp_t0) * 1000.0
            if _LICHT_PROBE and _hp_ms > 300.0:
                logger.warning(
                    "HITPRED-SLOW total=%.0fms n_waiting=%d "
                    "find_longest_cache_hit=%.0fms arena_lookup=%.0fms — "
                    "prefill schedule() 卡在逐 waiting 请求的 hit 预测 "
                    "(LICHT_SCHED_HIT_PRED), 霸占 prefill GIL",
                    _hp_ms, _hp_n, _hp_flc * 1000.0, _hp_ext * 1000.0)
        except Exception:
            pass

    def _licht_dyn_precompute(self) -> None:
        self._dyn_mode = os.environ.get("LICHT_DYN_CHUNK")
        if self._dyn_mode not in ("A", "B", "C", "D", "E", "F"):
            return
        # brb = beta_r/b (tokens). Prefer a startup-calibrated value from a
        # json file (LICHT_DYN_BRB_FILE, written by the boot calibration
        # microbench) -> else env LICHT_DYN_BRB -> else default. Cached once.
        _cb = getattr(self, "_dyn_brb_cached", None)
        if _cb is None:
            _cb = float(os.environ.get("LICHT_DYN_BRB", "216"))  # A800 calib fallback
            _bf = os.environ.get("LICHT_DYN_BRB_FILE")
            if _bf and os.path.exists(_bf):
                try:
                    _cb = float(json.load(open(_bf))["brb"])
                    logger.info("dynamic_chunk: loaded calibrated brb=%.1f "
                                "from %s", _cb, _bf)
                except Exception as _e:
                    logger.warning("brb file load failed (%s); using %.1f",
                                   _e, _cb)
            self._dyn_brb_cached = _cb
        self._dyn_brb = _cb
        # cstar (dynamic short/long boundary) IS the same concept as the longcap
        # LICHT_LONG_C boundary -> default to it so one knob moves both; an
        # explicit LICHT_DYN_CSTAR still overrides if you want them to differ.
        self._dyn_cstar = int(os.environ.get(
            "LICHT_DYN_CSTAR", os.environ.get("LICHT_LONG_C", "5120")))
        self._dyn_floor = int(os.environ.get("LICHT_DYN_FLOOR", "256"))
        cstar = self._dyn_cstar
        # mode E: smooth long/short via lambda = smoothstep((C-Clow)/(Chigh-Clow))
        clow = float(os.environ.get("LICHT_DYN_CLOW", "2048"))
        chigh = float(os.environ.get("LICHT_DYN_CHIGH", "5120"))
        self._dyn_clow = int(clow)
        rema = []                 # remaining new-prefill tokens, all in-system
        longs = []                # (C_i, D_i) for long reqs (rem > cstar)
        e_num = e_gA = e_wsoft = 0.0   # mode E: Σλ_iD_iC_i, Σλ_iD_i, Σ(1-λ_i)
        # mode F: 长项(λ)只算 running; 短项(1-λ)分 running/waiting 两堆,
        # 由 LICHT_DYN_SHORT_SET 选用哪堆。f_lC=Σλ_run C, f_Nlong=Σλ_run。
        f_lDC = f_lC = f_lD = f_Nlong = 0.0
        f_Ts_run = f_ws_run = f_Ts_wait = f_ws_wait = 0.0
        # mode F: predict THIS step's batch (running carryover + FCFS-estimated
        # waiting admits) so the long-terms see the long requests that will
        # actually run this step -- not just last step's running.  Fixes the
        # precompute-before-admit timing gap (a soon-to-run long is still in
        # `waiting` -> f_lD=0 -> S*=0 -> run whole -> never chunked -> F≈nochunk).
        # Default-on for F; E keeps the raw running/waiting split.
        _pred = (self._dyn_estimate_step_batch(cstar)
                 if self._dyn_mode == "F" else None)
        _n_run = len(self.running)
        for _idx, r in enumerate(list(self.running) + list(self.waiting)):
            # predicted cross-tier hit for waiting (real num_computed for
            # running / when prediction is off) -> real D and C.
            _hd = self._sched_hit_pred.get(r.request_id, r.num_computed_tokens)
            rem = r.num_tokens - _hd
            if rem > 0:
                rema.append(rem)
                if rem > cstar:
                    longs.append((rem, _hd))
                # smoothstep longness lambda_i in [0,1]
                _x = (rem - clow) / max(chigh - clow, 1e-9)
                _lam = (0.0 if _x <= 0.0 else
                        (1.0 if _x >= 1.0 else (3.0 * _x * _x - 2.0 * _x ** 3)))
                _D = _hd
                e_num += _lam * _D * rem
                e_gA += _lam * _D
                e_wsoft += (1.0 - _lam)
                _s = 1.0 - _lam
                # "runs this step" = running carryover OR predicted-admitted (F);
                # E/off falls back to the raw running/waiting split.
                _in = (r.request_id in _pred) if _pred is not None \
                    else (_idx < _n_run)
                if _in:                                 # in this step's batch
                    f_lDC += _lam * _D * rem
                    f_lC += _lam * rem
                    f_lD += _lam * _D
                    f_Nlong += _lam
                    f_Ts_run += _s * _D * rem
                    f_ws_run += _s
                else:                                   # not scheduled this step
                    f_Ts_wait += _s * _D * rem
                    f_ws_wait += _s
        self._dyn_ntotal = len(rema)
        self._dyn_wt = sum(1 for x in rema if x <= cstar)   # short victims
        self._dyn_sorted = sorted(rema)
        self._dyn_S = 0
        if self._dyn_mode == "E":
            # S* = sqrt( brb * Σλ_iD_iC_i / (W_soft * Σλ_iD_i) )  (minimize J(S))
            if e_num > 0.0 and e_gA > 0.0 and e_wsoft > 0.0:
                self._dyn_S = self._dyn_clamp(
                    math.sqrt(self._dyn_brb * e_num
                              / (e_wsoft * e_gA)), None)
            # else degenerate (no longs / all fresh D=0) -> _dyn_S=0, cap falls
            # back to static so a fresh lone long still gets a first chunk.
        elif self._dyn_mode == "F":
            # 大请求被拖慢(分子) = 重读 brb·Σλ_run DC  +  分摊版"等短请求":
            #   T_short·Σλ_run C / Σλ_run   (T_short=Σ(1-λ)DC, 恒为 running 短请求,
            #   物理意义=大请求每轮要等的、batch里实际在跑的短请求算力)
            # 小请求被拖慢(分母) = W_soft · Σλ_run D
            #   W_soft=Σ(1-λ)=被更大chunk拖慢的请求数, 由 LICHT_DYN_SHORT_SET 选:
            #     run = 只 running 短请求被拖慢;
            #     all = running 短请求被拖 + waiting 请求也要多等一轮.
            # 注意: 只有 W_soft 随 all/run 变, T_short 永远是 running.
            _sset = os.environ.get("LICHT_DYN_SHORT_SET", "all")
            _Tshort = f_Ts_run
            _wsoft = f_ws_run + (f_ws_wait if _sset == "all" else 0.0)
            _num = (self._dyn_brb * f_lDC
                    + _Tshort * f_lC / max(f_Nlong, 1e-9))
            _den = _wsoft * f_lD
            if _num > 0.0 and _den > 0.0:
                self._dyn_S = self._dyn_clamp(math.sqrt(_num / _den), None)
        elif self._dyn_mode in ("C", "D") and longs:
            sum_DC = sum(D * C for C, D in longs)
            if self._dyn_mode == "C":
                denom = max(self._dyn_wt, 1) * max(sum(D for C, D in longs), 1)
            else:
                denom = max(sum(self._dyn_nshort(C) * D for C, D in longs), 1)
            self._dyn_S = self._dyn_clamp(
                math.sqrt(self._dyn_brb * sum_DC / denom), None)

    def _dyn_estimate_step_batch(self, cstar: int) -> set:
        """Rough estimate of the request_ids that will RUN this step: running
        carryover + waiting admitted FCFS (short band + long band, each by
        arrival) by cumulative KV footprint until capacity / the θ cap is hit.
        Read-only.  Rough by design -- mode F only needs the long requests'
        *presence* in the batch (0/1), not exact sizing, so a cheap FCFS-by-
        footprint walk suffices.  HITPRED-aware: uses the same cross-tier hit as
        precompute so returning rounds are classified by their real remaining."""
        bs = self.block_size
        pred = {r.request_id for r in self.running}
        ff = getattr(self, "_licht_v2_future_free", None)
        try:
            avail = int(ff[0]) if ff else \
                self.kv_cache_manager.block_pool.get_num_free_blocks()
        except Exception:
            avail = self._total_kv_blocks
        _theta = os.environ.get("LICHT_LONG_THETA")
        cap_long = (float(_theta) * self._total_kv_blocks
                    if _theta else float(self._total_kv_blocks))

        def _rem(r):                                # real remaining (HITPRED-aware)
            _hd = self._sched_hit_pred.get(r.request_id, r.num_computed_tokens)
            return r.num_tokens - _hd

        def _fp(r):                                 # resident KV footprint (blocks)
            return (r.num_tokens + bs - 1) // bs

        long_used = sum(_fp(r) for r in self.running if _rem(r) > cstar)
        used = 0
        _wait = [r for r in self.waiting if _rem(r) > 0]
        _sh = sorted((r for r in _wait if _rem(r) <= cstar),
                     key=lambda r: r.arrival_time)      # short band, FCFS
        _lo = sorted((r for r in _wait if _rem(r) > cstar),
                     key=lambda r: r.arrival_time)      # long band, FCFS
        _order = os.environ.get("LICHT_LONGCAP_ORDER", "long")
        for r in (_sh + _lo) if _order == "short" else (_lo + _sh):
            f = _fp(r)
            if used + f > avail:
                break                               # KV full: nothing more fits
            if _rem(r) > cstar:
                if long_used + f > cap_long:
                    continue                        # long-lane θ cap: skip long
                long_used += f
            pred.add(r.request_id)
            used += f
        return pred

    def _dyn_nshort(self, remaining: int) -> int:
        return bisect.bisect_left(self._dyn_sorted, remaining)

    def _dyn_clamp(self, c: float, remaining) -> int:
        c = int(c) // 16 * 16                  # align to 16-token KV block
        c = max(c, self._dyn_floor)
        if remaining is not None:
            c = min(c, remaining)
        return c

    def _dyn_kmax(self, request: Request, s: int, remaining: int) -> int:
        """Per-request chunk-count cap (LICHT_DYN_KMAX, 0=off): raise the chunk
        to at least C0/k_max (C0 = total new prefill at admission) so no request
        is split into more than k_max pieces -> bounds the giant-request re-read
        blowup (protects tail) even when S* is aggressive (low p50)."""
        k = int(os.environ.get("LICHT_DYN_KMAX", "0"))
        if k > 0:
            c0 = request.num_tokens - self.licht_v2_num_computed_at_admit.get(
                request.request_id, request.num_computed_tokens)
            if c0 > 0:
                s = max(s, (c0 + k - 1) // k)
        return min(s, remaining)

    def _dyn_degenerate(self, request: Request, remaining: int) -> int:
        """S*=0 退化兜底 (防"未来 S*=0 -> 全整段 -> 打穿 KV 池"):
        已在 running 的请求(有 last_chunk 记录)-> max(上一步 chunk, C0/kmax)
        有界续切、不整段、与 timeline 预订一致; 本 step 新请求(无记录)-> 整段
        (其完整 footprint 已在准入闸门过 θ 帽, 整段安全)。"""
        _last = self._dyn_last_chunk.get(request.request_id)
        if _last is None:                       # 新请求: 整段(准入已按 footprint 过帽)
            return 0
        k = int(os.environ.get("LICHT_DYN_KMAX", "0"))
        _kf = 0
        if k > 0:
            c0 = request.num_tokens - self.licht_v2_num_computed_at_admit.get(
                request.request_id, request.num_computed_tokens)
            if c0 > 0:
                _kf = (c0 + k - 1) // k
        return min(max(_last, _kf), remaining)

    def _trace_ev(self, kind: str, **f) -> None:
        if self._trace_f is None:
            return
        try:
            f["kind"] = kind
            self._trace_f.write(json.dumps(f) + "\n")
            if kind == "step":
                self._trace_f.flush()
        except Exception:
            pass

    def _licht_dyn_cap(self, request: Request, remaining: int) -> int:
        # ★ 计划钉死: 已准入的请求返回准入那一刻快照的 cap (与 timeline 预订
        # 一致), 不随全局 S* 漂移. 未准入的候选走 live (用当前 S* 评估). 判据
        # = request_id 是否已在 num_computed_at_admit 里 (= 已 admit). 见
        # licht_v2_dyn_cap_at_admit 注释.
        if self._dyn_pin_cap:
            _pinned = self.licht_v2_dyn_cap_at_admit.get(request.request_id)
            if _pinned is not None:
                return _pinned
        cap = self._licht_dyn_cap_live(request, remaining)
        if cap > 0:                             # 记本步 chunk, 供未来 S*=0 复用
            self._dyn_last_chunk[request.request_id] = cap
        return cap

    def _licht_dyn_cap_live(self, request: Request, remaining: int) -> int:
        # 0 = no cap (run whole).  When disabled, fall back to the static
        # global threshold so existing behaviour is byte-for-byte unchanged.
        mode = getattr(self, "_dyn_mode", None)
        if mode not in ("A", "B", "C", "D", "E", "F"):
            return self.scheduler_config.long_prefill_token_threshold
        if mode in ("E", "F"):
            # short (lambda=0, C<=Clow) -> run whole.  Else chunk at S* with the
            # per-request k_max cap; S*=0 -> _dyn_degenerate (running: bounded
            # reuse of last chunk; new: whole) -- prevents "future S*=0 -> everyone
            # runs whole -> KV pool overflow" without pin-cap's fat tail.
            if remaining <= getattr(self, "_dyn_clow", 2048):
                return 0
            if self._dyn_S > 0:
                return self._dyn_kmax(request, self._dyn_S, remaining)
            return self._dyn_degenerate(request, remaining)
        if remaining <= self._dyn_cstar:       # short request: never chunk
            return 0
        if mode == "A":
            denom = max(self._dyn_ntotal - 2, 1)
            return self._dyn_clamp(
                math.sqrt(self._dyn_brb * remaining / denom), remaining)
        if mode == "B":
            denom = max(self._dyn_nshort(remaining), 1)
            return self._dyn_clamp(
                math.sqrt(self._dyn_brb * remaining / denom), remaining)
        # C / D: one batch-shared S with per-request k_max cap; S*=0 -> degenerate
        # fallback (running: bounded reuse of last chunk; new: run whole).
        return self._dyn_kmax(request, self._dyn_S, remaining) \
            if self._dyn_S > 0 else self._dyn_degenerate(request, remaining)

    def _peek_waiting_request(self) -> Request:
        if self.licht_prefill_sched_enabled:
            now = time.time()
            return max(
                self.waiting,
                key=lambda req: (
                    self._compute_licht_prefill_score(req, now),
                    -req.arrival_time,
                    req.request_id,
                ),
            )

        if self.licht_decode_fcfs_enabled:
            return min(
                self.waiting,
                key=lambda req: (req.arrival_time, req.request_id),
            )

        if self.policy == SchedulingPolicy.FCFS:
            return self.waiting.peek_request()
        if self.policy == SchedulingPolicy.PRIORITY:
            return self.waiting.peek_request()
        if self.policy == SchedulingPolicy.CONTINUUM:
            return self.waiting.peek_request(self.pinned_requests,
                                             self.kv_cache_manager,
                                             self.connector)

        raise ValueError(f"Invalid policy: {self.policy}")

    def _pick_preempt_victim_licht(
        self,
        scheduler_request: Request,
    ) -> Optional[Request]:
        """LICHT-aware preempt victim selection.

        Symmetric counterpart to _peek_waiting_request: where the selector
        admits the highest-scoring waiter, this method evicts the running
        request that is cheapest to evict under a weighted three-factor
        model.  Each factor is rank-normalised within the current running
        pool to [0, 1] (low = "more evictable"), then combined:

            EvictScore = 0.5 * rank_credit
                       + 0.2 * rank_preempt_count
                       + 0.3 * rank_real_computed

        - rank_credit:       LICHT prefill score (low → low priority)
        - rank_preempt_count: how many times already victimised
                              (low → hasn't been hit yet, fresh target)
        - rank_real_computed: computed tokens beyond prefix-cache hit
                              (low → little GPU work to throw away)

        scheduler_request is excluded from the pool (no self-preempt).
        Returns None iff the pool is empty (only the caller left in
        running).
        """
        candidates = [r for r in self.running if r is not scheduler_request]
        if not candidates:
            return None

        now = time.time()
        # Min-run grace (P0 fix): exclude requests admitted within the
        # last LICHT_PREEMPT_MIN_RUN_GRACE_S seconds.  Without this, a
        # request that LICHT just selected from waiting (rank_credit
        # high enough to win admission) is immediately the lowest-ranked
        # member of running on rank_credit AND rank_computed (computed=0,
        # has done no work yet), so the next allocate_slots failure
        # evicts it back out — the same "admit-then-evict" thrash that
        # FCFS-pop-tail used to cause, just via a different path.
        grace_s = self.LICHT_PREEMPT_MIN_RUN_GRACE_S
        seasoned = [
            r for r in candidates
            if (now - self.licht_running_admit_ts.get(r.request_id, 0.0))
                >= grace_s
        ]
        if seasoned:
            candidates = seasoned
        # else: every running request is within grace.  Fall back to the
        # full pool — we still need to free a block somewhere, and the
        # weighted score below will pick the least-bad option.
        if len(candidates) == 1:
            return candidates[0]
        n = len(candidates)

        def _real_computed(r: Request) -> int:
            # Strip prefix-cache contribution: that KV wasn't recomputed
            # on our GPU, so losing it costs nothing.
            cached = r.num_cached_tokens if r.num_cached_tokens > 0 else 0
            return max(r.num_computed_tokens - cached, 0)

        # Ascending rank → index 0 means "most evictable" on that axis.
        by_credit = sorted(
            candidates,
            key=lambda r: self._compute_licht_prefill_score(r, now),
        )
        by_preempt = sorted(candidates, key=lambda r: r.preempt_count)
        by_computed = sorted(candidates, key=_real_computed)

        rank_credit = {id(r): i / (n - 1) for i, r in enumerate(by_credit)}
        rank_preempt = {id(r): i / (n - 1) for i, r in enumerate(by_preempt)}
        rank_computed = {
            id(r): i / (n - 1)
            for i, r in enumerate(by_computed)
        }

        def _evict_score(r: Request) -> float:
            return (0.5 * rank_credit[id(r)]
                    + 0.2 * rank_preempt[id(r)]
                    + 0.3 * rank_computed[id(r)])

        # Tie-break: prefer evicting the newer arrival, then request_id
        # for deterministic ordering.
        return min(
            candidates,
            key=lambda r: (_evict_score(r), -r.arrival_time, r.request_id),
        )

    def _neutralize_lookup_prefix_stats(self, request: Request,
                                        local_tokens: int) -> None:
        """★ prefix-hit 指标改到 admit 点计数 (2026-07-13)。上游把计数放在
        get_computed_blocks (lookup 时) —— 上游 FCFS 里 lookup≈admit, 等价; 本
        fork 的 backfill 准入每步扫 O(队列) 个候选且大多不 admit, lookup 计数把
        没 admit 的候选反复算进命中率: 慢性 = 队头长请求每步重复计数 (打印值比
        真实复用低 ~10 个点); 急性 = can_admit 失败关 lane 那步全队列扫描, 一步
        灌 ~2M 低命中 queries 把 1000-request 窗口砸穿 (76%→66% 假暴跌)。旧方案
        按 skip 路径逐条撤销 (_uncount), 注定漏; 现在 gcb 一返回就无条件原地中和,
        真正的计数移到 admit 成功点 (_count_prefix_stats_at_admit)。与
        kv_cache_manager.get_computed_blocks 的计数严格对称相减。"""
        if (self.kv_cache_manager.log_stats
                and self.kv_cache_manager.prefix_cache_stats is not None):
            _pcs = self.kv_cache_manager.prefix_cache_stats
            _pcs.requests -= 1
            _pcs.queries -= request.num_tokens
            _pcs.hits -= local_tokens

    def _count_prefix_stats_at_admit(self, request: Request,
                                     num_computed_tokens: int) -> None:
        """★ admit 点计数: 每次准入恰好计一次 —— queries = 完整 prompt (含
        resume 时的已生成 tokens, 与上游口径一致), hits = admit 时刻真实免算的
        tokens (本地 + arena/外部, 即 trace 里的 hit_length 同一口径)。
        load_kv_async 请求首过转 WAITING_FOR_REMOTE_KVS 不计 (还没进 running);
        回来 resume 那次 num_computed>0 不走 gcb, 在这里计, 无重复。"""
        if (self.kv_cache_manager.log_stats
                and self.kv_cache_manager.prefix_cache_stats is not None):
            _pcs = self.kv_cache_manager.prefix_cache_stats
            _pcs.requests += 1
            _pcs.queries += request.num_tokens
            _pcs.hits += num_computed_tokens

    def _pop_waiting_request(self, request: Request) -> None:
        # LICHT custom selection may not choose queue head, so remove by object.
        if self.licht_prefill_sched_enabled or self.licht_decode_fcfs_enabled:
            self.waiting.remove_request(request)
            return

        if self.policy == SchedulingPolicy.CONTINUUM:
            self.waiting.pop_request(self.pinned_requests,
                                     self.kv_cache_manager,
                                     self.connector)
            return

        self.waiting.pop_request()

    # ------------------------------------------------------------------
    # LICHTV2 backfill-window helpers
    # ------------------------------------------------------------------
    # All four helpers below are no-ops unless
    # licht_v2_prefill_sched_enabled is True.  They share the timeline
    # state stored on `self`:
    #   self._licht_v2_future_free  : list[int], length N+1
    #   self._licht_v2_future_alloc : list[int], length N+1
    # which are rebuilt at the start of each waiting-loop pass via
    # _licht_v2_build_timeline().

    def _licht_v2_chunk_size(self) -> int:
        chunk = self.scheduler_config.long_prefill_token_threshold
        return chunk if chunk > 0 else self.scheduler_config.max_num_batched_tokens

    def _licht_v2_chunk_for(self, request: Request, remaining: int) -> int:
        """Per-request per-step chunk the timeline should assume — the SAME
        value the real prefill uses, so future_free/R_at match reality.

        When dynamic_chunk is active this is _licht_dyn_cap (short -> run whole
        in 1 step; long -> S*).  Otherwise the static threshold.  cap==0 means
        "run whole" -> 1 step -> chunk = full remaining.
        """
        cap = self._licht_dyn_cap(request, remaining)
        if cap <= 0:
            return max(remaining, 1)
        return cap

    def _licht_v2_R_at(self, request: Request, current_offset: int) -> int:
        """Remaining prefill scheduler steps starting from `current_offset`.

        For running requests, pass `request.num_computed_tokens` so R
        decreases as the request makes progress.  For a candidate that
        is about to be admitted, pass the value that will become its
        num_computed_tokens (typically the prefix-cache hit count).
        """
        remaining = max(request.num_tokens - current_offset, 0)
        if remaining <= 0:
            return 0
        chunk = self._licht_v2_chunk_for(request, remaining)
        return (remaining + chunk - 1) // chunk

    def _licht_v2_release_blocks(
            self, request: Request, num_computed_at_admit: int) -> int:
        """Blocks that go back to the free pool when prefill completes.

        Anchored at admission so it stays constant across scheduler
        steps even as `num_computed_tokens` advances.  Excludes prefix-
        cache shared portion (those blocks are refcount-managed and
        don't deterministically return to the free pool).
        """
        bs = self.block_size
        net_tokens = max(request.num_tokens - num_computed_at_admit, 0)
        return (net_tokens + bs - 1) // bs

    def _licht_v2_B_at(self, request: Request, current_offset: int,
                       t: int) -> int:
        """Blocks newly allocated by `request` at future step `t`,
        relative to NOW (offset = current num_computed for the request).

        B(t) = ⌈cum(t)/block_size⌉ - ⌈cum(t-1)/block_size⌉
        where cum(t) = current_offset + min(chunk*(t+1), remaining_now).

        For running requests, current_offset must be the current
        `num_computed_tokens` so already-allocated chunks are not
        double-counted.  For candidates, current_offset is the
        will-be-admitted num_computed (= the prefix-cache hit count).
        """
        Ri = self._licht_v2_R_at(request, current_offset)
        if not (0 <= t < Ri):
            return 0
        bs = self.block_size
        remaining_now = max(request.num_tokens - current_offset, 0)
        chunk = self._licht_v2_chunk_for(request, remaining_now)
        cum_t = current_offset + min(chunk * (t + 1), remaining_now)
        cum_prev = current_offset + (
            min(chunk * t, remaining_now) if t > 0 else 0)
        return max(((cum_t + bs - 1) // bs) - ((cum_prev + bs - 1) // bs),
                   0)

    def _licht_v2_get_admit_anchor(self, request_id: str) -> int:
        """Look up the num_computed_at_admit snapshot for a running req.

        Falls back to 0 if the request was admitted before LICHTV2 was
        active (shouldn't happen in normal operation but keeps the math
        safe).
        """
        return self.licht_v2_num_computed_at_admit.get(request_id, 0)

    def _licht_v2_get_evictable_prefix(self, request_id: str) -> int:
        """Look up the evictable-prefix-blocks snapshot for a running req."""
        return self.licht_v2_evictable_prefix_at_admit.get(request_id, 0)

    # 回传 REMOVED (2026-05-21): _v3_should_defer_for_pushback and
    # _v3_full_pushback_prefix_blocks deleted (defer + full-prefix anchor).

    def _licht_v2_count_evictable_prefix(self, new_computed_blocks) -> int:
        """Count blocks in `new_computed_blocks` that are currently in
        the free queue (ref_cnt == 0 and not null).

        These are the blocks that allocate_slots()'s
        get_num_blocks_to_allocate() adds to its required-blocks count
        (single_type_kv_cache_manager.py:83-86): touching them at admit
        will remove them from the free queue, so the timeline must
        subtract them at t=0 in addition to B_at(t=0).
        """
        if new_computed_blocks is None:
            return 0
        # KVCacheBlocks wraps a tuple of per-group block lists; raw
        # tuples/lists are also accepted for safety.
        block_lists = getattr(new_computed_blocks, "blocks",
                              new_computed_blocks)
        count = 0
        for blocks_per_group in block_lists:
            for blk in blocks_per_group:
                if blk.ref_cnt == 0 and not blk.is_null:
                    count += 1
        return count

    def _licht_v2_build_timeline(self, current_free: int) -> None:
        """Recompute future_free / future_alloc from current running state.

        IMPORTANT: this MUST be called BEFORE the running loop has
        allocated any blocks for this scheduler step.  In that ordering:

          - current_free reflects PRE-this-step state
          - request.num_computed_tokens is also the PRE-this-step value
          - At timeline t=0 we subtract running's "first future chunk"
            B_at(stale, 0), which IS this step's chunk — exactly the
            block that the running loop is about to allocate
          - future_free[0] therefore predicts the state at the END of
            this step (post-running-alloc, pre-waiting-alloc)

        Candidates' apply_to_timeline runs after each successful admit
        in the waiting loop, further decrementing future_free[0..] for
        each admitted candidate's own this-step alloc.  At end of the
        scheduler step, future_free[0] should match the actual physical
        free block count.

        Release events are reflected at t = R (one step after the
        request's final chunk alloc).  This 1-step delay models the
        BLOCK_MIGRATE pipeline (last chunk forward → KV migration →
        RELEASE round-trip → fast_release_poll), which empirically
        spans one full scheduler step in our setup.  Modeling release
        any earlier (e.g. at t=R-1) leads to over-optimistic admits
        and a sharp rise in preempts.
        """
        N = self.LICHTV2_N
        future_free = [0] * (N + 1)
        future_alloc = [0] * (N + 1)
        prev = current_free

        # Decode requests' block usage is already excluded from
        # `current_free`; the timeline assumes decode does not release
        # blocks within the lookahead horizon (conservative).
        running_prefill = [
            r for r in self.running
            if r.num_computed_tokens < r.num_tokens
        ]

        for t in range(0, N + 1):
            delta_free = 0
            delta_alloc = 0
            for r in running_prefill:
                cur = r.num_computed_tokens
                Ri = self._licht_v2_R_at(r, cur)
                if t < Ri:
                    # Alloc event: each in-progress chunk decrements free.
                    bit = self._licht_v2_B_at(r, cur, t)
                    delta_free -= bit
                    delta_alloc += bit
                elif t == Ri:
                    # Release event at t = Ri (1-step delay after last
                    # chunk).  We hand back BOTH the request's net new
                    # alloc and the evictable prefix blocks it touched
                    # at admit (the latter were taken out of the free
                    # queue at admit and now flow back as ref_cnt drops
                    # to 0).
                    admit_anchor = self._licht_v2_get_admit_anchor(
                        r.request_id)
                    ev = self._licht_v2_get_evictable_prefix(r.request_id)
                    delta_free += (self._licht_v2_release_blocks(
                        r, admit_anchor) + ev)
                # t > Ri: contributes nothing
            future_free[t] = prev + delta_free
            future_alloc[t] = delta_alloc
            prev = future_free[t]

        self._licht_v2_future_free = future_free
        self._licht_v2_future_alloc = future_alloc

        # guard scheme: when small (1-step, C<=chunk) requests are waiting,
        # reserve KV for their footprint so big (multi-step) requests can't fill
        # all blocks and starve them.  Reserve = K * largest waiting-small
        # footprint (block) — small reqs free their KV after 1 step so the
        # reserve recycles.  Computed once per step.
        self._licht_guard_reserve = 0
        if os.environ.get("LICHT_SCHED_SCHEME") == "guard":
            # "small" is a FIXED token threshold (decoupled from chunk_size, so
            # the guard works with coarse chunks too).  Reserve = K * largest
            # waiting-small footprint; smalls free their KV fast so it recycles.
            small_c = int(os.environ.get("LICHT_GUARD_SMALL_C", "1024"))
            K = int(os.environ.get("LICHT_GUARD_K", "2"))
            bs = self.block_size
            mx = 0
            for w in self.waiting:
                rem = w.num_tokens - w.num_computed_tokens
                if 0 < rem <= small_c:                  # small = new tokens <= SMALL_C
                    fb = (w.num_tokens + bs - 1) // bs   # footprint L+C in blocks
                    if fb > mx:
                        mx = fb
            self._licht_guard_reserve = K * mx

        # sjf_reservation: find the oldest waiting big (C>thr) job; reserve its
        # KV footprint against other admits so it isn't starved (SLURM backfill).
        self._resv_head_id = None
        self._resv_head_blk = 0
        if os.environ.get("LICHT_SCHED_SCHEME") == "sjf_reservation":
            thr = int(os.environ.get("LICHT_LONG_C", "5120"))
            bs = self.block_size
            best = None
            for w in self.waiting:
                if (w.num_tokens - w.num_computed_tokens) > thr:
                    if best is None or w.arrival_time < best.arrival_time:
                        best = w
            if best is not None:
                self._resv_head_id = best.request_id
                self._resv_head_blk = (best.num_tokens + bs - 1) // bs

        # longcap reservation-backfill (LICHT_LONG_RESV=1): give the oldest
        # waiting BIG request (C>thr) a future slot T* (earliest step where the
        # timeline frees enough blocks), so small backfill can't endlessly
        # steal its space.  Only when it's blocked by CAPACITY (timeline) and
        # NOT by theta (theta-block is intentional throttling -> let it wait).
        self._resv2_id = None
        self._resv2_blk = 0
        self._resv2_T = 0
        _sch2 = os.environ.get("LICHT_SCHED_SCHEME")
        if (_sch2 in ("longcap_sjf", "longcap_fcfs")
                and os.environ.get("LICHT_LONG_RESV") == "1"):
            thr = int(os.environ.get("LICHT_LONG_C", "5120"))
            bs = self.block_size
            ff = self._licht_v2_future_free
            best = None
            for w in self.waiting:
                if (w.num_tokens - w.num_computed_tokens) > thr:
                    if best is None or w.arrival_time < best.arrival_time:
                        best = w
            if best is not None:
                F_big = (best.num_tokens + bs - 1) // bs
                # theta-block? (long lane already full) -> no reservation.
                _theta = os.environ.get("LICHT_LONG_THETA")
                theta_blocked = False
                if _theta is not None:
                    cur_long = sum(
                        (r.num_tokens + bs - 1) // bs for r in self.running
                        if (r.num_tokens - r.num_computed_tokens) > thr)
                    theta_blocked = (cur_long > 0 and
                                     cur_long + F_big >
                                     float(_theta) * self._total_kv_blocks)
                # only reserve if it can't fit NOW but CAN within the horizon.
                if not theta_blocked and ff and ff[0] < F_big:
                    for t in range(len(ff)):
                        if ff[t] >= F_big:
                            self._resv2_id = best.request_id
                            self._resv2_blk = F_big
                            self._resv2_T = t
                            break

    def _licht_v2_count_long_running(self) -> int:
        """Count running prefill requests with R(i) > N (long-tail).

        Uses CURRENT num_computed (not admit anchor) so requests that
        have shrunk below the threshold while running are no longer
        counted as long-tail.
        """
        N = self.LICHTV2_N
        count = 0
        for r in self.running:
            if r.num_computed_tokens >= r.num_tokens:
                continue
            if self._licht_v2_R_at(r, r.num_computed_tokens) > N:
                count += 1
        return count

    def _licht_v2_can_admit(
            self, request: Request,
            num_computed_at_admit: int,
            evictable_prefix: int = 0) -> bool:
        """Three-guard admission check against the current timeline.

        Guard 1 (long-tail concurrency cap): if R(j) > N, the number of
            already-running long-tail requests + 1 must not exceed
            LICHTV2_MAX_LONG_BRIDGE.
        Guard 2 (block timeline + headroom): for every t in [0, N],
            future_free[t] + cum_delta_j(t) >= threshold, with threshold
            = LICHTV2_LONG_TAIL_HEADROOM_RATIO * total_blocks for
            long-tail candidates and 0 otherwise.
        Guard 3 (per-step alloc budget): for every t in [0, R(j)-1],
            future_alloc[t] + B_j(t) <= max_num_batched_tokens /
            block_size.

        `evictable_prefix` is the count of prefix-cache blocks currently
        in the free queue (ref_cnt==0) that this candidate's
        new_computed_blocks will touch on admit.  These leave the free
        queue at admit (additional t=0 consumption beyond B_at(0)) and
        flow back at t=R when the request's ref drops to 0.
        """
        N = self.LICHTV2_N
        # For a candidate, the "current offset" at the moment of admit
        # equals num_computed_at_admit (= prefix-cache hit count).
        Rj = self._licht_v2_R_at(request, num_computed_at_admit)
        if Rj <= 0:
            # Nothing to schedule (already done) — let the regular path
            # handle it.
            return True

        long_tail = Rj > N
        _capr = os.environ.get("LICHT_ADMIT_PROBE")
        _capr_on = (os.environ.get("LICHT_ADMIT_PROBE_ALL") == "1") or (
            bool(_capr) and _capr in (getattr(request, "job_id", "") or ""))
        if _capr_on and os.environ.get("LICHT_ADMIT_PROBE_ALL") != "1":
            logger.info("PROBE-GUARD enter rid=%s Rj=%d longtail=%s",
                        request.request_id, Rj, long_tail)

        # Guard 1: long-tail concurrency cap.
        if long_tail:
            long_count = self._licht_v2_count_long_running()
            if long_count + 1 > self.LICHTV2_MAX_LONG_BRIDGE:
                if _capr_on:
                    logger.info("PROBE-GUARD reject=Guard1_MAXLONGBRIDGE "
                                "long_count=%d max=%d", long_count,
                                self.LICHTV2_MAX_LONG_BRIDGE)
                return False

        threshold = (int(self.LICHTV2_LONG_TAIL_HEADROOM_RATIO
                         * self._total_kv_blocks)
                     if long_tail else 0)
        # guard scheme: a big candidate (new tokens > SMALL_C) must leave the
        # small-request reserve free so waiting small requests don't starve.
        if os.environ.get("LICHT_SCHED_SCHEME") == "guard":
            small_c = int(os.environ.get("LICHT_GUARD_SMALL_C", "1024"))
            if request.num_tokens - num_computed_at_admit > small_c:
                threshold = max(threshold, getattr(self, "_licht_guard_reserve", 0))
        _sch = os.environ.get("LICHT_SCHED_SCHEME")
        if _sch in ("longcap_sjf", "longcap_fcfs"):
            # Cap the LONG (C>thr) lane.  Only long candidates are checked here;
            # short reqs are never blocked by this (they use whatever the
            # feasibility guards below allow -> no (1-theta) floor reservation).
            thr = int(os.environ.get("LICHT_LONG_C", "5120"))
            _is_long = request.num_tokens - num_computed_at_admit > thr
            _short_cap = os.environ.get("LICHT_SHORT_CAP") == "1"
            if _is_long and not _short_cap:
                # DEFAULT mode: cap the LONG lane (theta/N); shorts uncapped.
                # aging (LICHT_LONG_AGE_S): a long that has waited longer than
                # T_age bypasses the theta cap -> bounds the worst-case long
                # wait (pulls p99 down) while theta stays tight for the common
                # case (keeps p50 low).  off when env unset.
                _age = os.environ.get("LICHT_LONG_AGE_S")
                _aged = False
                if _age is not None:
                    ws = self.licht_waiting_round_start_ts.get(
                        request.request_id, request.arrival_time)
                    if (time.time() - ws) >= float(_age):
                        _aged = True
                _theta = os.environ.get("LICHT_LONG_THETA")
                if _aged:
                    pass                                 # aged -> skip theta cap
                elif _theta is not None:
                    bs = self.block_size
                    cand = (request.num_tokens + bs - 1) // bs
                    # head-of-line: if no long is running yet, admit anyway (a
                    # request larger than any cap still runs alone -> no permanent
                    # starvation; feasibility guards below keep safety).
                    any_long = any((r.num_tokens - r.num_computed_tokens) > thr
                                   for r in self.running)
                    # θ-relax (LICHT_LONG_THETA_RELAX=1): 本步没有短请求被挡下
                    # (短请求都进去了) → θ 帽没有保护对象 → 松开 θ, 让长请求把
                    # 空着的 KV 填满 (物理 future-free 仍兜底, 装不下才停)。短请求
                    # order=short 优先 peek, 走到长请求时短请求都已处理完, 故
                    # "没短请求被 skip" 即"短请求全进去了"。
                    _theta_relax = (
                        os.environ.get("LICHT_LONG_THETA_RELAX") == "1"
                        and not getattr(self, "_licht_short_skipped", False))
                    if os.environ.get("LICHT_LONGCAP_FOOTPRINT", "1") == "1":
                        # FOOTPRINT theta cap: bound the LONG lane to theta of total
                        # KV by PROJECTED footprint (sum of running longs' full L+C
                        # blocks + this candidate).  Unlike the SHORT_RESERVE branch,
                        # this counts the FUTURE occupancy of longs still mid-prefill,
                        # so admitting many longs while KV is momentarily low can NOT
                        # over-commit and starve shorts a few seconds later (the bug
                        # that turned shorts into long recomputes once their prefix
                        # aged out of the arena).
                        cur_long = sum((r.num_tokens + bs - 1) // bs
                                       for r in self.running
                                       if (r.num_tokens - r.num_computed_tokens) > thr)
                        # θ-relax 有界版: 正常长请求投影 footprint <= θ (30%);
                        # 本步没短请求被挡时放宽到 <= LICHT_LONG_THETA_RELAX_CAP
                        # (默认 0.8), 但【仍是帽, 不跳过】-> 永远给短请求留
                        # (1-cap)=20% (PD 分离无 decode 占用 => 该不变式 == "长请求
                        # 投影 footprint <= 80%"), 下一步短请求可快速插队, 不会像
                        # 旧 θ-relax 填满物理容量把短请求饿死。
                        cap_long = float(_theta) * self._total_kv_blocks
                        if _theta_relax:
                            cap_long = float(os.environ.get(
                                "LICHT_LONG_THETA_RELAX_CAP", "0.8")
                            ) * self._total_kv_blocks
                        if any_long and cur_long + cand > cap_long:
                            if _capr_on:
                                logger.info("PROBE-GUARD reject=LONG_THETA_FP "
                                            "cur_long=%d cand=%d cap=%d relax=%d",
                                            cur_long, cand, int(cap_long),
                                            int(_theta_relax))
                            return False
                        # physical feasibility within the timeline horizon: don't
                        # admit a long the predicted free blocks can't hold now.
                        ff = getattr(self, "_licht_v2_future_free", None)
                        if any_long and ff and ff[0] < cand:
                            if _capr_on:
                                logger.info("PROBE-GUARD reject=LONG_FUTURE_FREE "
                                            "ff0=%d cand=%d", ff[0], cand)
                            return False
                    else:
                        # SHORT-RESERVE throttle (legacy patch; default).  Only
                        # throttle a NEW long when TOTAL ACTUAL KV usage would push
                        # past (1 - reserve).  Uses CURRENT usage, NOT projected
                        # footprint -> over-admits longs while KV sits low because it
                        # ignores how big the admitted-but-mid-prefill longs will grow.
                        reserve = float(os.environ.get("LICHT_SHORT_RESERVE", "0.2"))
                        used = self.kv_cache_manager.usage * self._total_kv_blocks
                        ceiling = (1.0 - reserve) * self._total_kv_blocks
                        if any_long and used + cand > ceiling and not _theta_relax:
                            if _capr_on:
                                logger.info("PROBE-GUARD reject=SHORT_RESERVE used=%d "
                                            "cand=%d ceiling=%d reserve=%.2f",
                                            int(used), cand, int(ceiling), reserve)
                            return False
                else:
                    Nlong = int(os.environ.get("LICHT_LONG_N", "2"))
                    nbig = sum(1 for r in self.running
                               if (r.num_tokens - r.num_computed_tokens) > thr)
                    if nbig >= Nlong:
                        if _capr_on:
                            logger.info("PROBE-GUARD reject=N_CAP nbig=%d N=%d", nbig, Nlong)
                        return False
            elif (not _is_long) and _short_cap:
                # SHORT-CAP mode (mirror of longs-first): shorts get priority but
                # collectively hold <= (1-theta) * total KV.  Longs are UNCAPPED
                # (theta not applied to them) and take whatever shorts leave
                # (>= theta), so a short flood can never starve longs.  head-of-
                # line safe: when no short is running (cur_s==0) admit anyway.
                _theta_s = os.environ.get("LICHT_LONG_THETA")
                if _theta_s is not None:
                    bs = self.block_size
                    cur_s = sum((r.num_tokens + bs - 1) // bs for r in self.running
                                if (r.num_tokens - r.num_computed_tokens) <= thr)
                    cand_s = (request.num_tokens + bs - 1) // bs
                    cap_s = (1.0 - float(_theta_s)) * self._total_kv_blocks
                    if cur_s > 0 and cur_s + cand_s > cap_s:
                        if _capr_on:
                            logger.info("PROBE-GUARD reject=SHORT_CAP")
                        return False
            # reservation-backfill: protect the head big's slot T* (set in
            # build_timeline). Other bigs head-of-line behind it; smalls may
            # backfill only if they don't delay the big's start T*.
            if (getattr(self, "_resv2_id", None) is not None
                    and request.request_id != self._resv2_id):
                if request.num_tokens - num_computed_at_admit > thr:
                    if _capr_on:
                        logger.info("PROBE-GUARD reject=RESV_OTHER_BIG "
                                    "resv_head=%s resv_blk=%d resv_T=%d",
                                    self._resv2_id, self._resv2_blk, self._resv2_T)
                    return False           # no other big past the reserved head
                T = min(self._resv2_T, len(self._licht_v2_future_free) - 1)
                free_at_T = self._licht_v2_future_free[T]
                bs = self.block_size
                F_small = (request.num_tokens + bs - 1) // bs
                # Rj = candidate's steps-to-finish (releases its KV at t=Rj).
                # If it still holds blocks at T*, it must leave slack >= F_big.
                if Rj > T and free_at_T - F_small < self._resv2_blk:
                    if _capr_on:
                        logger.info("PROBE-GUARD reject=RESV_BACKFILL")
                    return False
        if _sch == "sjf_reservation":
            # reserve the head big job's footprint against all other candidates.
            if request.request_id != getattr(self, "_resv_head_id", None):
                threshold = max(threshold, getattr(self, "_resv_head_blk", 0))
        max_alloc_per_step = (
            self.scheduler_config.max_num_batched_tokens // self.block_size)

        # Guard 2 + Guard 3: walk the horizon with cumulative delta.
        # Release event at t = Rj (1-step delay after last chunk alloc),
        # which models the BLOCK_MIGRATE round-trip empirically spanning
        # one scheduler step.  Alloc at [0, Rj-1] and release at Rj are
        # mutually exclusive in time, so we use if/elif.
        cum_delta = 0
        for t in range(0, N + 1):
            # One-time prefix-touch consumption at admit step.  Mirrors
            # what allocate_slots()'s num_evictable_computed_blocks does
            # to num_blocks_to_allocate.
            if t == 0:
                cum_delta -= evictable_prefix
            bit_j = 0
            if t < Rj:
                bit_j = self._licht_v2_B_at(request,
                                            num_computed_at_admit, t)
                cum_delta -= bit_j
            elif t == Rj:
                cum_delta += (self._licht_v2_release_blocks(
                    request, num_computed_at_admit) + evictable_prefix)
            # else: cum_delta unchanged (j has finished and released)

            # Guard 2: block availability (with optional headroom).
            if self._licht_v2_future_free[t] + cum_delta < threshold:
                if _capr_on:
                    logger.info("PROBE-GUARD reject=GUARD2_BLOCKAVAIL t=%d "
                                "future_free=%d cum_delta=%d threshold=%d",
                                t, self._licht_v2_future_free[t], cum_delta, threshold)
                return False

            # Guard 3: per-step alloc budget (only applies during j's
            # alloc window).
            if t < Rj:
                if (self._licht_v2_future_alloc[t] + bit_j
                        > max_alloc_per_step):
                    if _capr_on:
                        logger.info("PROBE-GUARD reject=GUARD3_ALLOCBUDGET t=%d "
                                    "future_alloc=%d bit_j=%d max=%d", t,
                                    self._licht_v2_future_alloc[t], bit_j, max_alloc_per_step)
                    return False

        return True

    def _licht_v2_apply_to_timeline(
            self, request: Request,
            num_computed_at_admit: int,
            evictable_prefix: int = 0) -> None:
        """Commit `request`'s events into the live timeline.

        MUST mirror exactly the cum_delta accumulation used in
        `_licht_v2_can_admit`, otherwise subsequent candidates will see
        an inconsistent timeline.
        """
        N = self.LICHTV2_N
        Rj = self._licht_v2_R_at(request, num_computed_at_admit)
        if Rj <= 0:
            return

        cum_delta = 0
        for t in range(0, N + 1):
            if t == 0:
                cum_delta -= evictable_prefix   # mirror can_admit
            if t < Rj:
                bit = self._licht_v2_B_at(request,
                                          num_computed_at_admit, t)
                cum_delta -= bit
                self._licht_v2_future_alloc[t] += bit
            elif t == Rj:
                cum_delta += (self._licht_v2_release_blocks(
                    request, num_computed_at_admit) + evictable_prefix)
            self._licht_v2_future_free[t] += cum_delta

    def schedule(self) -> SchedulerOutput:
        # P9/兜底: hold _schedule_lock across the whole method so the
        # install-center bg drain (separate thread) can drain reserves
        # during the forward but never concurrently with allocate_slots.
        with self._schedule_lock:
            return self._schedule_impl()

    def _schedule_impl(self) -> SchedulerOutput:
        if self._trace_f is not None:
            self._trace_step += 1
        # LICHTV3 K_queue actual: increment per-call step counter.  Used
        # both for stamping arrival_step on add_request and admit_step
        # when a waiting request is moved into scheduled_new_reqs.
        if self._v3_kqueue_log_enabled:
            self._v3_sched_step += 1

        # beta_r/b probe (env LICHT_BRB_PROBE=path): dt since last schedule()
        # ~= previous step's forward time (prefill-heavy). Pair it with the
        # prev step's (Σc, ΣD, Σ c·D) features stored at the end of schedule.
        _brb_path = os.environ.get("LICHT_BRB_PROBE")
        if _brb_path:
            _bnow = time.perf_counter()
            _pf = getattr(self, "_brb_prev_feat", None)
            if _pf is not None and getattr(self, "_brb_prev_ts", None):
                _pf["dt"] = _bnow - self._brb_prev_ts
                try:
                    with open(_brb_path, "a") as _bf:
                        _bf.write(json.dumps(_pf) + "\n")
                except Exception:
                    pass
            self._brb_prev_ts = _bnow

        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        # Change 3: fast-release side-channel — drain RELEASE signals that
        # arrived between forward steps and free blocks immediately.
        self._poll_fast_releases()

        # 回传 REMOVED: no install center to drain.
        # LICHTV3 decode side: process GPU-tier release callbacks parked
        # by the prewarm thread.  Must run on scheduler thread for
        # block_pool safety (calling _free_blocks from prewarm thread
        # races with scheduler.allocate_slots → linked-list corruption
        # → 13 Running reqs stuck at 0 tok/s, observed in production).
        if self.licht_v3_decode_manager is not None:
            self.licht_v3_decode_manager.drain_pending_releases()

        # Change 6 (bug fix): ensure delay-free request IDs are included in
        # finished_req_ids so they are passed to the worker-side
        # get_finished().  Without this, empty iterations (no scheduled
        # tokens) pass an empty finished_req_ids set, so the worker's
        # get_finished() never checks RELEASE status for these requests.
        # This makes _update_from_kv_xfer_finished a reliable fallback
        # path for freeing delay-free blocks, in addition to the bg
        # thread path.
        if self._delay_free_req_ids:
            self.finished_req_ids.update(self._delay_free_req_ids)

        self.unpin_requests_regular()
        
        #Qiuyang (DEBUG) logging all running queue jobs and waiting queue jobs
        logger.debug(f"Running queue jobs: {[req.request_id for req in self.running]}")
        logger.debug(f"Waiting queue jobs: {[req.request_id for req in self.waiting]}")


        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # Prefix-hit prediction (LICHT_SCHED_HIT_PRED=1): BEFORE scoring /
        # dyn_precompute / admission, predict each waiting request's real
        # cross-tier prefix hit so they are classified by REAL remaining C.
        self._licht_build_sched_hit_pred()

        # LICHTV2: build the backfill window BEFORE the running loop so
        # that current_free / num_computed_tokens are both in their
        # pre-this-step state.  Running's t=0 alloc (= this step's
        # chunk) is captured by the timeline; admitted candidates'
        # apply_to_timeline calls later in the waiting loop further
        # subtract their own this-step allocs.  At end of the scheduler
        # step, future_free[0] should match actual physical free.
        if self.licht_v2_prefill_sched_enabled:
            with self._kv_free_lock:
                _lv2_current_free = (
                    self.kv_cache_manager.block_pool.get_num_free_blocks())
            self._licht_v2_build_timeline(_lv2_current_free)

        # Dynamic chunk: per-step chunk-size precompute (env-gated, no-op off).
        self._licht_dyn_precompute()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)


            _dyn_cap = self._licht_dyn_cap(request, num_new_tokens)
            if (0 < _dyn_cap < num_new_tokens):
                num_new_tokens = _dyn_cap
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_compute_budget
                 ) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_compute_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue
            
            logger.debug(f"Trying to schedule request {request.request_id} for {num_new_tokens} tokens")
            while True:
                with self._kv_free_lock:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is not None:
                    logger.debug(f"New blocks: {new_blocks}")
                else:
                    logger.debug(f"New blocks is None")

                if new_blocks is None:
                    # Delay-free admission control: if there are
                    # delay-free blocks that will be freed soon, skip
                    # preemption and defer this request to the next step.
                    if self._num_delay_free_blocks > 0:
                        can_schedule = False
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    is_unpin = False
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        self.continuum_recorder.request_evicted_from_running_queue(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)

                    # TODO (Hanchen) need to implement CONTINUUM preemption, find the request that is not pinned something is pinned, do not preempt
                    elif self.policy == SchedulingPolicy.CONTINUUM:
                        #NOTE (Hanchen) we need to not evict last step requests
                        preempted_req, is_unpin = self.pop_running_request_based_on_last_step(request)

                        #TODO (Hanchen) we need to add a check unpin requests with the same job id.
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)

                        if preempted_req.request_id in num_scheduled_tokens:
                            del num_scheduled_tokens[preempted_req.request_id]
                        if preempted_req.request_id in req_to_new_blocks:
                            del req_to_new_blocks[preempted_req.request_id]
                        self.continuum_recorder.request_evicted_from_running_queue(preempted_req)
                    elif self.licht_prefill_sched_enabled:
                        # LICHT-aware preempt (Bug 2 fix): pick the
                        # cheapest-to-evict running request by a weighted
                        # rank of LICHT credit, preempt_count and real
                        # computed tokens.  Symmetric to
                        # _peek_waiting_request, so preempts no longer
                        # victimise the exact requests LICHT just picked
                        # (which under FCFS-pop-tail were always at the
                        # running tail).
                        preempted_req = self._pick_preempt_victim_licht(
                            request)
                        if preempted_req is None:
                            # Only `request` is in running; fall through
                            # to the self-preempt path below.
                            preempted_req = request
                        _pb = os.environ.get("LICHT_DYN_PROBE")
                        if _pb and preempted_req.num_prompt_tokens >= int(_pb):
                            logger.info("PROBE PREEMPT t=%.3f rid=%s prompt=%d "
                                        "computed=%d (by rid=%s prompt=%d)",
                                        time.time(), preempted_req.request_id,
                                        preempted_req.num_prompt_tokens,
                                        preempted_req.num_computed_tokens,
                                        request.request_id, request.num_prompt_tokens)
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                        if preempted_req.request_id in num_scheduled_tokens:
                            del num_scheduled_tokens[preempted_req.request_id]
                        if preempted_req.request_id in req_to_new_blocks:
                            del req_to_new_blocks[preempted_req.request_id]
                        self.continuum_recorder.request_evicted_from_running_queue(preempted_req)
                    else:
                        preempted_req = self.running.pop()
                        self.continuum_recorder.request_evicted_from_running_queue(preempted_req)

                    # Phase 1 (save-on-preempt): hand the victim's KV to
                    # the connector for synchronous D2H into the round-kv
                    # arena BEFORE we free its blocks.  The connector's
                    # consumer recovery path will read from arena on the
                    # next admission, avoiding the recompute-thrash that
                    # crashes long-decode requests.  Gated behind the
                    # LICHT_PHASE1_SAVE_ON_PREEMPT env (no-op if off).
                    # Skipped for is_unpin (request stays running) and
                    # for self-preempt with nothing computed.
                    if (not is_unpin
                            and self.connector is not None
                            and getattr(self.connector,
                                        "_phase1_save_on_preempt", False)
                            and preempted_req.num_computed_tokens > 0):
                        try:
                            (_pbids, ) = self.kv_cache_manager.get_block_ids(
                                preempted_req.request_id)
                            _ptoks = (
                                list(preempted_req.prompt_token_ids)
                                + list(getattr(preempted_req,
                                               "output_token_ids", []) or []))
                            self.connector.save_preempt(
                                preempted_req, _pbids, _ptoks)
                        except Exception as _e:
                            logger.warning(
                                "Phase1 save_preempt hook failed req=%s: %s",
                                preempted_req.request_id, _e)

                    with self._kv_free_lock:
                        self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    if is_unpin:
                        pass
                    else:
                        preempted_req.status = RequestStatus.PREEMPTED
                        preempted_req.num_computed_tokens = 0
                        # Drop this victim's admit timestamp so that when
                        # it is later re-admitted the grace window starts
                        # fresh from the new admission moment.
                        self.licht_running_admit_ts.pop(
                            preempted_req.request_id, None)
                        # LICHTV2: drop the timeline anchor so the
                        # next admission re-snapshots num_computed and
                        # evictable-prefix at that moment.
                        self.licht_v2_num_computed_at_admit.pop(
                            preempted_req.request_id, None)
                        self.licht_v2_evictable_prefix_at_admit.pop(
                            preempted_req.request_id, None)
                        # 计划钉死: 抢占后清 cap 快照, 下次重新准入时按新
                        # 时刻的 S* 重新钉 (它此刻是全新候选).
                        self.licht_v2_dyn_cap_at_admit.pop(
                            preempted_req.request_id, None)
                        # 抢占后清 last_chunk: 重新准入时按"新请求"处理.
                        self._dyn_last_chunk.pop(
                            preempted_req.request_id, None)
                        # Accumulate victim count for the LICHT preempt
                        # selector.  Unconditional: non-LICHT paths never
                        # read this field, so the write is harmless.
                        preempted_req.preempt_count += 1
                        if self.log_stats:
                            preempted_req.record_event(
                                EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                        self.waiting.prepend_request(preempted_req)
                        # Plan B: _reset_licht_waiting_state now always
                        # sets wait_start back to request.arrival_time, so
                        # this call is idempotent (T_wait continues to
                        # accumulate from arrival and is never zeroed).
                        self._reset_licht_waiting_state(preempted_req)
                        preempted_reqs.append(preempted_req)
                        if preempted_req == request:
                            # No more request to preempt.
                            can_schedule = False
                            break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            if os.environ.get("LICHT_DYN_LOG_CHUNKS") == "1":
                logger.info("DYNCHUNK rid=%s prompt=%d computed=%d chunk=%d",
                            request.request_id, request.num_prompt_tokens,
                            request.num_computed_tokens, num_new_tokens)
            _pb = os.environ.get("LICHT_DYN_PROBE")
            if _pb and request.num_prompt_tokens >= int(_pb):
                logger.info("PROBE RUN t=%.3f rid=%s prompt=%d computed=%d "
                            "chunk=%d nrun=%d nwait=%d",
                            time.time(), request.request_id,
                            request.num_prompt_tokens, request.num_computed_tokens,
                            num_new_tokens, len(self.running), len(self.waiting))
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)
        # P5b: fresh per-step stash of probed prefix hits (repopulated by
        # the can_admit probes below; read by _v3_publish_step_event).
        self._v3_waiting_hit.clear()

        # Next, schedule the WAITING requests.
        # TODO (Hanchen) need to add scheduling logic for returns from functions. It should not be FCFS
        if not preempted_reqs:
            self._ensure_licht_waiting_start_timestamps()
            # Phase 2: publish (usage, total_blocks) so the connector's
            # path-selector can project post-admit occupancy without an
            # extra cross-module dep.  Sampled once per pass — the
            # projection in get_num_new_matched_tokens does per-req
            # incremental math on top.  No-op if the connector or the
            # gate env is off.
            if self.connector is not None:
                try:
                    self.connector.set_admission_kv_usage(
                        self.kv_cache_manager.usage,
                        self.kv_cache_manager.block_pool.num_gpu_blocks)
                except AttributeError:
                    # Older connector implementations lack the method;
                    # silently skip (gate stays at 0.0 = disabled).
                    pass
            # LICHTV2: timeline was already built before the running
            # loop.  Each successful admit below applies its events to
            # the same timeline so subsequent candidates see the impact
            # of earlier ones in the same scheduler step.
            if os.environ.get("LICHT_LOG_KV") == "1" and self.waiting:
                logger.info("KVLOG usage=%.3f running=%d waiting=%d",
                            self.kv_cache_manager.usage,
                            len(self.running), len(self.waiting))
            _adm_pr = os.environ.get("LICHT_ADMIT_PROBE")
            _adm_pr_wait0 = bool(_adm_pr) and any(
                _adm_pr in (getattr(r, "job_id", "") or "") for r in self.waiting)
            _adm_peeked = False
            # LICHT_SCHED_HIT_PRED: replace the hard FCFS_BREAK with a
            # "close the long lane" flag (per-step).  When a GENUINE long (by
            # REAL remaining) can't be admitted we close the long lane so
            # younger longs can't jump it, but the loop CONTINUES so shorts /
            # returning-rounds keep backfilling (no spurious break on a
            # big-prompt-but-real-short request).
            _hit_pred_on = os.environ.get("LICHT_SCHED_HIT_PRED") == "1"
            _lc_thr = int(os.environ.get("LICHT_LONG_C", "5120"))
            _long_lane_closed = False
            # LICHT_LONG_THETA_RELAX: θ 容量帽是给【还在排队的短请求】留的预留。
            # 本步一旦有短请求被挡下(放不进), 说明短请求还需要空间 → θ 保持。
            # 若本步所有短请求都进去了(没有短请求被 skip), θ 就没有保护对象 →
            # can_admit 里对长请求松开 θ 帽, 只保留物理 future-free 检查, 把空着
            # 的 KV 塞长请求, 直到某个长请求物理装不下。每步重置。
            self._licht_short_skipped = False
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    if _adm_pr_wait0:
                        logger.info("PROBE-EXIT reason=max_running rid_present "
                                    "peeked=%s nrun=%d nwait=%d kv=%.3f",
                                    _adm_peeked, len(self.running),
                                    len(self.waiting), self.kv_cache_manager.usage)
                    break

                request = self._peek_waiting_request()
                if _adm_pr and _adm_pr in (getattr(request, "job_id", "") or ""):
                    _adm_peeked = True
                    logger.info("PROBE-PEEK t=%.3f rid=%s status=%s C=%d nrun=%d "
                                "nwait=%d kv=%.3f tokbudget=%d",
                                time.time(), request.request_id, request.status,
                                request.num_tokens - request.num_computed_tokens,
                                len(self.running), len(self.waiting),
                                self.kv_cache_manager.usage, token_budget)

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id)
                        self._pop_waiting_request(request)
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self._pop_waiting_request(request)
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (self.lora_config and request.lora_request and
                    (len(scheduled_loras) == self.lora_config.max_loras and
                     request.lora_request.lora_int_id not in scheduled_loras)):
                    # Scheduling would exceed max_loras, skip.
                    self._pop_waiting_request(request)
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)
                    # ★ lookup 时的上游计数原地中和; admit 成功点再计
                    # (见 _neutralize_lookup_prefix_stats docstring)。
                    self._neutralize_lookup_prefix_stats(
                        request, num_new_local_computed_tokens)
                    if self._hitprobe_on:
                        self._hp["gcb"] += 1
                        self._hp["q_tok"] += request.num_tokens
                        self._hp["h_local"] += num_new_local_computed_tokens

                    # ★ 改动2A: 80% 投影门 —— 对【所有】waiting 请求(含已 sink)。
                    # 加入它后 usage 会超 80% → break(★改动1:停止往下扫,没扫到的
                    # 在循环外通知 prefill 下沉 arena)。usage 每次读实时值,累计本步
                    # 前面已 admit 的影响;need_blk = 这个请求要新占的块(prompt −
                    # 本地缓存命中)。门控关时(_phase2_admission_gate 假)整段跳过。
                    if (self.connector is not None and getattr(
                            self.connector, "_phase2_admission_gate", False)):
                        _total = (self.kv_cache_manager.block_pool
                                  .num_gpu_blocks)
                        if _total > 0:
                            _need_blk = (
                                max(0, len(request.prompt_token_ids)
                                    - num_new_local_computed_tokens)
                                + self.block_size - 1) // self.block_size
                            _proj = (self.kv_cache_manager.usage
                                     + _need_blk / float(_total))
                            _thr = getattr(self.connector,
                                           "_phase2_gate_threshold", 0.80)
                            if _proj > _thr:
                                # (lookup 计数已在 gcb 后统一中和, 无需撤销)
                                break   # ★改动1: 停扫; 没扫到的循环外 sink

                    # NOTE (Hanchen) The logic here is that we will see if the connector can get the tokens.
                    # If it can, we will use them.

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))

                        if self._hitprobe_on:
                            if num_external_computed_tokens is None:
                                self._hp["ext_none"] += 1
                            elif num_external_computed_tokens > 0:
                                self._hp["ext_pos"] += 1
                                self._hp["h_ext"] += num_external_computed_tokens
                            else:
                                self._hp["ext_zero"] += 1

                        # NOTE (Hanchen) this will not be called in cpu offloading.
                        if num_external_computed_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            # (lookup 计数已在 gcb 后统一中和, 无需撤销)
                            self._pop_waiting_request(request)
                            skipped_waiting_requests.prepend_request(request)
                            continue
                        # LICHT round-kv external (cross-round) hits 不再在这里
                        # 补记 —— admit 点计数的 num_computed_tokens 已含
                        # local + external (见 _count_prefix_stats_at_admit)。

                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                    # vLLM v1 corner case: GPU prefix cache + external
                    # connector together claim the ENTIRE prompt is
                    # already computed (num_computed_tokens ==
                    # num_tokens).  Leaves 0 tokens to forward and
                    # trips `assert num_new_tokens > 0` below, killing
                    # EngineCore.  Mirror the protection at
                    # _update_waiting_for_remote_kv L2955-2956: reserve
                    # at least 1 token for the forward.  Only applies
                    # to the sync path; load_kv_async sets num_new=0
                    # below regardless and the async path's
                    # _update_waiting_for_remote_kv has its own guard.
                    if (not load_kv_async
                            and num_computed_tokens >= request.num_tokens
                            and request.num_tokens > 0):
                        num_computed_tokens = request.num_tokens - 1
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                # 回传 REMOVED (2026-05-21, user request): no defer / no
                # push-back trigger.  Returning requests admit normally (plain
                # LICHT-V2 timeline behaviour); the prefix is recomputed if not
                # cached, exactly like before回传 existed.

                # Long lane closed: skip any further GENUINE long (by REAL
                # remaining = num_tokens - real hit) so it can't jump the stuck
                # oldest long.  Shorts / returning-rounds (real rem <= thr)
                # fall through and keep backfilling.
                if (_hit_pred_on and _long_lane_closed
                        and (request.num_tokens
                             - num_computed_tokens) > _lc_thr):
                    self._pop_waiting_request(request)
                    skipped_waiting_requests.prepend_request(request)
                    continue

                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    _dyn_cap = self._licht_dyn_cap(request, num_new_tokens)
                    if (0 < _dyn_cap < num_new_tokens):
                        num_new_tokens = _dyn_cap

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.chunked_prefill_enabled and \
                        num_new_tokens > token_budget:
                        self._pop_waiting_request(request)
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (encoder_inputs_to_schedule, num_new_tokens,
                         new_encoder_compute_budget
                         ) = self._try_schedule_encoder_inputs(
                             request, num_computed_tokens, num_new_tokens,
                             encoder_compute_budget)
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (0 if request.num_computed_tokens
                                              == 0 else
                                              self.num_lookahead_tokens)

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    assert ("whisper"
                            in self.vllm_config.model_config.model.lower()), (
                                "Whisper is the only supported "
                                "encoder-decoder model.")
                    num_encoder_tokens = MULTIMODAL_REGISTRY.\
                        get_encdec_max_encoder_len(
                        self.vllm_config.model_config)
                else:
                    num_encoder_tokens = 0

                # LICHTV2: capture the evictable-prefix count BEFORE
                # allocate_slots touches those blocks (after touch,
                # ref_cnt > 0 and they'd no longer be detected as
                # evictable).  We reuse this count both for can_admit's
                # timeline check below and for apply_to_timeline /
                # snapshot at admit success further down.
                evictable_prefix_lv2 = 0
                anchor_lv2 = num_computed_tokens
                free_blocks_before_admit_lv2 = -1
                if self.licht_v2_prefill_sched_enabled:
                    evictable_prefix_lv2 = (
                        self._licht_v2_count_evictable_prefix(
                            new_computed_blocks))
                    # LICHT round-kv reuse: the externally-loaded prior-round
                    # prefix occupies FRESH blocks (not free-queue prefix
                    # cache), so _count_evictable_prefix does not see them.
                    # Charge them like an evictable prefix: consumed at admit
                    # (t=0), held through this request's short prefill, and
                    # released together at t=R (BLOCK_MIGRATE).  anchor_lv2
                    # already includes these tokens (num_computed_tokens =
                    # local + external), so R/B cover only the new tokens;
                    # this term is the only missing block accounting.
                    if num_external_computed_tokens > 0:
                        evictable_prefix_lv2 += (
                            num_external_computed_tokens // self.block_size)
                    # 回传 REMOVED: P11d full-prefix anchor override is gone.
                    # anchor = actual num_computed (plain LICHT-V2): the
                    # timeline accounts only what's really cached (plus any
                    # round-kv reuse charged above).
                    # Snapshot the free-pool BEFORE can_admit so the
                    # offline simulator gets the same `current_free`
                    # the scheduler saw at THIS probe (success or fail).
                    free_blocks_before_admit_lv2 = (
                        self.kv_cache_manager.block_pool.get_num_free_blocks())

                # LICHTV2: consult the backfill window before attempting
                # any block allocation.  If the timeline rejects this
                # candidate, pop it from waiting and stash to skipped so
                # the loop continues with the next score-ranked candidate
                # instead of triggering preempt.  The evictable-prefix
                # count mirrors what allocate_slots's
                # get_num_blocks_to_allocate() will charge for prefix
                # touches; passing it keeps can_admit's prediction in
                # lockstep with the actual block accounting.
                if (self.licht_v2_prefill_sched_enabled
                        and not load_kv_async):
                    can_admit_lv2 = self._licht_v2_can_admit(
                        request, anchor_lv2, evictable_prefix_lv2)
                    # LICHTV2 monitoring: probe-level log fired on EVERY
                    # can_admit call (success or fail).  This is the
                    # critical fix over the admit-only event log: target
                    # requests are probed many times before they finally
                    # admit, and the simulator needs evictable_prefix
                    # truth at each of those earlier probe steps too —
                    # not just at the final-admit step.
                    monitoring_recorder.record_licht_admit_probe(
                        request_id=request.request_id,
                        job_id=request.job_id,
                        agent_round=request.agent_round,
                        evictable_prefix=evictable_prefix_lv2,
                        free_blocks_before_admit=
                            free_blocks_before_admit_lv2,
                        num_computed_at_probe=anchor_lv2,
                        num_new_tokens=num_new_tokens,
                        num_running_before=len(self.running),
                        will_admit=can_admit_lv2,
                    )
                    # P5b: stash this probe's real prefix hit so the
                    # StepEvent can report it for requests that stay
                    # WAITING (their num_computed_tokens is 0 pre-admit).
                    self._v3_waiting_hit[request.request_id] = (
                        anchor_lv2, evictable_prefix_lv2)
                    if not can_admit_lv2:
                        if _adm_pr and _adm_pr in (getattr(request, "job_id", "") or ""):
                            logger.info("PROBE-CANADMIT-FALSE rid=%s C=%d nrun=%d "
                                        "kv=%.3f", request.request_id,
                                        request.num_tokens - request.num_computed_tokens,
                                        len(self.running), self.kv_cache_manager.usage)
                        # θ-relax: 一个【短请求】没能进 → 短请求还需要空间 →
                        # 本步不松 θ 帽 (留着给短请求, 别被长请求占了)。
                        if (request.num_tokens - anchor_lv2) <= _lc_thr:
                            self._licht_short_skipped = True
                        self._pop_waiting_request(request)
                        skipped_waiting_requests.prepend_request(request)
                        # LICHT_SCHED_HIT_PRED: replace the hard break with a
                        # "close the long lane" flag, classified by the REAL
                        # remaining (num_computed_tokens local = real hit).  A
                        # genuine long that can't be admitted closes the lane
                        # (younger longs skipped above) but the loop CONTINUES
                        # so shorts/returning-rounds keep backfilling.  A
                        # big-prompt-but-real-short request that fails admit no
                        # longer spuriously breaks the loop.
                        if _hit_pred_on:
                            if (os.environ.get("LICHT_SCHED_SCHEME")
                                    in ("longcap_sjf", "longcap_fcfs")
                                    and (request.num_tokens
                                         - num_computed_tokens) > _lc_thr):
                                _long_lane_closed = True
                                if os.environ.get("LICHT_LOG_KV") == "1":
                                    logger.info(
                                        "EXITLOG reason=longcap_lane_closed "
                                        "kv=%.3f nrun=%d nwait=%d C=%d",
                                        self.kv_cache_manager.usage,
                                        len(self.running), len(self.waiting),
                                        request.num_tokens - num_computed_tokens)
                            continue
                        # change 3 (LICHT_LONGCAP_FCFS_BREAK=1, legacy / hit-pred
                        # off): a LONG that can't be admitted STOPS the waiting
                        # loop — younger longs must not jump ahead of it (strict
                        # FCFS among longs).  Shorts were already peeked first
                        # (higher score) so they keep backfilling; only the long
                        # lane is gated.
                        if (os.environ.get("LICHT_LONGCAP_FCFS_BREAK") == "1"
                                and os.environ.get("LICHT_SCHED_SCHEME")
                                in ("longcap_sjf", "longcap_fcfs")):
                            _thr_b = int(os.environ.get("LICHT_LONG_C", "5120"))
                            if (request.num_tokens
                                    - request.num_computed_tokens) > _thr_b:
                                if os.environ.get("LICHT_LOG_KV") == "1":
                                    logger.info(
                                        "EXITLOG reason=longcap_break kv=%.3f "
                                        "nrun=%d nwait=%d C=%d",
                                        self.kv_cache_manager.usage,
                                        len(self.running), len(self.waiting),
                                        request.num_tokens
                                        - request.num_computed_tokens)
                                break
                        continue

                # NOTE (Hanchen) This is allocating new slots. We have already decided to schedule this request
                with self._kv_free_lock:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens + num_external_computed_tokens,
                        num_new_local_computed_tokens,
                        new_computed_blocks,
                        num_lookahead_tokens=effective_lookahead_tokens,
                        delay_cache_blocks=load_kv_async,
                        num_encoder_tokens=num_encoder_tokens,
                    )

                if new_blocks is None:
                    if os.environ.get("LICHT_LOG_KV") == "1":
                        logger.info(
                            "EXITLOG reason=alloc_none kv=%.3f nrun=%d nwait=%d "
                            "C=%d numnew=%d delayfree=%d",
                            self.kv_cache_manager.usage, len(self.running),
                            len(self.waiting),
                            request.num_tokens - request.num_computed_tokens,
                            num_new_tokens, self._num_delay_free_blocks)
                    if _adm_pr and _adm_pr in (getattr(request, "job_id", "") or ""):
                        logger.info("PROBE-ALLOC-NONE rid=%s C=%d nrun=%d kv=%.3f "
                                    "(passed can_admit but allocate_slots failed)",
                                    request.request_id,
                                    request.num_tokens - request.num_computed_tokens,
                                    len(self.running), self.kv_cache_manager.usage)
                    #print(f"Request {request.request_id} cannot be scheduled due to no slots")
                    # The request cannot be scheduled.
                    # Delay-free admission control: if delay-free blocks
                    # will be freed soon, wait instead of evicting pinned.
                    if self._num_delay_free_blocks > 0:
                        break
                    # TODO (Hanchen) need to add preemption logic here for CONTINUUM
                    if len(self.running) == 0 and self.pinned_requests:
                        if self.policy == SchedulingPolicy.CONTINUUM:
                            preempted_req, _ = self.pop_running_request_based_on_last_step(request)
                            if preempted_req in scheduled_running_reqs:
                                scheduled_running_reqs.remove(preempted_req)
                            with self._kv_free_lock:
                                self.kv_cache_manager.free(preempted_req)
                            self.encoder_cache_manager.free(preempted_req)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                self._pop_waiting_request(request)
                self._drop_licht_waiting_state(request.request_id)

                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                req_index += 1
                self.running.append(request)
                # ★ prefix-hit 指标 admit 点计数 (lookup 处已中和): 每次准入恰好
                # 一次, hits = 本次 admit 真实免算的 tokens (本地 + 外部)。
                self._count_prefix_stats_at_admit(request, num_computed_tokens)
                if self.licht_prefill_sched_enabled:
                    # Stamp the admission time so the LICHT preempt
                    # selector can grant a min-run grace window before
                    # this request becomes evictable again.
                    self.licht_running_admit_ts[request.request_id] = (
                        time.time())
                if self.licht_v2_prefill_sched_enabled:
                    # Snapshot the prefix-aware token anchor AND the
                    # evictable-prefix count captured before
                    # allocate_slots touched those blocks.  Both are
                    # reused later: anchor by R/B math, evictable by
                    # build_timeline's release accounting.  Then commit
                    # this admit's events into the live timeline so
                    # subsequent candidates see its impact.
                    self.licht_v2_num_computed_at_admit[
                        request.request_id] = anchor_lv2
                    self.licht_v2_evictable_prefix_at_admit[
                        request.request_id] = evictable_prefix_lv2
                    # ★ 计划钉死: 快照 chunk cap.  用 live 版 (此刻 request_id
                    # 还没进 num_computed_at_admit, 但下一行就进了, 所以直接调
                    # live 避免自引用) 按【全额剩余】算, 与 timeline 的
                    # _licht_v2_chunk_for(remaining=num_tokens-anchor) 同口径.
                    if self._dyn_pin_cap:
                        self.licht_v2_dyn_cap_at_admit[request.request_id] = (
                            self._licht_dyn_cap_live(
                                request,
                                max(request.num_tokens - anchor_lv2, 1)))
                    self._licht_v2_apply_to_timeline(
                        request, anchor_lv2, evictable_prefix_lv2)
                    # LICHTV2 monitoring: dump the four block-accounting
                    # quantities the offline simulator needs to verify
                    # its per-step ledger against ground truth.
                    new_blocks_count_lv2 = sum(
                        len(g) for g in new_blocks.blocks)
                    monitoring_recorder.record_licht_admit_event(
                        request_id=request.request_id,
                        job_id=request.job_id,
                        agent_round=request.agent_round,
                        evictable_prefix=evictable_prefix_lv2,
                        new_blocks_allocated=new_blocks_count_lv2,
                        free_blocks_before_admit=
                            free_blocks_before_admit_lv2,
                        num_computed_at_admit=anchor_lv2,
                        num_new_tokens=num_new_tokens,
                        num_running_before=len(self.running) - 1,
                    )
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if self._hitprobe_on and request.status == RequestStatus.WAITING:
                    self._hp["admit"] += 1
                if request.status == RequestStatus.WAITING:
                    self.continuum_recorder.request_waiting_to_running(
                        request,
                        prompt_length=request.num_prompt_tokens,
                        hit_length=num_computed_tokens
                    )
                    scheduled_new_reqs.append(request)
                    self._trace_ev("admit", rid=request.request_id,
                                   t=time.time(), step=self._trace_step,
                                   ntok=request.num_tokens,
                                   ncomp=num_computed_tokens)
                    # LICHTV3 K_queue ground truth: a WAITING request was
                    # just admitted to RUNNING — write the diff between
                    # admit step and arrival step.
                    if self._v3_kqueue_log_enabled:
                        # NOTE: use get() not pop() — StepEvent emission
                        # at end of schedule() also needs arrival_step
                        # for admitted requests.  Stale dict entries are
                        # overwritten by add_request on re-arrival; no
                        # cleanup needed for finished reqs in this run.
                        arr_step = self._v3_arrival_step.get(
                            request.request_id, None)
                        arr_ts = self._v3_arrival_ts.get(
                            request.request_id, None)
                        if arr_step is not None:
                            try:
                                rec = {
                                    "ts": time.time(),
                                    "request_id": request.request_id,
                                    "job_id": getattr(request,
                                                       "job_id", None),
                                    "agent_round": getattr(request,
                                                            "agent_round",
                                                            None),
                                    "arrival_step": arr_step,
                                    "admit_step": self._v3_sched_step,
                                    "k_queue_actual":
                                        self._v3_sched_step - arr_step,
                                    "arrival_ts": arr_ts,
                                    "wait_wall_s": (
                                        time.time() - arr_ts
                                        if arr_ts is not None else None),
                                    "num_prompt_tokens":
                                        request.num_prompt_tokens,
                                    "num_running_at_admit":
                                        len(self.running),
                                    "num_waiting_at_admit":
                                        len(self.waiting),
                                }
                                with open(self._v3_kqueue_log_path,
                                          "a") as _kf:
                                    _kf.write(json.dumps(rec) + "\n")
                            except Exception:
                                pass
                elif request.status == RequestStatus.PREEMPTED:
                    self.continuum_recorder.request_evicted_to_running(
                        request,
                        prompt_length=request.num_prompt_tokens,
                        hit_length=num_computed_tokens
                    )
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                #NOTE (Hanchen) we do not need to care about lora.
                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id))
                num_scheduled_tokens[request.request_id] = num_new_tokens
                if os.environ.get("LICHT_DYN_LOG_CHUNKS") == "1":
                    logger.info("DYNCHUNK rid=%s prompt=%d computed=%d chunk=%d",
                                request.request_id, request.num_prompt_tokens,
                                request.num_computed_tokens, num_new_tokens)
                _pb = os.environ.get("LICHT_DYN_PROBE")
                if _pb and request.num_prompt_tokens >= int(_pb):
                    logger.info("PROBE ADMIT t=%.3f rid=%s prompt=%d arr=%.3f "
                                "queue_wait=%.2f computed=%d chunk=%d nrun=%d nwait=%d",
                                time.time(), request.request_id,
                                request.num_prompt_tokens, request.arrival_time,
                                time.time() - request.arrival_time,
                                request.num_computed_tokens, num_new_tokens,
                                len(self.running), len(self.waiting))
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # DIAGNOSTIC (LICHT_LIFECYCLE_LOG=1): one line per admit showing
                # what SCORING saw (C_score = prompt, because num_computed=0 for
                # waiting reqs) vs the REAL remaining (C_real = prompt - real hit).
                # A returning round with a big prefix shows scored=LONG / real=short
                # -> it was mis-classified into the long lane and (if waited is big)
                # starved there by FCFS_BREAK despite being cheap to run.
                if (os.environ.get("LICHT_LIFECYCLE_LOG") == "1"
                        and self.licht_prefill_sched_enabled):
                    _thrL = int(os.environ.get("LICHT_LONG_C", "5120"))
                    _c_score = request.num_tokens
                    _c_real = request.num_tokens - num_computed_tokens
                    _now = time.time()
                    # total = engine-entry -> admit (includes input-queue delay);
                    # sched = add-to-waiting -> admit (pure scheduler wait, the
                    # part the long-lane mis-scoring / FCFS_BREAK can inflate).
                    _q_total = _now - request.arrival_time
                    _q_sched = _now - getattr(request, "_licht_add_ts",
                                              request.arrival_time)
                    logger.info(
                        "LIFECYCLE rid=%s job=%s prompt=%d hit=%d C_score=%d "
                        "C_real=%d scored=%s real=%s waited=%.2fs sched_wait=%.2fs",
                        request.request_id,
                        (getattr(request, "job_id", "") or "")[:40],
                        request.num_tokens, num_computed_tokens, _c_score,
                        _c_real, "LONG" if _c_score > _thrL else "short",
                        "LONG" if _c_real > _thrL else "short",
                        _q_total, _q_sched)
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

            # ★ 改动1/2B: 上面 80% break(或 running满/分配失败 break)后, self.waiting
            # 里剩的是【没扫到】的请求(break 那个 + 它后面的)。把其中【还没 sink 的】
            # 通知 prefill 下沉 arena(放掉 prefill GPU, 别一直占着)。已 sink 的留着等
            # 下轮 room。skipped(本步已 defer/跳过的)在另一队列, 不在此 self.waiting。
            # 在 `if not preempted_reqs` 块内 → 只非 preempt 步执行; 在 skipped 恢复前
            # 迭代 → 只动没扫到的。
            # 条件 `self.waiting and token_budget>0`: 只在循环【因 break 停下】(还有
            # waiting 且 budget 没耗尽)时 sink —— 区分"正常扫完/budget耗尽"(不 sink)。
            if (self.waiting and token_budget > 0
                    and self.connector is not None
                    and hasattr(self.connector, "mark_arena_sink")):
                for _wreq in self.waiting:
                    # ★ 跳过 KV 已就绪的请求(num_computed_tokens>0 = post-NCCL,
                    #   KV 已搬进 decode 显存、bridge 已被消费)。再标 sink 只会
                    #   发个扑空的 ARENA_SINK RPC(declined 噪声)且毫无意义 ——
                    #   它马上就能从本地 admit, 不需要 prefill 下沉。
                    if _wreq.num_computed_tokens > 0:
                        continue
                    try:
                        self.connector.mark_arena_sink(_wreq)
                    except Exception:  # pragma: no cover
                        pass

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        # beta_r/b probe: store THIS step's compute features (context D_i is
        # pre-step here -> the attention the forward will do). Paired with dt
        # at the top of the NEXT schedule call.
        if os.environ.get("LICHT_BRB_PROBE"):
            _sc = _sctx = _scd = 0
            for _r in self.running:
                _c = num_scheduled_tokens.get(_r.request_id, 0)
                if _c > 0:
                    _d = _r.num_computed_tokens
                    _sc += _c
                    _sctx += _d
                    _scd += _c * _d
            self._brb_prev_feat = {"sum_c": int(_sc), "sum_ctx": int(_sctx),
                                   "sum_c_ctx": int(_scd),
                                   "n_run": len(self.running),
                                   "n_sched": len(num_scheduled_tokens)}
        # Per-step per-request chunk ground-truth log (env LICHT_LOG_CHUNK=1).
        # Each line = one scheduler step: the batch S* plus every scheduled
        # request's pre-step num_computed (D) and this step's chunk.  Lets us
        # reconstruct each request's exact D trajectory without any guessing.
        if os.environ.get("LICHT_LOG_CHUNK") == "1" and num_scheduled_tokens:
            try:
                self._chunk_log_step = getattr(self, "_chunk_log_step", 0) + 1
                _reqs = []
                for _r in self.running:
                    _ck = num_scheduled_tokens.get(_r.request_id, 0)
                    if _ck > 0:
                        _reqs.append({
                            "rid": _r.request_id,
                            "job": getattr(_r, "job_id", None),
                            "computed": int(_r.num_computed_tokens),
                            "chunk": int(_ck),
                            "prompt": int(_r.num_prompt_tokens),
                        })
                # waiting 也记: 公式用 running+waiting, waiting 的 num_computed
                # 此刻=0(代码口径), 记 prompt+computed 以便完整复现 W_soft。
                _wait = [{"rid": _w.request_id,
                          "prompt": int(_w.num_prompt_tokens),
                          "computed": int(_w.num_computed_tokens)}
                         for _w in list(self.waiting)]
                _p = os.environ.get("LICHT_CHUNK_LOG_PATH",
                                    "/tmp/licht_chunk_log.jsonl")
                with open(_p, "a") as _f:
                    _f.write(json.dumps({
                        "step": self._chunk_log_step,
                        "S_star": int(getattr(self, "_dyn_S", 0) or 0),
                        "n_run": len(self.running),
                        "n_wait": len(self.waiting),
                        "running": _reqs,
                        "waiting": _wait}) + "\n")
            except Exception:
                pass
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        if self._trace_f is not None:
            try:
                _free = self.kv_cache_manager.block_pool.get_num_free_blocks()
                _tot = self._total_kv_blocks
                self._trace_ev("step", step=self._trace_step, t=time.time(),
                               kv_used=_tot - _free, kv_total=_tot,
                               chunks=dict(num_scheduled_tokens))
            except Exception:
                pass

        structured_output_request_ids, grammar_bitmask = (
            self.get_grammar_bitmask(self.running,
                                     scheduled_spec_decode_tokens))

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.
            get_freed_mm_hashes(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )
        #print(f"scheduler_output: {scheduler_output}")

        # LICHT: per-step occupancy for the metrics log.  In PD-disagg the
        # prefill request leaves `running` and frees its blocks WITHIN the
        # step, so num_running_reqs / kv usage sampled at log time read ~0.
        # Capture here, BEFORE _update_after_schedule, what was actually
        # scheduled this step + the block footprint of those requests (full
        # prefix incl. round-kv loaded blocks + this step's new prompt).
        try:
            # True allocated-block usage RIGHT NOW (after this step's
            # allocations, before the prefill migrates + frees).  Use
            # block_pool.get_usage() which dedups shared prefix-cache blocks
            # via refcount -> bounded to ~100%.  A per-request sum would
            # double-count shared prefixes and exceed 100% (the earlier bug).
            # Captured here (not at make_stats time) because prefill frees its
            # blocks within the step, so usage sampled later reads ~0.
            self._step_sched_reqs = len(num_scheduled_tokens)
            self._step_block_usage = (
                self.kv_cache_manager.block_pool.get_usage())
        except Exception:
            self._step_sched_reqs = 0
            self._step_block_usage = 0.0

        # NOTE (Hanchen) this will handle the KVConnector
        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)

        # Step-time ground truth: log per-step compute load so the
        # offline step_time/ training pipeline has the actual
        # num_scheduled_tokens (= chunked-prefill + decode tokens THIS
        # step) — not the iteration_stats.num_prompt_tokens which counts
        # finished-prefill prompt lengths (≠ per-step compute).
        try:
            new_admit_tokens = sum(
                num_scheduled_tokens.get(r.request_id, 0)
                for r in scheduled_new_reqs)
            monitoring_recorder.record_step_compute(
                step_id=self._v3_sched_step,
                num_scheduled_tokens=int(
                    sum(num_scheduled_tokens.values())
                    if num_scheduled_tokens else 0),
                num_running=len(self.running),
                num_waiting=len(self.waiting),
                num_new_admits=len(scheduled_new_reqs),
                num_new_admit_tokens=int(new_admit_tokens),
            )
        except Exception as e:
            logger.debug("record_step_compute failed: %s", e)

        # LICHTV3 StepEvent publish: send authoritative snapshot of
        # waiting + running so decode-side ShadowScheduler can mirror
        # state in real time.  Best-effort; failure never blocks
        # the scheduler.
        if self._v3_step_event_pub is not None:
            try:
                num_tokens_total = (
                    sum(num_scheduled_tokens.values())
                    if num_scheduled_tokens else 0)
                self._v3_publish_step_event(
                    scheduled_new_reqs,
                    list(self.finished_req_ids),
                    num_scheduled_tokens_this_step=num_tokens_total)
            except Exception as e:
                logger.debug("LICHTV3 StepEvent publish failed: %s", e)

        if self._hitprobe_on:
            self._hp_step += 1
            if self._hp_step % 500 == 0:
                h = self._hp
                _q = max(h["q_tok"], 1)
                logger.info(
                    "HITPROBE gcb=%d admit=%d (gcb/admit=%.2f) | ext: none=%d "
                    "zero=%d pos=%d | uncount=%d | tok: q=%d hLocal=%d hExt=%d "
                    "-> 重建命中=%.1f%% (对照 vLLM 指标; ext_zero 高=arena瞬时miss, "
                    "gcb>>admit=重复计数)",
                    h["gcb"], h["admit"], h["gcb"] / max(h["admit"], 1),
                    h["ext_none"], h["ext_zero"], h["ext_pos"], h["uncount"],
                    h["q_tok"], h["h_local"], h["h_ext"],
                    100.0 * (h["h_local"] + h["h_ext"]) / _q)

        return scheduler_output

    # ----------------------------------------------------------------------
    # LICHTV3 StepEvent helpers
    # ----------------------------------------------------------------------

    def _v3_publish_step_event(self,
                                scheduled_new_reqs: list,
                                finished_now: list,
                                num_scheduled_tokens_this_step: int = 0
                                ) -> None:
        """Build + publish a StepEvent for this scheduler step."""
        from vllm.v1.core.sched.licht_v3.step_event import (
            ReqSnapshot, StepEvent, encode_step_event)
        now = time.time()
        # Rolling sec/step (window of 32 step ends).
        self._v3_step_wall_history.append(now)
        if len(self._v3_step_wall_history) > 32:
            self._v3_step_wall_history.pop(0)
        if len(self._v3_step_wall_history) >= 2:
            span = (self._v3_step_wall_history[-1]
                    - self._v3_step_wall_history[0])
            n = len(self._v3_step_wall_history) - 1
            sec_per_step = span / n if n > 0 else 0.05
        else:
            sec_per_step = 0.05

        def _snap(req, *, admit_step=None, arrival_step=None,
                  r_remaining=None) -> "ReqSnapshot":
            return ReqSnapshot(
                request_id=req.request_id,
                traj_id=getattr(req, "job_id", None),
                agent_round=getattr(req, "agent_round", None),
                num_prompt_tokens=int(
                    getattr(req, "num_prompt_tokens", 0) or 0),
                hit_length=int(
                    getattr(req, "num_computed_tokens", 0) or 0),
                admit_step=admit_step,
                arrival_step=arrival_step,
                r_remaining=r_remaining,
                evictable_prefix=0,
            )

        # Compute true r_remaining (chunked-prefill chunks left).  Only
        # meaningful when LICHTV2 is enabled (chunk_size constant comes
        # from scheduler_config.long_prefill_token_threshold).  For
        # decode-phase requests _licht_v2_R_at returns 0; decode side
        # then treats them as "static (no chunks, never auto-release)".
        def _r_remaining_for(req) -> Optional[int]:
            if not self.licht_v2_enabled:
                return None
            try:
                return int(self._licht_v2_R_at(
                    req, req.num_computed_tokens))
            except Exception:
                return None

        admitted = [
            _snap(r,
                  admit_step=self._v3_sched_step,
                  arrival_step=self._v3_arrival_step.get(r.request_id),
                  r_remaining=_r_remaining_for(r))
            for r in scheduled_new_reqs
        ]
        # waiting snapshot — only requests still in queue post-schedule.
        try:
            waiting_iter = list(self.waiting)
        except Exception:
            waiting_iter = []
        waiting_now = []
        for w in waiting_iter:
            arr = self._v3_arrival_step.get(w.request_id, self._v3_sched_step)
            snap = _snap(w, arrival_step=arr)
            # P5b: a WAITING request's request.num_computed_tokens is 0
            # pre-admit, so _snap's hit_length is 0.  Override with the
            # real prefix hit captured by this step's can_admit probe.
            he = self._v3_waiting_hit.get(w.request_id)
            if he is not None:
                snap.hit_length = int(he[0])
                snap.evictable_prefix = int(he[1])
            waiting_now.append(snap)
        # running snapshot.  r_remaining is computed exactly so decode
        # side can model release timing correctly.
        running_now = []
        for r in self.running:
            running_now.append(
                _snap(r, r_remaining=_r_remaining_for(r)))
        evt = StepEvent(
            step_id=int(self._v3_sched_step),
            step_wall_ts=float(now),
            sec_per_step_recent=float(sec_per_step),
            admitted=admitted,
            finished=[str(x) for x in finished_now],
            preempted=[],
            waiting_now=waiting_now,
            running_now=running_now,
            max_num_seqs=int(self.max_num_running_reqs),
            max_num_batched_tokens=int(self.max_num_scheduled_tokens),
            block_size=int(self.block_size),
            total_kv_blocks=int(self.kv_cache_config.num_blocks
                                if hasattr(self.kv_cache_config,
                                           "num_blocks") else 0),
            num_scheduled_tokens_this_step=int(
                num_scheduled_tokens_this_step),
            # LICHTV2 timeline + constants — the simulator on decode
            # side mirrors prefill's admit decisions using these.
            future_free=(list(self._licht_v2_future_free)
                         if self.licht_v2_enabled else []),
            future_alloc=(list(self._licht_v2_future_alloc)
                          if self.licht_v2_enabled else []),
            lichtv2_horizon_n=int(getattr(self, "LICHTV2_N", 0)),
            chunk_size_tokens=int(
                (getattr(self, "_dyn_S", 0)
                 or self._licht_v2_chunk_size())
                if self.licht_v2_enabled else 0),
            max_alloc_per_step_blocks=int(
                self.max_num_scheduled_tokens // max(self.block_size, 1)),
            long_tail_headroom_blocks=int(
                0.025 * (self.kv_cache_config.num_blocks
                         if hasattr(self.kv_cache_config,
                                    "num_blocks") else 0)),
            long_running_count=int(
                self._licht_v2_count_long_running()
                if self.licht_v2_enabled else 0),
            max_long_bridge=int(
                getattr(self, "LICHTV2_MAX_LONG_BRIDGE", 2)),
            score_a=float(getattr(self, "LICHT_PREFILL_SCORE_A", 3.0)),
            score_b=float(getattr(self, "LICHT_PREFILL_SCORE_B", 1.0)),
            score_tmax_s=float(
                getattr(self, "LICHT_PREFILL_SCORE_TMAX_S", 120.0)),
            round_decay_alpha=float(
                getattr(self, "LICHT_PREFILL_ROUND_DECAY_ALPHA", 0.5)),
        )
        try:
            self._v3_step_event_pub.send(encode_step_event(evt),
                                          flags=0x1)  # NOBLOCK
        except Exception as e:
            logger.debug("StepEvent send failed: %s", e)

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[Optional[tuple[list[int], ...]]] = []
        num_computed_tokens: list[int] = []

        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = (num_scheduled_tokens[req_id] -
                          len(spec_decode_tokens.get(req_id, ())))
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[req.num_computed_tokens:req.
                                              num_computed_tokens + num_tokens]
                new_token_ids.append(token_ids)
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True))
            num_computed_tokens.append(req.num_computed_tokens)
        # Because resumed_reqs is usually empty, it is more efficient to do
        # in-place appending so that we don't need to allocate a new list.
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_tokens_to_schedule = 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used.")
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if request.mm_hashes[i] in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(
                        request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if not self.encoder_cache_manager.can_allocate(
                    request, i, encoder_compute_budget,
                    num_tokens_to_schedule):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            num_tokens_to_schedule += num_encoder_tokens
            encoder_compute_budget -= num_encoder_tokens
            mm_hashes_to_schedule.add(request.mm_hashes[i])
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
        )

    def get_grammar_bitmask(
        self,
        requests: list[Request],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ):
        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to its index in the batch.
        # This will help us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}
        for i, req in enumerate(requests):
            if req.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[req.request_id] = i

        if not structured_output_request_ids:
            bitmask = None
        else:
            bitmask = self.structured_output_manager.grammar_bitmask(
                self.requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )
        return structured_output_request_ids, bitmask

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # Drain deferred frees from bg thread — blocks are already freed,
        # this handles del requests / pin / monitoring.
        if self._bg_free_thread is not None:
            self._drain_deferred_frees()

        # Change 4: process KV transfer completions first so that blocks
        # are freed before we process new outputs (reduces scheduling lag).
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output)

        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                request.num_computed_tokens -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids)

            # Stop checking for pooler models. 
            # NOTE (Hanchen) this should never be called in our case
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len,
                                     pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # checked above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            #NOTE (Hanchen) do we need to care?
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    ))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        # Stats are only emitted when this iteration has real activity,
        # i.e. at least one request produced a token, a pooler output,
        # or finished (populated kv_transfer_params).  Empty iterations
        # (KV-stall spins, no scheduled tokens, no finished requests)
        # used to emit a stats-only EngineCoreOutputs every loop — at
        # ~1000/s this saturated the API server output_handler
        # coroutine (delaying HTTP responses by seconds) and bloated
        # monitoring_timestamps to multi-GB.  There is no observability
        # loss worth the cost: during a stall the scheduler state does
        # not change, so repeated snapshots carry zero new information.
        # The periodic LoggingStatLogger.log() will reuse the last
        # snapshot it saw for its per-second summary line.
        if self.log_stats and engine_core_outputs:
            stats = self.make_stats(spec_decoding_stats)
            if stats is not None:
                # engine_core_outputs is non-empty by the outer check, so
                # next(iter(...)) always returns a valid EngineCoreOutputs.
                eco = next(iter(engine_core_outputs.values()))
                eco.scheduler_stats = stats

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        emitted_token_ids: list[int] = []
        for output_token_id in new_token_ids:
            if request.trace_replay_enabled:
                try:
                    output_token_id = request.pop_next_trace_token()
                except IndexError:
                    request.status = RequestStatus.FINISHED_STOPPED
                    request.stop_reason = "trace_replay_end"
                    stopped = True
                    break

            request.append_output_token_ids(output_token_id)
            emitted_token_ids.append(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                break
        return emitted_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = (
            self.encoder_cache_manager.get_cached_input_ids(request))
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_positions = request.mm_positions[input_id]
            start_pos = mm_positions.offset
            num_tokens = mm_positions.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)

    def update_draft_token_ids(
        self,
        draft_token_ids: DraftTokenIds,
    ) -> None:
        for req_id, spec_token_ids in zip(
                draft_token_ids.req_ids,
                draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # Add newly generated spec token ids to the request.
            if not spec_token_ids:
                # NOTE(woosuk): request.spec_token_ids should be updated.
                request.spec_token_ids.clear()
            elif self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                    spec_token_ids)
            else:
                request.spec_token_ids = spec_token_ids

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        # Synthetic-history perf benchmark (gated by request_id prefix "exp_"):
        # mark the first L prompt tokens as already-computed history so the
        # engine allocates L+C blocks (first L never written = garbage KV) and
        # only forwards the last C tokens, attending to all L+C.  Same code
        # path as a KV-transfer-resumed request.  No effect on other requests.
        if (request.request_id.startswith("exp_")
                and request.num_computed_tokens == 0):
            try:
                _L = int(request.request_id.split("_L", 1)[1].split("_", 1)[0])
                request.num_computed_tokens = max(0, min(_L,
                                                         request.num_tokens - 1))
            except (IndexError, ValueError):
                pass
        self.tool_call_estimator.request_arrives(request)
        self.continuum_recorder.request_arrives(request)
        # DIAGNOSTIC: stamp when the SCHEDULER first sees this request (enters
        # the waiting set).  request.arrival_time is the ENGINE-entry time
        # (processor.py), which includes time spent in the engine input queue
        # before the scheduler picked it up.  This stamp isolates the pure
        # scheduler-waiting time (add -> admit) from that input-queue delay.
        request._licht_add_ts = time.time()
        self._trace_ev("arrive", rid=request.request_id,
                       t=request._licht_add_ts, ntok=request.num_tokens,
                       ncomp=request.num_computed_tokens)

        #print(f"Adding request {request.job_id} to waiting queue")
        #print(f"Request last_func_call: {request.last_func_call}")
        #print(f"Request is_last_step: {request.is_last_step}")
        #print(f"Request this_func_call: {request.this_func_call}")
        # Track the first entry time for this job_id if not already recorded
        if request.job_id not in self.running_job_id_first_entry_time:
            self.running_job_id_first_entry_time[request.job_id] = request.arrival_time
        self.waiting.add_request(request)
        self._reset_licht_waiting_state(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)
        # LICHTV3 K_queue ground truth: stamp arrival step + wall ts so
        # the diff at admit time gives the true number of scheduler steps
        # the request waited in `self.waiting`.
        if self._v3_kqueue_log_enabled:
            self._v3_arrival_step[request.request_id] = self._v3_sched_step
            self._v3_arrival_ts[request.request_id] = time.time()

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _licht_upd_acc(self, phase: str, dt: float,
                       bump: bool = False) -> None:
        """UPD-PROF (LICHT_STEP_PROFILE=1): 累积 update 收尾各子操作耗时, 每
        LICHT_UPD_PROFILE_N(默认200)个结束请求汇总一次 → 看 update 暴击是
        decode_manager / tool_estimator / recorder 哪个吃的(per-finish avg)。"""
        try:
            acc = getattr(self, "_licht_upd_buf", None)
            if acc is None:
                acc = self._licht_upd_buf = {}
                self._licht_upd_n = 0
            acc[phase] = acc.get(phase, 0.0) + dt
            if not bump:
                return
            self._licht_upd_n += 1
            if self._licht_upd_n < int(
                    os.environ.get("LICHT_UPD_PROFILE_N", "200")):
                return
            n = self._licht_upd_n
            parts = " ".join(
                f"{k}={v * 1000.0:.0f}ms(avg{v / n * 1000.0:.3f})"
                for k, v in sorted(acc.items()))
            logger.info("UPD-PROF finishes=%d | per-phase total(per-finish "
                        "avg ms): %s", n, parts)
            self._licht_upd_buf = {}
            self._licht_upd_n = 0
        except Exception:  # pragma: no cover - profiling must never break
            pass

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()
        self._trace_ev("finish", rid=request.request_id, t=time.time())
        # LICHTV3: on decode, every finished request is a "round end".
        # Notify the v3 coordinator BEFORE we free state so its predictor
        # sees consistent num_tokens / agent_round on the request.  This
        # call never raises (manager wraps the pipeline in try/except).
        _uprof = os.environ.get("LICHT_STEP_PROFILE") == "1"
        if self.licht_v3_decode_manager is not None:
            _ut = time.perf_counter() if _uprof else 0.0
            self.licht_v3_decode_manager.on_round_finished(
                request, decode_finish_ts=time.time())
            if _uprof:
                self._licht_upd_acc("decode_manager",
                                    time.perf_counter() - _ut)
        # LICHTV3 GPU-tier retention: if the v3 decision wanted to keep
        # KV in decode GPU until the prewarm push fires, stash the
        # request here and SKIP the normal free path.  The decode
        # manager calls back into _v3_release_retained() after the push
        # completes (success or failure) to perform the deferred free.
        if (self.licht_v3_decode_manager is not None
                and self.licht_v3_decode_manager.should_retain_gpu_blocks(
                    request.request_id)):
            self._v3_retained_requests[request.request_id] = request
            return None
        _ut = time.perf_counter() if _uprof else 0.0
        self.tool_call_estimator.request_finished(request)
        if _uprof:
            self._licht_upd_acc("tool_estimator", time.perf_counter() - _ut)
            _ut = time.perf_counter()
        self.continuum_recorder.request_finished(request)
        if _uprof:
            self._licht_upd_acc("recorder", time.perf_counter() - _ut,
                                bump=True)
        self._drop_licht_waiting_state(request.request_id)
        self.licht_running_admit_ts.pop(request.request_id, None)
        self.licht_v2_num_computed_at_admit.pop(request.request_id, None)
        self.licht_v2_evictable_prefix_at_admit.pop(request.request_id, None)
        self.licht_v2_dyn_cap_at_admit.pop(request.request_id, None)
        self._dyn_last_chunk.pop(request.request_id, None)

        # NOTE (Hanchen) in unpin, we need to make sure it is not delay free blocks because it could be still waiting for transfer, need to copy something similar to the kv_xfer_params

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        # Phase A: if this is the trajectory's last step (client set
        # is_last_step=True in vllm_xargs), tell the connector to
        # evict the job's arena entries — manifest deletion runs
        # immediately; worker-side in-memory bookkeeping updates next
        # step via meta.finished_jobs.
        if (getattr(request, "is_last_step", None) is True
                and self.connector is not None
                and getattr(request, "job_id", None)):
            try:
                self.connector.mark_finished_job(str(request.job_id))
            except AttributeError:
                pass  # older connectors don't have it
            except Exception as e:  # pragma: no cover
                logger.debug("mark_finished_job hook failed req=%s: %s",
                             request.request_id, e)
        # NOTE (Hanchen) we do not care about encoder here, ignore
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)
        else:
            # Track delay-free blocks for admission control.
            block_ids = self.kv_cache_manager.get_block_ids(
                request.request_id)
            n = sum(len(ids) for ids in block_ids)
            self._num_delay_free_blocks += n
            self._delay_free_req_ids.add(request.request_id)
            monitoring_recorder.record_delay_free_start(
                request.request_id,
                getattr(request, "job_id", None))

        return kv_xfer_params

    def _v3_release_retained(self, request_id: str) -> None:
        """Free the KV blocks of a request that was previously retained
        by LICHTV3 GPU tier.  Called from the decode manager's prewarm
        thread after the v3 push completes.  No-op if the request is
        not in the retained map (e.g., scheduler restarted between
        retain and release)."""
        request = self._v3_retained_requests.pop(request_id, None)
        if request is None:
            return
        try:
            # Mirror the tail of _free_request that we skipped earlier.
            self.tool_call_estimator.request_finished(request)
            self.continuum_recorder.request_finished(request)
            self._drop_licht_waiting_state(request.request_id)
            self.licht_running_admit_ts.pop(request.request_id, None)
            self.licht_v2_num_computed_at_admit.pop(request.request_id, None)
            self.licht_v2_evictable_prefix_at_admit.pop(
                request.request_id, None)
            self.licht_v2_dyn_cap_at_admit.pop(request.request_id, None)
            self._dyn_last_chunk.pop(request.request_id, None)
            delay_free_blocks, _ = self._connector_finished(request)
            self.encoder_cache_manager.free(request)
            self.finished_req_ids.add(request_id)
            if self.finished_req_ids_dict is not None:
                self.finished_req_ids_dict[request.client_index].add(
                    request_id)
            if not delay_free_blocks:
                self._free_blocks(request)
        except Exception as e:
            logger.warning("LICHTV3 release_retained free failed req=%s: %s",
                           request_id, e)

    def _dec_delay_free_counter(self, request: Request) -> None:
        """Decrement delay-free block counter before freeing blocks.

        Must be called BEFORE kv_cache_manager.free() because free()
        removes blocks from req_to_blocks.
        """
        req_id = request.request_id
        if req_id in self._delay_free_req_ids:
            block_ids = self.kv_cache_manager.get_block_ids(req_id)
            n = sum(len(ids) for ids in block_ids)
            self._num_delay_free_blocks -= n
            self._delay_free_req_ids.discard(req_id)

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        # Decrement delay-free counter first — must happen before any
        # early return (e.g. Continuum pin) and before kv_cache_manager.free().
        self._dec_delay_free_counter(request)

        #NOTE (Hanchen) this is called when the request is finished
        for req, end_time in self.pinned_requests:
            if req.job_id == request.job_id:
                self.unpin_request(req, end_time)

        # TODO (Hanchen) check if we want to pin this memory here for how long, pin them on scheduler level.
        #############
        if self.policy == SchedulingPolicy.CONTINUUM and not request.is_last_step:
            length_of_pin = self.tool_call_estimator.set_up_pin(request)

            #print(f"Setting up pin for request {request.request_id} with length {length_of_pin}")
            #Floating point error
            if length_of_pin > 0.01:
                self.pin_request(request, length_of_pin)
                del self.requests[request.request_id]
                return
        #############

        with self._kv_free_lock:
            self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        num_waiting_for_remote_kvs = sum(
            1 for req in self.waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS)
        num_preempted = sum(
            1 for req in self.waiting
            if req.status == RequestStatus.PREEMPTED)
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            step_sched_reqs=getattr(self, "_step_sched_reqs", 0),
            step_block_usage=getattr(self, "_step_block_usage", 0.0),
            num_waiting_for_remote_kvs=num_waiting_for_remote_kvs,
            num_preempted=num_preempted,
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(req.is_output_corrupted
                                   for req in self.running),
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        try:
            self.continuum_recorder.print_history()
        except Exception:
            logger.exception("Failed to dump scheduler_timestamps")

        try:
            monitoring_recorder.dump()
        except Exception:
            logger.exception("Failed to dump monitoring_timestamps")

        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    def _connector_finished(
            self, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        return self.connector.request_finished(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        # Now that the blocks are ready, actually cache them.
        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        num_computed_tokens = len(block_ids) * self.block_size
        # Handle the case where num request tokens less than one block.
        num_computed_tokens = min(num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        # This will cache the blocks iff caching is enabled.
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

        # Update the request state for scheduling.
        request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self,
                                      kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        worker_ts = kv_connector_output.delay_free_timestamps or {}
        for req_id in (kv_connector_output.finished_recving or ()):
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in (kv_connector_output.finished_sending or ()):
            # Skip if already freed by the fast-release side-channel.
            if req_id in self._fast_released_req_ids:
                self._fast_released_req_ids.discard(req_id)
                continue
            # Guard: request may have been freed by another path (bg
            # thread + _drain_deferred_frees) between get_finished()
            # returning and this code running.
            request = self.requests.get(req_id)
            if request is None:
                continue
            logger.debug("Finished sending KV transfer for request %s", req_id)
            monitoring_recorder.record_delay_free_end(
                req_id, worker_ts.get(req_id))
            self._free_blocks(request)

    def _poll_fast_releases(self) -> None:
        """Change 3+5: drain deferred frees produced by the background
        block-free thread, then fall back to direct queue drain if no
        background thread is running.

        When the bg thread is active, it continuously drains
        _fast_release_queue, frees KV blocks under _kv_free_lock, and
        pushes cleanup items into _deferred_frees.  This method only
        handles the deferred cleanup (del requests, pin, monitoring).

        When the bg thread is NOT active (no connector, not producer),
        this method drains _fast_release_queue directly as before.
        """
        # --- Path A: bg thread is active, drain its deferred output ---
        if self._bg_free_thread is not None:
            self._drain_deferred_frees()
            return

        # --- Path B: no bg thread, direct drain (original Change 3) ---
        if self.connector is None:
            return
        poll_fn = getattr(self.connector, "poll_fast_releases", None)
        if poll_fn is None:
            return
        released = poll_fn()
        if not released:
            return
        now = time.time()
        for req_id, ts in released:
            request = self.requests.get(req_id)
            if request is None or not request.is_finished():
                continue
            ts.setdefault("finished_sending_ts", now)
            monitoring_recorder.record_delay_free_end(req_id, ts)
            self._free_blocks(request)
            self._fast_released_req_ids.add(req_id)

    # ------------------------------------------------------------------
    # Background block-free thread (Change 5)
    # ------------------------------------------------------------------

    def _bg_free_loop(self) -> None:
        """Background thread: drain fast-release queue and free KV blocks.

        This thread runs concurrently with execute_model().  It acquires
        _kv_free_lock only for the brief kv_cache_manager.free() call,
        so contention with the main thread (which holds the lock during
        allocate_slots in schedule()) is near-zero — they run in
        different phases of the engine core loop.

        After freeing blocks, it pushes a deferred cleanup item so that
        the main thread can handle del requests, pin logic, and
        monitoring at the next drain point.

        IMPORTANT: A RELEASE can arrive (via fast-release queue) BEFORE
        update_from_output() marks the request as finished.  This happens
        when migration completes within the same engine-core loop iteration.
        We must NOT discard these items — instead we keep them in a local
        pending buffer and retry on the next poll cycle.
        """
        poll_fn = getattr(self.connector, "poll_fast_releases", None)
        if poll_fn is None:
            logger.warning("bg_free_loop: connector has no poll_fast_releases")
            return

        is_continuum = (self.policy == SchedulingPolicy.CONTINUUM)
        # Pending items where the request was not yet finished when we
        # first saw the RELEASE.  Retried every poll cycle.
        # Each item: (req_id, ts_dict, first_seen_time)
        pending: list[tuple[str, dict, float]] = []
        # Max time to keep a pending item before discarding.
        # Handles the case where the request was force-freed by timeout                                                 
        # in get_finished() and no longer exists in self.requests.                                                      
        _PENDING_STALENESS_S = 30.0   

        while True:
            # Fix B: block ON the queue instead of poll+sleep, so the thread
            # truly OS-sleeps (GIL fully released) when idle — no busy wakeups.
            # When `pending` is non-empty, use a short timeout so pending is
            # rechecked promptly: pending items resolve on ENGINE-thread
            # progress (is_finished() flip), not on a queue event, so we must
            # wake periodically to recheck rather than block indefinitely.
            # When idle, use a long timeout for a genuine sleep. Bounded (never
            # None) so this daemon thread stays responsive at shutdown.
            released = poll_fn(timeout=(0.005 if pending else 0.5))
            if not released and not pending:
                # Timed out with nothing to do — sleep again.
                continue

            # Merge newly released items with pending retries.
            now = time.time()
            to_process: list[tuple[str, dict, float]] = list(pending)
            pending = []
            if released:
                for req_id, ts in released:
                    to_process.append((req_id, ts, now))

            for req_id, ts, first_seen in to_process:
                request = self.requests.get(req_id)
                if request is None:
                    # Request not in dict — might have been freed by
                    # another path (e.g. timeout or Change 6 fallback)
                    # or not yet added.
                    if now - first_seen < _PENDING_STALENESS_S:
                        pending.append((req_id, ts, first_seen))
                    else:
                        logger.debug(
                            "bg_free_loop: discarding stale pending "
                            "item %s (%.1fs old)", req_id,                                                              
                            now - first_seen)   
                    continue
                if not request.is_finished():
                    # RELEASE arrived before update_from_output() marked
                    # the request finished.  Keep for retry.
                    if now - first_seen < _PENDING_STALENESS_S:
                        pending.append((req_id, ts, first_seen))
                    else:
                        logger.warning(
                            "bg_free_loop: discarding item %s — "
                            "request not finished after %.1fs",
                            req_id, now - first_seen)
                    continue

                # Check if Continuum pin logic might keep blocks alive.
                # If so, do NOT free blocks here — defer to main thread.
                might_pin = (is_continuum
                             and not getattr(request, "is_last_step", True))
                if might_pin:
                    # blocks_freed = False → main thread will call _free_blocks
                    self._deferred_frees.put_nowait((req_id, ts, False))
                    continue

                # Safe to free blocks in bg thread.
                self._dec_delay_free_counter(request)
                with self._kv_free_lock:
                    self.kv_cache_manager.free(request)
                # Set block_freed_ts AFTER actual free so it accurately
                # reflects when blocks were returned to the pool.
                # (Previously set before checks via setdefault, which
                # was misleading when the item went to pending instead.)
                ts.setdefault("block_freed_ts", time.time())
                # blocks_freed = True → main thread only does cleanup
                self._deferred_frees.put_nowait((req_id, ts, True))

    def _drain_deferred_frees(self) -> None:
        """Main-thread: process deferred items from the bg free thread.

        Each item is (req_id, timestamps, blocks_freed).
        - blocks_freed=True:  bg thread already called kv_cache_manager.free(),
                              main thread only does del requests + monitoring.
        - blocks_freed=False: bg thread skipped free (request may need pin),
                              main thread calls full _free_blocks().
        """
        now = time.time()
        while True:
            try:
                item = self._deferred_frees.get_nowait()
            except Exception:
                break

            req_id, ts, blocks_freed = item
            request = self.requests.get(req_id)
            if request is None:
                continue

            # Record timestamps for monitoring.
            ts.setdefault("finished_sending_ts", ts.get("block_freed_ts", now))
            monitoring_recorder.record_delay_free_end(req_id, ts)

            if not blocks_freed:
                # Bg thread did NOT free blocks — run full _free_blocks
                # which handles Continuum pin logic correctly.
                self._free_blocks(request)
            else:
                # Blocks already freed by bg thread.  Only do cleanup.
                for req, end_time in list(self.pinned_requests):
                    if req.job_id == request.job_id:
                        self.unpin_request(req, end_time)
                        break
                del self.requests[req_id]

            self._fast_released_req_ids.add(req_id)
