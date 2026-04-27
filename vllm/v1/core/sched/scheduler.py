# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
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

        # LICHT is a global algorithm switch. For now it only changes
        # scheduler behavior; KV transfer strategy remains default.
        self.instance_role = _resolve_instance_role(vllm_config)
        self.licht_enabled = self.scheduler_config.licht
        self.licht_prefill_sched_enabled = (self.licht_enabled
                                            and self.instance_role != "decode")
        self.licht_decode_fcfs_enabled = (self.licht_enabled
                                          and self.instance_role == "decode")
        self.licht_waiting_round_start_ts: dict[str, float] = {}
        # Wall-clock timestamp of the most recent admission into running.
        # Used by _pick_preempt_victim_licht to enforce a min-run grace
        # so that just-admitted requests are not immediately evicted.
        self.licht_running_admit_ts: dict[str, float] = {}
        if self.licht_enabled:
            logger.info(
                "LICHT mode enabled (instance_role=%s). "
                "KV transfer strategy currently uses default implementation.",
                self.instance_role,
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
    
    def schedule(self) -> SchedulerOutput:
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

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)


            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
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

        # Next, schedule the WAITING requests.
        # TODO (Hanchen) need to add scheduling logic for returns from functions. It should not be FCFS
        if not preempted_reqs:
            self._ensure_licht_waiting_start_timestamps()
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self._peek_waiting_request()

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

                    # NOTE (Hanchen) The logic here is that we will see if the connector can get the tokens. 
                    # If it can, we will use them.

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))

                        # NOTE (Hanchen) this will not be called in cpu offloading.
                        if num_external_computed_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self._pop_waiting_request(request)
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

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
                    if (0 < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)

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
                if self.licht_prefill_sched_enabled:
                    # Stamp the admission time so the LICHT preempt
                    # selector can grant a min-run grace window before
                    # this request becomes evictable again.
                    self.licht_running_admit_ts[request.request_id] = (
                        time.time())
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    self.continuum_recorder.request_waiting_to_running(
                        request, 
                        prompt_length=request.num_prompt_tokens,
                        hit_length=num_computed_tokens
                    )
                    scheduled_new_reqs.append(request)
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
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
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
        
        return scheduler_output

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
        self.tool_call_estimator.request_arrives(request)
        self.continuum_recorder.request_arrives(request)

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

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()
        self.tool_call_estimator.request_finished(request)
        self.continuum_recorder.request_finished(request)
        self._drop_licht_waiting_state(request.request_id)
        self.licht_running_admit_ts.pop(request.request_id, None)

        # NOTE (Hanchen) in unpin, we need to make sure it is not delay free blocks because it could be still waiting for transfer, need to copy something similar to the kv_xfer_params

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
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
            # Block until at least one item is available.
            released = poll_fn()
            if not released and not pending:
                # poll_fn returned empty and no pending — sleep briefly.
                time.sleep(0.001)
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
