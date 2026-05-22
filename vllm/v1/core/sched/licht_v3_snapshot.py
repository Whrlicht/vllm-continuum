# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LICHTV3 prefill-scheduler snapshot dataclasses.

These types are the contract between the prefill `Scheduler` (producer)
and the LICHTV3 decode-side queue-time predictor (consumer).  The
snapshot is built by `Scheduler.snapshot_for_v3_simulator()` and is
strictly read-only with respect to the live scheduler state.

The snapshot intentionally mirrors what the offline `LichtV2Simulator`
consumed from oracle data — running + waiting views plus the timeline
arrays — but every field here is derived from runtime scheduler state
without any oracle access to the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LichtV3RunningView:
    """A prefill-running request as seen by the v3 simulator."""
    request_id: str
    arrival_ts: float
    admit_ts: float
    num_prompt_tokens: int
    num_computed_tokens: int
    # Remaining prefill scheduler chunks computed via
    # Scheduler._licht_v2_R_at(req, req.num_computed_tokens).
    r_remaining_chunks: int
    # Real agent/dialog round from API metadata (== request.agent_round).
    agent_round: int


@dataclass(frozen=True)
class LichtV3WaitingView:
    """A prefill-waiting request as seen by the v3 simulator."""
    request_id: str
    arrival_ts: float
    num_prompt_tokens: int
    # Full-prefill chunk count (from offset=0).  Same formula as
    # Scheduler._licht_v2_R_at(req, 0).
    r_full_chunks: int
    agent_round: int


@dataclass(frozen=True)
class LichtV3Constants:
    """LICHT/LICHTV2 score and timeline constants in effect on the
    producing scheduler.  Copied into the snapshot so the consumer
    does not need to import vLLM internals to compute scores."""
    score_a: float
    score_b: float
    score_tmax_s: float
    round_decay_alpha: float
    lichtv2_horizon_n: int
    chunk_size_tokens: int


@dataclass(frozen=True)
class LichtV3PrefillSnapshot:
    """Read-only snapshot of a prefill scheduler at a wall-clock instant."""
    timestamp: float
    instance_role: str
    block_size: int
    free_blocks: int
    total_kv_blocks: int
    max_num_seqs: int
    max_num_batched_tokens: int
    constants: LichtV3Constants
    running: tuple[LichtV3RunningView, ...]
    waiting: tuple[LichtV3WaitingView, ...]
    # Live licht-v2 backfill timeline; both arrays have length N+1 when
    # licht_v2_prefill_sched_enabled is on, otherwise None.
    licht_v2_future_free: Optional[tuple[int, ...]] = None
    licht_v2_future_alloc: Optional[tuple[int, ...]] = None
    # Network address (ip:port) where the prefill's P2P NCCL engine
    # ROUTER socket is bound.  The decode-side ConnectorBridge needs
    # this to send V3_RESERVE_AND_QUERY / V3_INSTALL RPCs; the env
    # var `LICHT_V3_PREFILL_KV_ADDRESS` is an unreliable fallback
    # because `127.0.0.1` does not work when the engine binds to the
    # machine's real interface IP via `get_ip()`.
    prefill_kv_zmq_address: Optional[str] = None
