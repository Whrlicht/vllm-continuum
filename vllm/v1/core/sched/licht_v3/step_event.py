# SPDX-License-Identifier: Apache-2.0
"""StepEvent: the per-scheduler-step message published by prefill and
consumed by decode-side ShadowScheduler.

Design notes
------------
* One message per `Scheduler.schedule()` call on prefill.
* Carries the FULL `waiting_now` / `running_now` snapshots (not deltas)
  — this is robust against any ZMQ message loss without needing a
  separate resync protocol.  Bandwidth is trivial (~200 KB/s at 100Hz
  with typical concurrency).
* Serialization is plain JSON.  Switching to msgpack later is one
  line if we ever need to.
* `traj_id` / `agent_round` on each req entry let the decode side
  look up its pending registrations and run the Stage-2 correction
  (replace estimated num_tokens with the real value once a request
  actually shows up in prefill).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field  # noqa: F401
from typing import Optional


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

@dataclass
class ReqSnapshot:
    """One request's state as seen by prefill at the end of a step."""
    request_id: str
    traj_id: Optional[str]
    agent_round: Optional[int]
    num_prompt_tokens: int
    hit_length: int = 0
    # admit_step is set for RUNNING requests; arrival_step for WAITING.
    admit_step: Optional[int] = None
    arrival_step: Optional[int] = None
    # R_remaining = how many chunked-prefill chunks this running req
    # has left.  Decode side derives this for its sim mirror.
    r_remaining: Optional[int] = None
    evictable_prefix: int = 0


@dataclass
class StepEvent:
    """Snapshot of prefill scheduler state at the end of one schedule()
    call, published to all subscribed decode-side shadow schedulers."""
    step_id: int                      # = scheduler.schedule() call count
    step_wall_ts: float               # time.time() at emit
    sec_per_step_recent: float        # rolling sec-per-step estimate
    # Delta lists (this step only)
    admitted: list[ReqSnapshot] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)
    preempted: list[str] = field(default_factory=list)
    # Full snapshots at end of this step
    waiting_now: list[ReqSnapshot] = field(default_factory=list)
    running_now: list[ReqSnapshot] = field(default_factory=list)
    # Scheduler constants needed by simulator
    max_num_seqs: int = 0
    max_num_batched_tokens: int = 0
    block_size: int = 16
    total_kv_blocks: int = 0
    # Total tokens scheduled in this step (sum of num_scheduled_tokens
    # across all reqs that worker.execute will process).  Used by the
    # decode-side StepTimeModel to predict wall-clock duration.
    num_scheduled_tokens_this_step: int = 0
    # ---- LICHTV2 timeline + admission constants ----
    # Full prefill backfill timeline as built by
    # _licht_v2_build_timeline().  decode-side simulator left-shifts
    # this by one per simulated step and fills the tail with
    # total_kv_blocks ("future we don't know, assume max free").
    future_free: list[int] = field(default_factory=list)
    future_alloc: list[int] = field(default_factory=list)
    lichtv2_horizon_n: int = 0
    chunk_size_tokens: int = 0
    max_alloc_per_step_blocks: int = 0
    long_tail_headroom_blocks: int = 0
    long_running_count: int = 0
    max_long_bridge: int = 2
    # LICHT prefill scoring (used to order admit candidates)
    score_a: float = 3.0
    score_b: float = 1.0
    score_tmax_s: float = 120.0
    round_decay_alpha: float = 0.5


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def encode_step_event(evt: StepEvent) -> bytes:
    return json.dumps(asdict(evt), ensure_ascii=False).encode("utf-8")


def decode_step_event(buf: bytes) -> StepEvent:
    obj = json.loads(buf.decode("utf-8"))
    return StepEvent(
        step_id=int(obj["step_id"]),
        step_wall_ts=float(obj["step_wall_ts"]),
        sec_per_step_recent=float(obj.get("sec_per_step_recent", 0.05)),
        admitted=[_req_from_dict(d) for d in obj.get("admitted", [])],
        finished=[str(x) for x in obj.get("finished", [])],
        preempted=[str(x) for x in obj.get("preempted", [])],
        waiting_now=[_req_from_dict(d) for d in obj.get("waiting_now", [])],
        running_now=[_req_from_dict(d) for d in obj.get("running_now", [])],
        max_num_seqs=int(obj.get("max_num_seqs", 0)),
        max_num_batched_tokens=int(obj.get("max_num_batched_tokens", 0)),
        block_size=int(obj.get("block_size", 16)),
        total_kv_blocks=int(obj.get("total_kv_blocks", 0)),
        num_scheduled_tokens_this_step=int(
            obj.get("num_scheduled_tokens_this_step", 0)),
        future_free=[int(x) for x in obj.get("future_free", [])],
        future_alloc=[int(x) for x in obj.get("future_alloc", [])],
        lichtv2_horizon_n=int(obj.get("lichtv2_horizon_n", 0)),
        chunk_size_tokens=int(obj.get("chunk_size_tokens", 0)),
        max_alloc_per_step_blocks=int(
            obj.get("max_alloc_per_step_blocks", 0)),
        long_tail_headroom_blocks=int(
            obj.get("long_tail_headroom_blocks", 0)),
        long_running_count=int(obj.get("long_running_count", 0)),
        max_long_bridge=int(obj.get("max_long_bridge", 2)),
        score_a=float(obj.get("score_a", 3.0)),
        score_b=float(obj.get("score_b", 1.0)),
        score_tmax_s=float(obj.get("score_tmax_s", 120.0)),
        round_decay_alpha=float(obj.get("round_decay_alpha", 0.5)),
    )


def _req_from_dict(d: dict) -> ReqSnapshot:
    return ReqSnapshot(
        request_id=str(d["request_id"]),
        traj_id=(None if d.get("traj_id") is None
                 else str(d["traj_id"])),
        agent_round=(None if d.get("agent_round") is None
                     else int(d["agent_round"])),
        num_prompt_tokens=int(d.get("num_prompt_tokens", 0)),
        hit_length=int(d.get("hit_length", 0)),
        admit_step=(None if d.get("admit_step") is None
                    else int(d["admit_step"])),
        arrival_step=(None if d.get("arrival_step") is None
                      else int(d["arrival_step"])),
        r_remaining=(None if d.get("r_remaining") is None
                     else int(d["r_remaining"])),
        evictable_prefix=int(d.get("evictable_prefix", 0)),
    )
