# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LICHTV3 decode-side round-gap KV manager.

Module layout:
  - snapshot_io       : prefill-side write + decode-side read of the
                        scheduler snapshot (file-based IPC).
  - predictors        : ProductionStepCountPredictor wrapping the offline
                        queue_time simulator; ToolTimePredictorWrapper
                        for tool_call_time.Predictor (or fallback).
  - tier_planner      : (K_queue, T_tool) → GPU / CPU / SSD / DROP.
  - warm_pool         : KV warm-pool data structure (scaffold).
  - prewarm           : prewarm timer thread + KV push stub.
  - decode_manager    : top-level coordinator wired into Scheduler.

Only `decode_manager.LichtV3DecodeManager` is meant to be imported from
the scheduler.  All other modules are private to v3.
"""
