# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LICHTV3 decode-side predictor coordinator.

Module layout:
  - predictors                 : ToolTimePredictorWrapper around
                                 tool_call_time.Predictor (or fallback).
  - features_adapter           : Request → tool_call dict + TrajectoryState
                                 feature extraction for the tool predictor.
  - step_event                 : StepEvent wire format published by prefill
                                 and consumed by the decode ShadowScheduler.
  - shadow_scheduler           : event-driven mirror of prefill's scheduler;
                                 sole K_queue (stage1/2) predictor + step-time.
  - online_step_time_predictor : RLS step-duration model used by the shadow.
  - decode_manager             : top-level coordinator wired into Scheduler.

Only `decode_manager.LichtV3DecodeManager` is meant to be imported from
the scheduler.  All other modules are private to v3.

NOTE: the legacy file-based snapshot IPC (snapshot_io +
licht_v3_snapshot) and its ProductionStepCountPredictor, along with the
回传 KV push-back machinery (tier_planner / warm_pool / prewarm /
install center), were removed once the ShadowScheduler ZMQ path became
the sole K_queue predictor and KV reverted to plain delay-free release.
"""
