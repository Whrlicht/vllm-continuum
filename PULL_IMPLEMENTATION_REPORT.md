# Pull KV Migration Implementation Report

## Goal
Implement full decode-driven pull for disaggregated KV transfer, and close correctness/resource-lifecycle gaps.

## What Was Changed

### 1) P2P engine pull semantics and lifecycle hardening
File: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`

- Added decode-side GET retry loop:
  - `get_retry_timeout_s` (default 30s)
  - `get_retry_interval_s` (default 0.002s)
- Added request-level completion tracking:
  - `expected_tensor_ids`
  - `send_request_id_to_tensor_ids`
  - `recv_request_id_to_tensor_ids`
  - `pending_sending_deadlines`
  - `pending_recving_deadlines`
- Added request-level cleanup with timeout fallback:
  - `request_completion_timeout_s` (default 120s)
  - cleanup now frees send/recv stores and pooled memory deterministically
- Added blocking backpressure for GET producer staging buffer:
  - removed silent oldest-entry drop behavior for GET
  - if producer buffer cannot make progress before timeout, transfer returns failure
- Updated GET server path:
  - no LRU pop/reinsert side effect
  - marks per-tensor send completion when GET is served
- Updated send completion accounting:
  - `send_sync` now marks sent tensor IDs consistently

### 2) Connector switched to explicit decode pull
File: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`

- Consumer (`start_load_kv`) now parses prefill endpoint from request id and actively pulls:
  - `parse_request_id(..., is_prefill=False)`
  - passes `remote_address` to `recv_tensor(...)` for each layer
- Missing KV is now a hard failure (raises), not a warning-and-continue.
- Producer (`save_kv_layer`) now checks send/stage return value and raises on failure.
- Scheduler-side `request_finished` now delays free for producer when transfer mode is async/pull:
  - returns `True` for `send_type in {PUT_ASYNC, GET}`

### 3) Production launcher defaults moved to pull mode
File: `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/run_disagg_p2p_nccl_xpyd_prod.sh`

- Default transfer mode changed to pull:
  - `KV_SEND_TYPE=GET`
- Added tunables and CLI options:
  - `--kv-send-type`
  - `--request-completion-timeout`
  - `--get-retry-timeout`
  - `--get-retry-interval`
- Propagated these values to both producer and consumer `kv_connector_extra_config`.

### 4) Demo launcher aligned to pull mode
File: `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh`

- Changed both producer and consumer `send_type` from `PUT_ASYNC` to `GET`.

## Current Data Flow (After This Change)

1. Proxy chooses one prefill node and one decode node, embeds both addresses in `request_id`.
2. Prefill worker computes prompt KV and stages per-layer KV by request/layer key.
3. Prefill request finishes; scheduler calls connector `request_finished`.
4. Producer returns delayed-free (`True`) for pull mode, so blocks are not freed immediately.
5. Decode worker starts execution, parses prefill address from `request_id`, and actively issues GET per layer.
6. Producer receives GET, serves KV over NCCL, and marks per-layer send completion.
7. Decode receives KV and injects into paged KV cache; missing KV now fails fast.
8. Worker connector reports transfer completion through `finished_sending`/`finished_recving`.
9. Scheduler receives completion and releases delayed producer blocks.
10. Engine request cleanup frees transfer buffers and tracking state; timeout fallback prevents permanent pinning.

## Why This Is Now Pull

- Transfer initiation is decode-side (consumer-driven GET), not producer push.
- Producer only stages KV and waits for consumer retrieval.
- Producer block release is coupled to migration completion signal, not request finish timing.

## Operational Notes

- Pull mode is now the default in the production launcher.
- If decode cannot pull KV in time, transfer fails explicitly instead of silently decoding with missing KV.
- Timeouts are configurable from launcher arguments and forwarded into connector runtime config.
