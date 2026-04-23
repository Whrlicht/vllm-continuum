# Delay-Free Block Release Stall Bug Fix

## Problem

In disaggregated P/D serving with `BLOCK_MIGRATE` mode, the prefill (P)
scheduler can enter a cascading stall lasting tens of seconds.  During the
stall no compute steps execute, no HTTP responses are returned, and the
decode (D) side sits idle because it never receives forwarded requests from
the proxy.

### Observed symptoms (real trace)

| Metric | Value |
|--------|-------|
| Stall duration | 36.6 s (iter 202 → iter 203) |
| Empty iterations during stall | 5 157 |
| `kv_cache_usage` during stall | 0.9875 (constant) |
| `num_running` during stall | 6 (constant) |
| Bottlepy request `wait_bridge_pop` | 31.3 s |

### Root-cause chain

```
P finishes requests → blocks enter delay-free state
→ RELEASE arrives at P's listener thread
→ bg thread or main thread should free blocks
→ BUG: blocks are NOT freed (kv_usage unchanged)
→ _num_delay_free_blocks > 0 persists
→ schedule() running loop: allocate_slots fails + delay_free > 0 → break
→ total_num_scheduled_tokens = 0 → no forward pass
→ no request finishes → no HTTP response → Proxy blocked
→ D has no decode request → D never bridge_pop
→ next group of requests finishes, adds MORE delay-free blocks
→ cascading stall
```

## Analysis

### Two paths for freeing delay-free blocks

When a prefill request finishes and its KV must be migrated to D via
`BLOCK_MIGRATE`, the scheduler marks the request's blocks as "delay-free":

```python
# _free_request(), delay_free_blocks=True path
self._num_delay_free_blocks += n
self._delay_free_req_ids.add(request.request_id)
# blocks NOT freed, request stays in self.requests
```

After D migrates the blocks and sends a RELEASE RPC back to P, the blocks
should be freed.  There are two intended paths:

1. **Background thread path** (`_bg_free_loop`):
   Listener thread → `_fast_release_queue` → bg thread drains queue →
   `kv_cache_manager.free()` → `_deferred_frees` → main thread
   `_drain_deferred_frees()` does cleanup.

2. **Worker `get_finished` path** (`_update_from_kv_xfer_finished`):
   Worker-side `get_finished(finished_req_ids)` checks connector's
   `_delay_free_ts` for RELEASE received → returns `finished_sending` →
   scheduler `_update_from_kv_xfer_finished()` calls `_free_blocks()`.

### The bug: path 2 is dead during empty iterations

After a request finishes, its ID is added to `self.finished_req_ids` once
(in `_free_request`).  At the end of `schedule()`,
`_update_after_schedule()` resets `self.finished_req_ids = set()`.

In subsequent empty iterations (no scheduled tokens), `finished_req_ids`
is empty.  The worker's `get_finished({})` iterates over an empty set and
never checks RELEASE status for delay-free requests.
`kv_connector_no_forward()` returns `EMPTY_MODEL_RUNNER_OUTPUT`.
`_update_from_kv_xfer_finished()` is never called.

This means path 2 is completely dead during empty iterations, and the
system relies entirely on the bg thread (path 1).

### Why the bg thread alone is insufficient

The bg thread path has several fragility points:

- `block_freed_ts` was set *before* checking whether the request existed
  or was finished (via `ts.setdefault()`).  When the item went to
  `pending` instead of being freed, `block_freed_ts` was already set,
  making monitoring data misleading.

- If the request is freed by another code path (e.g., timeout in
  `get_finished`), the bg thread finds `request is None` and puts the
  item in pending.  After 30 s staleness timeout it is silently discarded
  — blocks are never freed.

- There is no error handling in the bg thread's free path.  An exception
  in `kv_cache_manager.free()` would kill the daemon thread, causing all
  subsequent RELEASE signals to pile up in `_fast_release_queue`
  permanently.

- `_dec_delay_free_counter` runs outside `_kv_free_lock`, creating a
  window where the counter is decremented but blocks are not yet freed.

When the bg thread fails to free blocks for any reason, there is no
fallback, and the stall persists until some external event (e.g., request
timeout) breaks the cycle.

## Fix

### Change 6: re-inject delay-free req IDs into `finished_req_ids`

**File:** `vllm/v1/core/sched/scheduler.py`, in `schedule()`

```python
# After _poll_fast_releases(), before the running loop:
if self._delay_free_req_ids:
    self.finished_req_ids.update(self._delay_free_req_ids)
```

This ensures that delay-free request IDs are included in every
`SchedulerOutput.finished_req_ids`, so the worker-side `get_finished()`
can check their RELEASE status.  When RELEASE has arrived:

1. `get_finished()` returns the req ID in `finished_sending`.
2. `kv_connector_no_forward()` returns a non-empty `kv_connector_output`.
3. `_update_from_kv_xfer_finished()` calls `_free_blocks()` for the
   request, immediately freeing blocks and decrementing
   `_num_delay_free_blocks`.
4. The next `schedule()` call sees free blocks, schedules running
   requests, and executes a compute step — breaking the stall.

This path is safe for double-free scenarios:
- If the bg thread already freed the blocks, `_fast_released_req_ids`
  contains the req ID, and `_update_from_kv_xfer_finished` skips it.
- If both paths race, `kv_cache_manager.free()` uses
  `req_to_blocks.pop(id, [])` — the second call pops an empty list
  (harmless no-op).
- `_dec_delay_free_counter` checks `req_id in _delay_free_req_ids` before
  decrementing — the second call finds it already removed and does
  nothing.

### Guard in `_update_from_kv_xfer_finished`

Added a `self.requests.get(req_id)` guard to handle the case where the bg
thread freed the request between `get_finished()` and
`_update_from_kv_xfer_finished()` executing.

### Fix misleading `block_freed_ts`

Moved `ts.setdefault("block_freed_ts", ...)` in `_bg_free_loop` from
before all checks to after the actual `kv_cache_manager.free()` call.
This ensures monitoring data accurately reflects when blocks were
returned to the pool, rather than when the RELEASE signal was first seen.

## Impact

- **Latency:** Eliminates cascading multi-second stalls caused by
  delay-free blocks not being freed.  In the observed trace, a single
  17 K-token request had `wait_bridge_pop = 31.3 s`; with this fix it
  would be < 2 s.

- **Correctness:** No change to correctness — the fix adds a redundant
  (but reliable) block-free path.  Double-free is safe by design.

- **Performance:** Negligible overhead.  The only added work per empty
  iteration is a set union (`finished_req_ids.update`) and the existing
  `get_finished()` check, both O(number of delay-free requests).
