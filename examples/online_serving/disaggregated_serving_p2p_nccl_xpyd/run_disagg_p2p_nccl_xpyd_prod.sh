#!/usr/bin/env bash

set -Eeuo pipefail

# Production launcher for P2P NCCL XpYd disaggregated serving.
# Example:
#   ./run_disagg_p2p_nccl_xpyd_prod.sh --prefill-gpus 0 --decode-gpus 1,2

MODEL_PATH="/data/huggingface/models--meta-llama--Llama-3.1-8B-Instruct"
PREFILL_GPUS="7"
DECODE_GPUS="4,5,6"

PROXY_DISCOVERY_HOST="0.0.0.0"
PROXY_DISCOVERY_PORT=30001
PROXY_API_HOST="0.0.0.0"
PROXY_API_PORT=10234

PROXY_IP_FOR_WORKERS="127.0.0.1"
PREFILL_HTTP_PORT_BASE=20003
DECODE_HTTP_PORT_BASE=20005
PREFILL_KV_PORT_BASE=21001
DECODE_KV_PORT_BASE=22001
# LICHT-V3 StepEvent channel: prefill scheduler PUBLISHes its per-step
# state here; the decode-side ShadowScheduler SUBSCRIBEs to run the
# K_queue (stage1/2) + step-time predictors.  prefill[i] binds base+i*2;
# decode[i] connects to the matching prefill (round-robin if D>P).
STEP_EVENT_PORT_BASE=25001

PREFILL_GPU_MEMORY_UTILIZATION=0.95
DECODE_GPU_MEMORY_UTILIZATION=0.95
# CUDA graph on the DECODE worker only (prefill stays --enforce-eager: graphs
# don't help variable-length chunked prefill and risk the long-ctx path).
# Decode-only cudagraph + eager-prefill is an officially supported P/D pattern
# (gpu_model_runner.py:~1680).  Default OFF (both eager, unchanged).  Enable
# with --decode-cuda-graph.  NOTE: capture needs spare VRAM; at gpu-mem-util
# 0.95 it may OOM at "Capturing CUDA graphs" — if so, lower
# DECODE_GPU_MEMORY_UTILIZATION to ~0.90.  Verify trace_replay outputs still
# match after enabling (correctness check that the KV-connector hooks captured
# cleanly).
DECODE_CUDA_GRAPH=false

# DistServe-style direct block migration mode.
# Decode actively pops bridge metadata and migrates blocks from prefill.
KV_SEND_TYPE="BLOCK_MIGRATE"
# LICHT cross-round KV reuse: when set to a host-shared dir (e.g.
# /dev/shm/licht_round_kv for a RAM/"CPU" tier, or an SSD mount), decode
# persists each finished round's full-sequence KV there and the next
# round's prefill loads that prefix straight into GPU (skipping recompute).
# Empty here = use the default below.  Under --licht-v3 it auto-enables
# at /dev/shm/licht_round_kv unless --no-round-kv is given.  Override with
# --round-kv-reuse-path PATH (also works without --licht-v3).
ROUND_KV_REUSE_PATH=""
ROUND_KV_DEFAULT_PATH="/dev/shm/licht_round_kv"
NO_ROUND_KV=false
# Layer-wise pipelined round-kv load: prefill loads layer i+1's reused
# prefix while computing layer i (vs the default: read+scatter all layers
# before the forward, blocking).  Off by default; enable with
# --round-kv-pipeline (or env LICHT_ROUND_KV_PIPELINE=1).
ROUND_KV_PIPELINE=false
# Diagnostic profiling of the round-kv load (contention probe + per-segment
# pin_copy/h2d/index timing).  Adds cuda syncs => slower; use only to find
# the bottleneck, then turn off.  Enable with --round-kv-profile.
ROUND_KV_PROFILE=false
# Background HBM headroom dual-probe (pure-H2D vs H2D+scatter, every ~30ms),
# to see how much DMA/SM bandwidth is available DURING the real forward.
# Diagnostic; adds a small constant load — turn off after.  Enable --hbm-probe.
HBM_PROBE=false
# Round-kv ASYNC load (default OFF): async parks requests for a variable
# load-dependent time, which breaks the LICHT-V3 admit predictor.  Default is
# SYNCHRONOUS (admit a batch -> load all their KV -> run together).  Pass
# --round-kv-async to experiment with the non-blocking async load.
ROUND_KV_ASYNC=false
# Round-kv RAW storage (default ON): store KV as contiguous .bin chunks and
# load via mmap + bulk H2D + GPU scatter (no safetensors, no strided read).
# Kills the read bottleneck.  --no-round-kv-raw -> old safetensors path (A/B).
ROUND_KV_RAW=true
# Round-kv ARENA (default ON): resident shared PINNED arena (LMCache-style).
# decode memcpys KV into a /dev/shm region that prefill cudaHostRegisters once,
# so a load is a DIRECT H2D (~24GB/s) — no per-load file read / mmap->pinned
# copy / page faults.  Supersedes RAW.  --no-round-kv-arena -> RAW .bin path.
# --round-kv-arena-gb N sizes the arena (default 24; ring-evicts oldest when
# full; prefill pays a one-time ~N/2 s cudaHostRegister at startup).
ROUND_KV_ARENA=true
ROUND_KV_ARENA_GB="400"
# DIAGNOSTIC: drain the GPU before each round-kv load + time it, to tell
# contention-with-prior-forward apart from op-inefficiency.  --round-kv-sync-first.
ROUND_KV_SYNC_FIRST=false
# Fused multi-layer scatter CUDA kernel: replace per-chunk nL index_puts with
# ONE kernel launch (cuts CPU dispatch that starves the GPU in serving).
# --round-kv-fused to enable (opt-in until serving-validated).
ROUND_KV_FUSED=false
# Stage 2 LRU arena (slot-paged bitmap alloc + per-job LRU + tail-first evict +
# self-heal, cross-process mutex + reader pin). Replaces the FIFO ring arena.
# --round-kv-lru to enable (opt-in until serving-validated).
# 注意: 必须小写 "true" — 下面 export 的判断是 [[ ... == "true" ]], 大写 True
# 不匹配会静默退回 FIFO arena (内容寻址 dedup 只在 LRU 路径生效, 会失效).
ROUND_KV_LRU=true
# Stage 6 内容寻址 dedup (LICHT_ARENA_CONTENT_ADDR): 全局 hash 表 + per-slot
# refcnt, 跨 job 共享 prefix 只存一份 (store 命中 refcnt++ 不重复分配 slot, 不
# 重复 D2H).  需 ROUND_KV_LRU=true.  prefill/decode 两端由同一 export 继承,
# 取值自动一致 (两端必须相同, 否则 .slot v1/v2 不匹配会毁数据).  默认关;
# --arena-content-addr 开启.  调试看 dedup 命中率: 额外 export LICHT_ARENA_DEBUG=1.
ARENA_CONTENT_ADDR=false
NO_ARENA_CONTENT_ADDR=false   # --no-arena-content-addr: opt out even under --licht-v3
# decode(consumer)也 cudaHostRegister arena → 走直读 kernel(无 GPU staging).
# 默认开. 关掉则 consumer 复用 load 走逐请求 staging, 把整段前缀一次性搬上 GPU,
# 长前缀(SWE-bench 等)在 gpu-mem-util~0.95 下 OOM(load_request CUDA OOM 刷屏).
# 代价: decode 启动多付一次 cudaHostRegister(256GB arena ~分钟级) + 双向 pin.
# --no-arena-consumer-direct 退回旧 consumer-mmap-only.
ARENA_CONSUMER_DIRECT=true
# P2 提带宽: mbind(MPOL_INTERLEAVE) 把 256GB arena 摊到所有 NUMA 节点, 缓解单节点
# 内存控制器饱和 (大复用 load 读 + store 写争用). 默认关 (拓扑相关, 单 GPU DMA 可能
# 跨 socket 反略降, 建议 A/B). --arena-numa-interleave 开.
ARENA_NUMA_INTERLEAVE=false
# Bind each worker to its GPU's local NUMA node (numactl) so pinned buffers
# and H2D transfers are node-local (~2x H2D on multi-socket boxes).  Defaults
# ON under --licht-v3; disable with --no-numa-bind, or force on standalone
# with --numa-bind.
NUMA_BIND=false
NO_NUMA_BIND=false
REQUEST_COMPLETION_TIMEOUT_S=600
GET_RETRY_TIMEOUT_S=60
GET_RETRY_INTERVAL_S=0.005

DTYPE="float16"
# Keep 0 as "auto": use model's own max context length.
MAX_MODEL_LEN=0
MAX_NUM_BATCHED_TOKENS=265944
MAX_NUM_SEQS=256
SEED=1024
LICHT=false
LICHT_V2=false
LICHT_V3=false
# --prefill-opt: FINAL optimal prefill stack (all env-gated, prefill-only):
#   shorts-first longcap_fcfs : shorts get priority + are UNcapped; longs are
#       throttled only when KV is near-full -- a new long is rejected if total
#       ACTUAL usage + its footprint would leave shorts less than
#       LICHT_SHORT_RESERVE (0.2) free (LICHT_LONGCAP_ORDER=short +
#       LICHT_LONG_THETA presence enables it). When KV is abundant longs flow
#       freely (replaces the old fixed "longs <= 30% footprint" cap, which
#       blocked longs even at low real usage).
#   C=5120 boundary           : short/long split (LICHT_LONG_C; also drives the
#       dynamic_chunk cstar).
#   reservation               : oldest waiting long gets a reserved slot so a
#       short flood can't starve it (LICHT_LONG_RESV=1).
#   FCFS-break                : a long that can't be admitted STOPS the long lane
#       (strict FCFS among longs; younger longs can't jump it) -> fixes the
#       big-prefix-long starvation (max 230->148) (LICHT_LONGCAP_FCFS_BREAK=1).
#   dynamic_chunk (mode F)    : per-step chunk size = sqrt(num/den) from calibrated
#       beta_r/b. num = re-read (brb*sum lam*D*C) + shared "big waits for shorts"
#       (T_short*sum lam*C / N_long); den = drag (W_soft * sum lam*D). T_short is
#       running shorts only; W_soft counts running shorts + waiting (SHORT_SET=all).
#       Cures mode E over-chunking (LICHT_DYN_CHUNK=F + SHORT_SET=all + BRB_FILE).
# vs fcfs: p50 47->3s (15x), p99/max higher (longs wait). Requires the LICHT-V2
# timeline scheduler (auto-enabled below). Off = no change to scheduling.
PREFILL_OPT=false
PREFILL_FCFS=false           # --prefill-fcfs: LICHT-V3 + pure-FCFS priority + fixed
                             # native chunk (no longcap, no dynamic_chunk). Diagnostic
                             # baseline; mutually exclusive with --prefill-opt.
PREFILL_OPT_THETA=0.3        # long-lane KV ceiling (head-of-line safe; sweep best)
PREFILL_OPT_LONGC=5120       # short/long boundary C (sweep best for break stack)
PREFILL_OPT_CLOW=2048        # dynamic_chunk mode F: smooth long/short band low (lambda=0 below)
PREFILL_OPT_CHIGH=5120       # dynamic_chunk mode F: smooth band high (lambda=1 above)
PREFILL_OPT_SHORT_RESERVE=0.2 # keep this fraction of KV free for shorts (lower = use more KV)

WAIT_TIMEOUT_SECONDS=1200
SHUTDOWN_GRACE_SECONDS=20
CLIENT_STOP_GRACE_SECONDS=20
FAIL_ON_WAIT_TIMEOUT=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRACE_REPLAY_PATH="${REPO_ROOT}/trace_data/mixed"
# LICHT-V3 tool-time predictor bundle (tool_call_time): bash family → ML
# p50/p95, editor/submit family → bucket-median 查表.  Unset/missing →
# the predictor degrades to a constant fallback.  Override: --tool-predictor-dir.
TOOL_PREDICTOR_DIR="${REPO_ROOT}/tool_call_time/runs/run_2902_v2"
PROXY_SCRIPT="${SCRIPT_DIR}/disagg_proxy_p2p_nccl_xpyd_prod.py"
CLIENT_PID_FILE="${REPO_ROOT}/output/.multiturn_trace_client.pid"
STOP_CLIENT_ON_EXIT=true

PIDS=()
EXPECTED_TIMESTAMP_FILES=()

usage() {
  cat <<EOF
Usage:
  $0 [options]

Options:
  --model-path PATH            Model path (default: ${MODEL_PATH})
  --prefill-gpus IDS           Comma-separated prefill GPU IDs (default: ${PREFILL_GPUS})
  --decode-gpus IDS            Comma-separated decode GPU IDs (default: ${DECODE_GPUS})
  --proxy-discovery-host HOST  Proxy ZMQ bind host (default: ${PROXY_DISCOVERY_HOST})
  --proxy-discovery-port PORT  Proxy ZMQ bind port (default: ${PROXY_DISCOVERY_PORT})
  --proxy-api-host HOST        Proxy HTTP bind host (default: ${PROXY_API_HOST})
  --proxy-api-port PORT        Proxy HTTP bind port (default: ${PROXY_API_PORT})
  --proxy-ip-for-workers IP    Worker-visible proxy IP (default: ${PROXY_IP_FOR_WORKERS})
  --kv-send-type MODE          KV transfer mode, GET enables pull (default: ${KV_SEND_TYPE})
  --round-kv-reuse-path PATH   Host-shared dir for cross-round KV reuse
                               (e.g. /dev/shm/licht_round_kv).  Under
                               --licht-v3 this defaults to
                               ${ROUND_KV_DEFAULT_PATH}
  --no-round-kv                Disable cross-round KV reuse even under
                               --licht-v3 (for A/B comparison)
  --round-kv-pipeline          Layer-wise pipelined load: load layer i+1's
                               reused prefix while computing layer i (instead
                               of loading all layers before the forward).
                               Off by default; env LICHT_ROUND_KV_PIPELINE=1.
  --round-kv-profile           Diagnostic: log a contention probe + per-segment
                               (pin_copy/h2d/index) timing for each round-kv
                               load.  Adds cuda syncs (slower) — diagnostic
                               only.  Look for 'round-kv PROFILE:' in the log.
  --hbm-probe                  Diagnostic: background dual-probe measuring
                               pure-H2D (DMA) vs H2D+scatter (SM) bandwidth
                               available during the real forward.  Look for
                               'round-kv HBM-PROBE:' in the log.  Turn off after.
  --arena-content-addr         Stage 6 内容寻址 dedup: 跨 job 共享 prefix 只存一
                               份 (hash 表 + refcnt).  需 --round-kv-lru.  两端
                               自动一致 (同一 export 继承).  默认关.
  --no-arena-consumer-direct   decode 不 register arena (退回逐请求 staging).
                               默认 decode 也 register 走直读 (无 staging, 避免
                               长前缀复用 load 把整段搬 GPU 导致 CUDA OOM).
  --arena-numa-interleave      mbind arena 跨 NUMA 节点摊内存带宽 (缓解大复用
                               load 读 + store 写 争用单节点). 默认关, 拓扑相关,
                               建议 A/B 实测.
  --numa-bind                  Bind each worker to its GPU's local NUMA node
                               (numactl) for faster pinned H2D.  Defaults ON
                               under --licht-v3.
  --no-numa-bind               Disable NUMA binding even under --licht-v3
  --request-completion-timeout SECONDS
                               Timeout before forcing request KV cleanup
                               (default: ${REQUEST_COMPLETION_TIMEOUT_S})
  --get-retry-timeout SECONDS  Bridge/IPC retry timeout per request
                               (default: ${GET_RETRY_TIMEOUT_S})
  --get-retry-interval SECONDS Bridge/IPC retry polling interval
                               (default: ${GET_RETRY_INTERVAL_S})
  --max-model-len N            0=auto (follow model max context), >0=override
  --max-num-batched-tokens N   0=auto, >0=override
  --licht                      Enable LICHT algorithm switch
                               (prefill dynamic priority + decode FCFS)
  --licht-v2                   Enable LICHT-V2 (prefill timeline scheduler)
  --licht-v3                   Enable LICHT-V3 (LICHT-V2 timeline + the
                               tool/K_queue/step predictors)
  --tool-predictor-dir PATH    LICHT-V3 tool-time predictor bundle dir
                               (default: ${TOOL_PREDICTOR_DIR})
  --trace-replay-path PATH     Trace replay JSON path for workers
                               (default: ${TRACE_REPLAY_PATH})
  --wait-timeout SECONDS       Wait timeout for each worker endpoint (default: ${WAIT_TIMEOUT_SECONDS})
  --fail-on-wait-timeout       Exit launcher if any worker readiness check times out
                               (default: continue running and wait for Ctrl+C)
  --shutdown-grace-seconds N   Grace window per signal phase before force kill
                               (default: ${SHUTDOWN_GRACE_SECONDS})
  --client-pid-file PATH       PID file for multiturn_trace_client.py
                               (default: ${CLIENT_PID_FILE})
  --no-stop-client-on-exit     Do not signal client process on launcher exit
  --client-stop-grace-seconds N
                               Grace time for client shutdown before escalation
                               (default: ${CLIENT_STOP_GRACE_SECONDS})
  -h, --help                   Show this help

Notes:
  1) You can independently choose P and D GPU lists.
  2) Keep prefill/decode counts flexible; proxy will do round-robin per role.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --prefill-gpus)
      PREFILL_GPUS="$2"
      shift 2
      ;;
    --decode-gpus)
      DECODE_GPUS="$2"
      shift 2
      ;;
    --proxy-discovery-host)
      PROXY_DISCOVERY_HOST="$2"
      shift 2
      ;;
    --proxy-discovery-port)
      PROXY_DISCOVERY_PORT="$2"
      shift 2
      ;;
    --proxy-api-host)
      PROXY_API_HOST="$2"
      shift 2
      ;;
    --proxy-api-port)
      PROXY_API_PORT="$2"
      shift 2
      ;;
    --proxy-ip-for-workers)
      PROXY_IP_FOR_WORKERS="$2"
      shift 2
      ;;
    --kv-send-type)
      KV_SEND_TYPE="$2"
      shift 2
      ;;
    --round-kv-reuse-path)
      ROUND_KV_REUSE_PATH="$2"
      shift 2
      ;;
    --no-round-kv)
      NO_ROUND_KV=true
      shift
      ;;
    --round-kv-pipeline)
      ROUND_KV_PIPELINE=true
      shift
      ;;
    --round-kv-profile)
      ROUND_KV_PROFILE=true
      shift
      ;;
    --hbm-probe)
      HBM_PROBE=true
      shift
      ;;
    --round-kv-async)
      ROUND_KV_ASYNC=true
      shift
      ;;
    --no-round-kv-raw)
      ROUND_KV_RAW=false
      shift
      ;;
    --no-round-kv-arena)
      ROUND_KV_ARENA=false
      shift
      ;;
    --round-kv-arena-gb)
      ROUND_KV_ARENA_GB="$2"
      shift 2
      ;;
    --round-kv-sync-first)
      ROUND_KV_SYNC_FIRST=true
      shift
      ;;
    --round-kv-fused)
      ROUND_KV_FUSED=true
      shift
      ;;
    --round-kv-lru)
      ROUND_KV_LRU=true
      shift
      ;;
    --arena-content-addr)
      ARENA_CONTENT_ADDR=true
      shift
      ;;
    --no-arena-content-addr)
      NO_ARENA_CONTENT_ADDR=true
      ARENA_CONTENT_ADDR=false
      shift
      ;;
    --no-arena-consumer-direct)
      ARENA_CONSUMER_DIRECT=false
      shift
      ;;
    --arena-numa-interleave)
      ARENA_NUMA_INTERLEAVE=true
      shift
      ;;
    --numa-bind)
      NUMA_BIND=true
      shift
      ;;
    --no-numa-bind)
      NO_NUMA_BIND=true
      NUMA_BIND=false
      shift
      ;;
    --decode-cuda-graph)
      DECODE_CUDA_GRAPH=true
      shift
      ;;
    --request-completion-timeout)
      REQUEST_COMPLETION_TIMEOUT_S="$2"
      shift 2
      ;;
    --get-retry-timeout)
      GET_RETRY_TIMEOUT_S="$2"
      shift 2
      ;;
    --get-retry-interval)
      GET_RETRY_INTERVAL_S="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
      shift 2
      ;;
    --licht)
      LICHT=true
      shift
      ;;
    --licht-v2)
      LICHT_V2=true
      shift
      ;;
    --licht-v3)
      LICHT_V3=true
      shift
      ;;
    --prefill-opt)
      PREFILL_OPT=true
      shift
      ;;
    --prefill-fcfs)
      PREFILL_FCFS=true
      shift
      ;;
    --prefill-opt-theta)
      PREFILL_OPT_THETA="$2"
      shift 2
      ;;
    --tool-predictor-dir)
      TOOL_PREDICTOR_DIR="$2"
      shift 2
      ;;
    --trace-replay-path)
      TRACE_REPLAY_PATH="$2"
      shift 2
      ;;
    --wait-timeout)
      WAIT_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --fail-on-wait-timeout)
      FAIL_ON_WAIT_TIMEOUT=true
      shift
      ;;
    --shutdown-grace-seconds)
      SHUTDOWN_GRACE_SECONDS="$2"
      shift 2
      ;;
    --client-pid-file)
      CLIENT_PID_FILE="$2"
      shift 2
      ;;
    --no-stop-client-on-exit)
      STOP_CLIENT_ON_EXIT=false
      shift
      ;;
    --client-stop-grace-seconds)
      CLIENT_STOP_GRACE_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

# LICHT-V3 = LICHT-V2 timeline + the three predictors.  The K_queue
# (stage1/2) and step-time predictors live in the decode-side
# ShadowScheduler, which only initialises when this env var is set.
# Without it, only the tool-time predictor (v3_predictions.jsonl) writes;
# v3_shadow_predictions.jsonl and v3_step_time.jsonl stay empty.  Export
# so the decode worker (a setsid child) inherits it.
if [[ "${LICHT_V3}" == "true" ]]; then
  export LICHT_V3_USE_SHADOW_SCHED=0
  #export LICHT_V3_USE_SHADOW_SCHED=1
  # Tool-time predictor bundle (bash→ML p50/p95, editor/submit→bucket 查表).
  # Without this the predictor runs in constant-fallback mode.
  if [[ -f "${TOOL_PREDICTOR_DIR}/bundle.json" ]]; then
    #export LICHT_V3_TOOL_PREDICTOR_DIR="${TOOL_PREDICTOR_DIR}"
    export LICHT_V3_TOOL_PREDICTOR_DIR=""
  else
    echo "WARN: tool predictor bundle not found at ${TOOL_PREDICTOR_DIR}/bundle.json"
    echo "      → tool-time prediction will use the constant fallback."
    echo "      Set a valid dir with --tool-predictor-dir, e.g. tool_call_time/runs/run_2902_v2"
  fi
  # Cross-round KV reuse defaults ON under LICHT-V3 (decode persists each
  # finished round's KV; next-round prefill loads the prefix instead of
  # recomputing).  Disable with --no-round-kv; override dir with
  # --round-kv-reuse-path.
  if [[ "${NO_ROUND_KV}" != "true" && -z "${ROUND_KV_REUSE_PATH}" ]]; then
    ROUND_KV_REUSE_PATH="${ROUND_KV_DEFAULT_PATH}"
  fi
  # NUMA binding also defaults ON under LICHT-V3 (faster pinned H2D for the
  # round-kv load).  Disable with --no-numa-bind.
  if [[ "${NO_NUMA_BIND}" != "true" ]]; then
    NUMA_BIND=true
  fi
fi

# Layer-wise pipelined round-kv load (opt-in; works with/without --licht-v3).
# Exported into the workers' env; the store reads LICHT_ROUND_KV_PIPELINE.
if [[ "${ROUND_KV_PIPELINE}" == "true" ]]; then
  export LICHT_ROUND_KV_PIPELINE=1
fi
if [[ "${ROUND_KV_PROFILE}" == "true" ]]; then
  export LICHT_ROUND_KV_PROFILE=1
fi
if [[ "${HBM_PROBE}" == "true" ]]; then
  export LICHT_HBM_PROBE=1
fi
if [[ "${ROUND_KV_ASYNC}" == "true" ]]; then
  export LICHT_ROUND_KV_ASYNC=1
fi
# Phase 1 (save-on-preempt): when scheduler preempts a running decode
# req, save its KV increment to arena before freeing; on re-admit it
# reloads from arena instead of doing a full recompute of prompt+outputs.
export LICHT_PHASE1_SAVE_ON_PREEMPT=1
# Phase 2 (PD path selector): at PD-handoff admission, if projected
# decode KV occupancy after admitting this req would exceed the
# threshold, route the handoff via CPU arena (ARENA_SINK RPC) instead
# of NCCL GPU->GPU.  Prefill D2H's the KV and releases its GPU blocks
# immediately; decode loads from arena once it has space.  Solves the
# RELEASE-timeout/force-free pathology (prefill GPU blocked when decode
# is full).  Default threshold 0.80.
export LICHT_PHASE2_ADMISSION_GATE=1
export LICHT_PHASE2_GATE_THRESHOLD=0.90
# SINK 自愈: ARENA_SINK 写入复核出真实 LCP < manifest 声称的 total (中间被 dedup
# 淘空一段, 单数字 manifest 表达不了中间洞) 时, 把 manifest+进度回退到真实 LCP,
# 下次从洞口重存填洞 (数据在 hold 的 GPU 块里) → 一次收敛, 取代无限 SINK-RETRY +
# GPU 块泄漏. 两端都 export (prefill 存得多, decode 也存少量在途 KV). 默认即开,
# 这里显式声明. 关掉: LICHT_SINK_HEAL=0 (回到旧的无限重试, 仅 A/B 用).
export LICHT_SINK_HEAL=1
if [[ "${ROUND_KV_RAW}" != "true" ]]; then
  export LICHT_ROUND_KV_RAW=0
fi
if [[ "${ROUND_KV_ARENA}" != "true" ]]; then
  export LICHT_ROUND_KV_ARENA=0
fi
if [[ -n "${ROUND_KV_ARENA_GB}" ]]; then
  export LICHT_ROUND_KV_ARENA_GB="${ROUND_KV_ARENA_GB}"
fi
if [[ "${ROUND_KV_SYNC_FIRST}" == "true" ]]; then
  export LICHT_ROUND_KV_SYNC_FIRST=1
fi
if [[ "${ROUND_KV_FUSED}" == "true" ]]; then
  export LICHT_ROUND_KV_FUSED=1
fi
if [[ "${ROUND_KV_LRU}" == "true" ]]; then
  export LICHT_ROUND_KV_LRU=1
fi
# --licht-v3 implies Stage-6 content-addr dedup ON by default (opt out with
# --no-arena-content-addr).  Both ends inherit the same export -> stays
# consistent (the hard requirement).
if [[ "${LICHT_V3}" == "true" && "${NO_ARENA_CONTENT_ADDR}" != "true" \
      && "${ARENA_CONTENT_ADDR}" != "true" ]]; then
  ARENA_CONTENT_ADDR=true
  echo "  (auto) --licht-v3 -> ARENA_CONTENT_ADDR=true"
fi
# Stage 6 内容寻址 dedup.  ★ 一次 export, prefill+decode 两个 setsid 子进程都
# 继承同一值 (两端一致是硬要求).  仅在 LRU arena 路径生效.
if [[ "${ARENA_CONTENT_ADDR}" == "true" ]]; then
  if [[ "${ROUND_KV_LRU}" != "true" ]]; then
    echo "WARN: --arena-content-addr 需要 --round-kv-lru (LRU arena), 当前 LRU 未开 → 内容寻址不会生效"
  fi
  export LICHT_ARENA_CONTENT_ADDR=1
fi
# decode 也 register arena 走直读 (默认开). 关掉 -> consumer 逐请求 staging.
if [[ "${ARENA_CONSUMER_DIRECT}" == "true" ]]; then
  export LICHT_ARENA_CONSUMER_DIRECT=1
else
  export LICHT_ARENA_CONSUMER_DIRECT=0
fi
# arena NUMA interleave (默认关, A/B 用). 两端同一 export 继承.
if [[ "${ARENA_NUMA_INTERLEAVE}" == "true" ]]; then
  export LICHT_ARENA_NUMA_INTERLEAVE=1
fi

if [[ ! -f "${PROXY_SCRIPT}" ]]; then
  echo "Proxy script not found: ${PROXY_SCRIPT}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}"
  exit 1
fi

# Accept a single file OR a directory (TraceStore loads every *.json in a dir,
# so several datasets can be replayed together).  Comma-separated lists are
# supported by the loaders but not validated here.
if [[ ! -e "${TRACE_REPLAY_PATH}" ]]; then
  echo "Trace replay path does not exist: ${TRACE_REPLAY_PATH}"
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found. Please activate your vLLM environment first."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 command not found"
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl command not found"
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq command not found"
  exit 1
fi

cleanup() {
  set +e
  trap - INT TERM EXIT

  stop_client_if_needed() {
    if [[ "${STOP_CLIENT_ON_EXIT}" != "true" ]]; then
      return 0
    fi

    if [[ ! -f "${CLIENT_PID_FILE}" ]]; then
      return 0
    fi

    local client_pid
    client_pid="$(head -n 1 "${CLIENT_PID_FILE}" | tr -cd '0-9')"
    if [[ -z "${client_pid}" ]]; then
      echo "Client PID file malformed: ${CLIENT_PID_FILE}"
      return 0
    fi

    if ! kill -0 "${client_pid}" 2>/dev/null; then
      echo "Client PID ${client_pid} is not running; removing stale pid file"
      rm -f "${CLIENT_PID_FILE}" 2>/dev/null || true
      return 0
    fi

    local cmdline=""
    if [[ -r "/proc/${client_pid}/cmdline" ]]; then
      cmdline="$(tr '\0' ' ' < "/proc/${client_pid}/cmdline" 2>/dev/null || true)"
    fi
    if [[ "${cmdline}" != *"multiturn_trace_client.py"* ]]; then
      echo "PID ${client_pid} from ${CLIENT_PID_FILE} is not multiturn_trace_client.py; skip client stop"
      return 0
    fi

    echo "Stopping client process ${client_pid} (grace=${CLIENT_STOP_GRACE_SECONDS}s)..."
    kill -INT "${client_pid}" 2>/dev/null || true

    local deadline=$((SECONDS + CLIENT_STOP_GRACE_SECONDS))
    while kill -0 "${client_pid}" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 1
    done

    if kill -0 "${client_pid}" 2>/dev/null; then
      kill -TERM "${client_pid}" 2>/dev/null || true
      deadline=$((SECONDS + CLIENT_STOP_GRACE_SECONDS))
      while kill -0 "${client_pid}" 2>/dev/null && (( SECONDS < deadline )); do
        sleep 1
      done
    fi

    if kill -0 "${client_pid}" 2>/dev/null; then
      kill -KILL "${client_pid}" 2>/dev/null || true
    fi

    rm -f "${CLIENT_PID_FILE}" 2>/dev/null || true
  }

  all_timestamps_ready() {
    MISSING_TIMESTAMP_FILES=()
    INVALID_MONITORING_TIMESTAMP_FILES=()
    PENDING_MONITORING_TMP_FILES=()
    if [[ ${#EXPECTED_TIMESTAMP_FILES[@]} -eq 0 ]]; then
      return 0
    fi

    local f
    for f in "${EXPECTED_TIMESTAMP_FILES[@]}"; do
      if [[ ! -s "${f}" ]]; then
        MISSING_TIMESTAMP_FILES+=("${f}")
        continue
      fi

      if [[ "$(basename "${f}")" == "monitoring_timestamps" ]]; then
        if ! jq -e . "${f}" >/dev/null 2>&1; then
          INVALID_MONITORING_TIMESTAMP_FILES+=("${f}")
        fi
        if compgen -G "${f}.tmp.*" >/dev/null; then
          PENDING_MONITORING_TMP_FILES+=("${f}")
        fi
      fi
    done

    [[ ${#MISSING_TIMESTAMP_FILES[@]} -eq 0 && \
       ${#INVALID_MONITORING_TIMESTAMP_FILES[@]} -eq 0 && \
       ${#PENDING_MONITORING_TMP_FILES[@]} -eq 0 ]]
  }

  print_timestamp_integrity_issues() {
    local f
    for f in "${MISSING_TIMESTAMP_FILES[@]}"; do
      echo "  - missing/empty: ${f}"
    done
    for f in "${INVALID_MONITORING_TIMESTAMP_FILES[@]}"; do
      echo "  - invalid JSON (jq failed): ${f}"
    done
    for f in "${PENDING_MONITORING_TMP_FILES[@]}"; do
      echo "  - tmp still present: ${f}.tmp.*"
    done
  }

  wait_for_groups_and_timestamps() {
    local timeout="$1"
    local deadline=$((SECONDS + timeout))
    while (( SECONDS < deadline )); do
      local alive=0
      for pid in "${PIDS[@]}"; do
        if kill -0 -- "-${pid}" 2>/dev/null; then
          alive=1
          break
        fi
      done
      if (( alive == 0 )); then
        if all_timestamps_ready; then
          return 0
        fi
      fi
      sleep 1
    done
    return 1
  }

  stop_client_if_needed

  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "Stopping ${#PIDS[@]} processes (grace=${SHUTDOWN_GRACE_SECONDS}s)..."

    # 1) First attempt: SIGINT (lets vLLM run shutdown hooks and dump files).
    for pid in "${PIDS[@]}"; do
      # Each service is started with setsid, so its PID is also its PGID.
      kill -INT -- "-${pid}" 2>/dev/null || true
    done
    wait_for_groups_and_timestamps "${SHUTDOWN_GRACE_SECONDS}" || true

    # 2) Second attempt: SIGTERM for any survivors.
    for pid in "${PIDS[@]}"; do
      if kill -0 -- "-${pid}" 2>/dev/null; then
        kill -TERM -- "-${pid}" 2>/dev/null || true
      fi
    done
    wait_for_groups_and_timestamps "${SHUTDOWN_GRACE_SECONDS}" || true

    if ! all_timestamps_ready; then
      echo "Warning: strong timestamp integrity check not yet satisfied:"
      print_timestamp_integrity_issues
      echo "Waiting extra ${SHUTDOWN_GRACE_SECONDS}s for graceful finalization..."
      wait_for_groups_and_timestamps "${SHUTDOWN_GRACE_SECONDS}" || true
    fi

    # 3) Last resort: SIGKILL (only after strong integrity checks pass).
    if all_timestamps_ready; then
      for pid in "${PIDS[@]}"; do
        if kill -0 -- "-${pid}" 2>/dev/null; then
          kill -KILL -- "-${pid}" 2>/dev/null || true
        fi
      done
    else
      echo "Skipping SIGKILL to avoid truncating monitoring dumps."
      print_timestamp_integrity_issues
    fi
  fi
  wait 2>/dev/null || true
}

# Echo a `numactl --cpunodebind=N --membind=N` prefix that binds a worker to
# the NUMA node local to GPU $1 (derived from sysfs), or nothing if disabled /
# unavailable / node unknown.  numactl needs no privileges.
numa_wrap_for_gpu() {
  local gpu="$1" busid node
  [[ "${NUMA_BIND}" == "true" ]] || { echo ""; return; }
  command -v numactl >/dev/null 2>&1 || { echo ""; return; }
  busid="$(nvidia-smi -i "${gpu}" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null \
            | tr '[:upper:]' '[:lower:]' | sed 's/^0000//')"
  [[ -n "${busid}" ]] || { echo ""; return; }
  node="$(cat "/sys/bus/pci/devices/${busid}/numa_node" 2>/dev/null)"
  if [[ "${node}" =~ ^[0-9]+$ ]] && (( node >= 0 )); then
    echo "numactl --cpunodebind=${node} --membind=${node}"
  else
    echo ""
  fi
}

wait_for_http_ready() {
  local port="$1"
  local timeout="$2"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "HTTP ready on port ${port}"
      return 0
    fi
    local now
    now="$(date +%s)"
    if (( now - start_ts >= timeout )); then
      echo "Timeout waiting for HTTP endpoint on port ${port}"
      return 1
    fi
    sleep 1
  done
}

IFS=',' read -r -a PREFILL_GPU_ARRAY <<< "${PREFILL_GPUS}"
IFS=',' read -r -a DECODE_GPU_ARRAY <<< "${DECODE_GPUS}"

if [[ ${#PREFILL_GPU_ARRAY[@]} -eq 0 || -z "${PREFILL_GPU_ARRAY[0]}" ]]; then
  echo "At least one prefill GPU is required"
  exit 1
fi

if [[ ${#DECODE_GPU_ARRAY[@]} -eq 0 || -z "${DECODE_GPU_ARRAY[0]}" ]]; then
  echo "At least one decode GPU is required"
  exit 1
fi

echo "Configuration:"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  PREFILL_GPUS=${PREFILL_GPUS}"
echo "  DECODE_GPUS=${DECODE_GPUS}"
echo "  KV_SEND_TYPE=${KV_SEND_TYPE}"
echo "  ROUND_KV_REUSE_PATH=${ROUND_KV_REUSE_PATH:-(disabled)}"
echo "  ROUND_KV_PIPELINE=${ROUND_KV_PIPELINE} (layer-wise load overlap)"
echo "  ROUND_KV_PROFILE=${ROUND_KV_PROFILE} (diagnostic load timing)"
echo "  HBM_PROBE=${HBM_PROBE} (diagnostic HBM headroom probe)"
echo "  ROUND_KV_ASYNC=${ROUND_KV_ASYNC} (async load, engine non-blocking)"
echo "  ROUND_KV_RAW=${ROUND_KV_RAW} (contiguous .bin, no strided read)"
echo "  ROUND_KV_LRU=${ROUND_KV_LRU} (stage2 slot-paged LRU arena)"
echo "  ARENA_CONTENT_ADDR=${ARENA_CONTENT_ADDR} (stage6 跨 job dedup: hash表+refcnt)"
echo "  ARENA_CONSUMER_DIRECT=${ARENA_CONSUMER_DIRECT} (decode 也 register arena 走直读, 防长前缀 staging OOM)"
echo "  ARENA_NUMA_INTERLEAVE=${ARENA_NUMA_INTERLEAVE} (arena 跨 NUMA 摊带宽, A/B 用)"
echo "  REQUEST_COMPLETION_TIMEOUT_S=${REQUEST_COMPLETION_TIMEOUT_S}"
echo "  GET_RETRY_TIMEOUT_S=${GET_RETRY_TIMEOUT_S}"
echo "  GET_RETRY_INTERVAL_S=${GET_RETRY_INTERVAL_S}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN} (0 means auto/model default)"
echo "  MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS} (0 means auto)"
echo "  LICHT=${LICHT}"
echo "  LICHT_V2=${LICHT_V2}"
echo "  LICHT_V3=${LICHT_V3}"
echo "  TOOL_PREDICTOR_DIR=${TOOL_PREDICTOR_DIR}"
echo "  TRACE_REPLAY_PATH=${TRACE_REPLAY_PATH}"
echo "  FAIL_ON_WAIT_TIMEOUT=${FAIL_ON_WAIT_TIMEOUT}"
echo "  SHUTDOWN_GRACE_SECONDS=${SHUTDOWN_GRACE_SECONDS}"
echo "  CLIENT_PID_FILE=${CLIENT_PID_FILE}"
echo "  STOP_CLIENT_ON_EXIT=${STOP_CLIENT_ON_EXIT}"
echo "  CLIENT_STOP_GRACE_SECONDS=${CLIENT_STOP_GRACE_SECONDS}"
echo "  PROXY_DISCOVERY=tcp://${PROXY_DISCOVERY_HOST}:${PROXY_DISCOVERY_PORT}"
echo "  PROXY_API=http://${PROXY_API_HOST}:${PROXY_API_PORT}"
echo ""

trap cleanup INT TERM EXIT

cd "${SCRIPT_DIR}"

rm -rf "${SCRIPT_DIR}/continuum_exp"/prefill_* "${SCRIPT_DIR}/continuum_exp"/decode_* 2>/dev/null || true

# Start each run with a clean cross-round KV store (avoid stale files from
# a prior run lingering / filling the medium).
if [[ -n "${ROUND_KV_REUSE_PATH}" ]]; then
  rm -rf "${ROUND_KV_REUSE_PATH}" 2>/dev/null || true
  mkdir -p "${ROUND_KV_REUSE_PATH}" 2>/dev/null || true
fi

mkdir -p "${SCRIPT_DIR}/continuum_exp"
EXPECTED_TIMESTAMP_FILES=()

echo "Starting proxy..."
setsid python3 "${PROXY_SCRIPT}" \
  --host "${PROXY_API_HOST}" \
  --api-port "${PROXY_API_PORT}" \
  --discovery-host "${PROXY_DISCOVERY_HOST}" \
  --discovery-port "${PROXY_DISCOVERY_PORT}" \
  > proxy_prod.log 2>&1 &
PIDS+=("$!")

echo "Starting prefill workers..."
PREFILL_PORTS=()
for i in "${!PREFILL_GPU_ARRAY[@]}"; do
  gpu_id="${PREFILL_GPU_ARRAY[$i]}"
  http_port=$((PREFILL_HTTP_PORT_BASE + i * 2))
  kv_port=$((PREFILL_KV_PORT_BASE + i * 2))
  PREFILL_PORTS+=("${http_port}")

  echo "  prefill[$i]: gpu=${gpu_id}, http_port=${http_port}, kv_port=${kv_port}"
  PREFILL_EXTRA_ARGS=()
  if (( MAX_MODEL_LEN > 0 )); then
    PREFILL_EXTRA_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
  fi
  if (( MAX_NUM_BATCHED_TOKENS > 0 )); then
    PREFILL_EXTRA_ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
  fi
  if [[ "${LICHT}" == "true" ]]; then
    PREFILL_EXTRA_ARGS+=(--licht)
  fi
  if [[ "${LICHT_V2}" == "true" ]]; then
    PREFILL_EXTRA_ARGS+=(--licht-v2)
  fi
  if [[ "${LICHT_V3}" == "true" ]]; then
    PREFILL_EXTRA_ARGS+=(--licht-v3)
  fi
  # LICHT-V3: prefill PUBLISHes StepEvents so the decode shadow can record
  # K_queue + step-time predictions.  Export (NOT inline via a variable —
  # bash only treats LITERAL `VAR=val cmd` as an assignment) so the
  # backgrounded `setsid vllm serve` child inherits it.  Per-worker port.
  if [[ "${LICHT_V3}" == "true" ]]; then
    export LICHT_V3_STEP_EVENT_PUB_ADDR="tcp://0.0.0.0:$((STEP_EVENT_PORT_BASE + i * 2))"
  fi

  prefill_output_dir="${SCRIPT_DIR}/continuum_exp/prefill_${http_port}"
  rm -rf "${prefill_output_dir}"
  mkdir -p "${prefill_output_dir}"
  EXPECTED_TIMESTAMP_FILES+=(
    "${prefill_output_dir}/scheduler_timestamps"
    "${prefill_output_dir}/monitoring_timestamps"
  )

  NUMA_WRAP="$(numa_wrap_for_gpu "${gpu_id}")"
  [[ -n "${NUMA_WRAP}" ]] && echo "  prefill[$i]: numa-bind gpu ${gpu_id} via '${NUMA_WRAP}'"

  # --prefill-opt: optimal prefill stack (longcap_fcfs + theta + dynamic_chunk).
  # env-gated; needs the LICHT-V2 timeline (already on under --licht-v2/--licht-v3).
  # export (NOT inline) so the backgrounded setsid vllm serve child inherits it.
  if [[ "${PREFILL_OPT}" == "true" ]]; then
    _brb_dir="${SCRIPT_DIR}/../../../dynamic_chunk/brb_cache"
    # one-time fingerprint-cached beta_r/b calibration on this gpu (cache hit
    # is instant; same-model gpus reuse the same json).
    # pass the SERVED config so the fingerprint + calibration match the engine.
    # LICHT_CAL_TP must match --tensor-parallel-size below (here: 1, one GPU per
    # prefill worker). For TP>1 pass a comma list as the gpu arg + set TP.
    _brb_file="$(LICHT_CAL_MODEL="${MODEL_PATH}" LICHT_CAL_DTYPE="${DTYPE}" \
                 LICHT_CAL_MAXLEN="${MAX_MODEL_LEN}" \
                 LICHT_CAL_GMU="${PREFILL_GPU_MEMORY_UTILIZATION}" \
                 LICHT_CAL_TP=1 \
                 python "${SCRIPT_DIR}/../../../dynamic_chunk/calibrate_brb.py" \
                  "${gpu_id}" "${_brb_dir}" 2>/dev/null \
                  | grep -oE 'BRB_RESULT_PATH=[^ ]+' | tail -1 | cut -d= -f2)"
    export LICHT_SCHED_SCHEME=longcap_fcfs
    export LICHT_LONGCAP_ORDER=short
    # Prefix-hit-aware scheduling: predict each waiting request's real cross-tier
    # (HBM+arena) prefix hit BEFORE scoring, so a returning round with a big
    # cached prefix (big prompt, small REAL remaining) is classified short ->
    # short lane -> admitted fast (not starved in the long lane).  Also makes
    # FCFS-break "close the long lane" (continue, shorts backfill) instead of a
    # hard break, and feeds dyn_chunk the real D/C.  Auto-on with longcap.
    export LICHT_SCHED_HIT_PRED=1
    export LICHT_LONG_THETA="${PREFILL_OPT_THETA}"   # presence enables the long throttle
    export LICHT_SHORT_RESERVE="${PREFILL_OPT_SHORT_RESERVE}"  # keep this frac KV free
                                                     # for shorts; longs use the rest
                                                     # (only throttled near-full, not by
                                                     # a fixed footprint cap)
    export LICHT_LONG_C="${PREFILL_OPT_LONGC}"
    export LICHT_LONG_RESV=1
    export LICHT_LONGCAP_FCFS_BREAK=1
    # θ 容量帽松绑: 本步若【所有短请求都进去了】(没有短请求因放不下被挡),
    # θ 帽就没有保护对象 → 对长请求松开 θ, 用 future-free 物理检查把空着的 KV
    # 塞满长请求, 直到某个长请求装不下为止. 短请求一旦有被挡下的, θ 立即恢复.
    # 解决"长请求多、短请求已塞满、但 KV 还空着被 θ 挡住"的浪费. 配合
    # LICHT_LONGCAP_FOOTPRINT=1(footprint 帽)使用.
    export LICHT_LONG_THETA_RELAX=1
    # dynamic_chunk mode F: smooth long/short via lambda=smoothstep((C-Clow)/(Chigh-Clow)).
    # F adds the "big request also waits for short requests each round" penalty
    # (shared by N_long), curing mode E's over-chunking (S* floored ~256 under
    # congestion). SHORT_SET=all => drag penalty W_soft counts running shorts AND
    # waiting requests' extra wait => more conservative chunks, best p50/mean under
    # load (validated jps=10: p50 37.5->36.5, mean 54.4->53.7, all tail metrics down).
    # timeline (R_at/B_at) uses this real per-step chunk so future_free matches reality.
    export LICHT_DYN_CHUNK=F
    export LICHT_DYN_SHORT_SET=all
    export LICHT_DYN_CLOW="${PREFILL_OPT_CLOW}"
    export LICHT_DYN_CHIGH="${PREFILL_OPT_CHIGH}"
    [[ -n "${_brb_file}" ]] && export LICHT_DYN_BRB_FILE="${_brb_file}"
    echo "  prefill[$i]: PREFILL_OPT on (shorts-first longcap_fcfs + theta=${PREFILL_OPT_THETA}"\
         "+ C=${PREFILL_OPT_LONGC} + reservation + FCFS-break + dynamic_chunk[F/all] band=${PREFILL_OPT_CLOW}-${PREFILL_OPT_CHIGH}"\
         "+ short_reserve=${PREFILL_OPT_SHORT_RESERVE}; brb=${_brb_file:-default216})"
  elif [[ "${PREFILL_FCFS}" == "true" ]]; then
    # --prefill-fcfs: DIAGNOSTIC baseline. Keep LICHT-V3 (round-kv arena reuse)
    # intact but strip every prefill-opt scheduling trick:
    #   * priority = pure FCFS by arrival (LICHT_SCHED_SCHEME=fcfs) INSTEAD of the
    #     round-based licht score -> rounds of one conversation are NOT reordered
    #     apart, so a returning round's prefix is less likely evicted before it runs.
    #   * NO longcap (no theta / reserve / order / FCFS-break).
    #   * NO dynamic_chunk -> fixed chunk = vLLM native long_prefill_token_threshold
    #     (= int(0.04*max_model_len) = 5242 for 131072 ctx); LICHT_DYN_CHUNK unset.
    # Purpose: test whether the low prefix-cache hit rate is caused by scheduling
    # reorder (longcap/score separating a conversation's rounds in time) vs the
    # arena itself. If hit rate recovers -> scheduling; if still low -> arena.
    export LICHT_SCHED_SCHEME=fcfs
    echo "  prefill[$i]: PREFILL_FCFS on (LICHT-V3 + pure-FCFS priority + fixed chunk=native"\
         "long_prefill_token_threshold; NO longcap, NO dynamic_chunk)"
  fi
  CUDA_VISIBLE_DEVICES="${gpu_id}" VLLM_USE_V1=1 VLLM_TRACE_REPLAY_PATH="${TRACE_REPLAY_PATH}" RUN_OUTPUT_DIR="${prefill_output_dir}" CONTINUUM_INSTANCE_TAG="prefill_${http_port}" setsid ${NUMA_WRAP} vllm serve "${MODEL_PATH}" \
    --enforce-eager \
    --host 0.0.0.0 \
    --port "${http_port}" \
    --tensor-parallel-size 1 \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --trust-remote-code \
    --gpu-memory-utilization "${PREFILL_GPU_MEMORY_UTILIZATION}" \
    "${PREFILL_EXTRA_ARGS[@]}" \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":\"${kv_port}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${PROXY_IP_FOR_WORKERS}\",\"proxy_port\":\"${PROXY_DISCOVERY_PORT}\",\"http_port\":\"${http_port}\",\"send_type\":\"${KV_SEND_TYPE}\",\"nccl_num_channels\":\"16\",\"request_completion_timeout_s\":\"${REQUEST_COMPLETION_TIMEOUT_S}\",\"get_retry_timeout_s\":\"${GET_RETRY_TIMEOUT_S}\",\"get_retry_interval_s\":\"${GET_RETRY_INTERVAL_S}\",\"round_kv_reuse_path\":\"${ROUND_KV_REUSE_PATH}\"}}" \
    > "prefill_prod_$((i + 1)).log" 2>&1 &
  PIDS+=("$!")
done

# prefill-opt env is prefill-only: clear before launching decode workers so
# decode scheduling is untouched.
if [[ "${PREFILL_OPT}" == "true" ]]; then
  unset LICHT_SCHED_SCHEME LICHT_LONGCAP_ORDER LICHT_LONG_THETA LICHT_SHORT_RESERVE LICHT_LONG_C LICHT_LONG_RESV LICHT_LONGCAP_FCFS_BREAK LICHT_SCHED_HIT_PRED LICHT_LONG_THETA_RELAX LICHT_DYN_CHUNK LICHT_DYN_SHORT_SET LICHT_DYN_CLOW LICHT_DYN_CHIGH LICHT_DYN_BRB_FILE
fi

echo "Starting decode workers..."
DECODE_PORTS=()
for i in "${!DECODE_GPU_ARRAY[@]}"; do
  gpu_id="${DECODE_GPU_ARRAY[$i]}"
  http_port=$((DECODE_HTTP_PORT_BASE + i * 2))
  kv_port=$((DECODE_KV_PORT_BASE + i * 2))
  DECODE_PORTS+=("${http_port}")

  echo "  decode[$i]: gpu=${gpu_id}, http_port=${http_port}, kv_port=${kv_port}"
  DECODE_EXTRA_ARGS=()
  # Decode runs eager unless --decode-cuda-graph is passed (prefill is always
  # eager, hardcoded below).
  if [[ "${DECODE_CUDA_GRAPH}" != "true" ]]; then
    DECODE_EXTRA_ARGS+=(--enforce-eager)
  else
    [[ "$i" == "0" ]] && echo "  decode: CUDA graph ENABLED (eager off) — watch startup for 'Capturing CUDA graphs' + check trace_replay match"
  fi
  if (( MAX_MODEL_LEN > 0 )); then
    DECODE_EXTRA_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
  fi
  if (( MAX_NUM_SEQS > 0 )); then
    DECODE_EXTRA_ARGS+=(--max-num-batched-tokens "${MAX_NUM_SEQS}")
  fi
  if [[ "${LICHT}" == "true" ]]; then
    DECODE_EXTRA_ARGS+=(--licht)
  fi
  if [[ "${LICHT_V2}" == "true" ]]; then
    DECODE_EXTRA_ARGS+=(--licht-v2)
  fi
  if [[ "${LICHT_V3}" == "true" ]]; then
    DECODE_EXTRA_ARGS+=(--licht-v3)
  fi
  # LICHT-V3: decode shadow SUBSCRIBEs to the matching prefill's StepEvent
  # channel (round-robin if there are more decodes than prefills).  Export
  # so the backgrounded `setsid vllm serve` child inherits it.
  if [[ "${LICHT_V3}" == "true" ]]; then
    pf_idx=$(( i % ${#PREFILL_GPU_ARRAY[@]} ))
    export LICHT_V3_STEP_EVENT_SUB_ADDR="tcp://${PROXY_IP_FOR_WORKERS}:$((STEP_EVENT_PORT_BASE + pf_idx * 2))"
  fi

  decode_output_dir="${SCRIPT_DIR}/continuum_exp/decode_${http_port}"
  rm -rf "${decode_output_dir}"
  mkdir -p "${decode_output_dir}"
  EXPECTED_TIMESTAMP_FILES+=(
    "${decode_output_dir}/scheduler_timestamps"
    "${decode_output_dir}/monitoring_timestamps"
  )

  NUMA_WRAP="$(numa_wrap_for_gpu "${gpu_id}")"
  [[ -n "${NUMA_WRAP}" ]] && echo "  decode[$i]: numa-bind gpu ${gpu_id} via '${NUMA_WRAP}'"
  CUDA_VISIBLE_DEVICES="${gpu_id}" VLLM_USE_V1=1 VLLM_TRACE_REPLAY_PATH="${TRACE_REPLAY_PATH}" RUN_OUTPUT_DIR="${decode_output_dir}" CONTINUUM_INSTANCE_TAG="decode_${http_port}" setsid ${NUMA_WRAP} vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${http_port}" \
    --tensor-parallel-size 1 \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --trust-remote-code \
    --gpu-memory-utilization "${DECODE_GPU_MEMORY_UTILIZATION}" \
    "${DECODE_EXTRA_ARGS[@]}" \
    --enable-chunked-prefill \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_port\":\"${kv_port}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${PROXY_IP_FOR_WORKERS}\",\"proxy_port\":\"${PROXY_DISCOVERY_PORT}\",\"http_port\":\"${http_port}\",\"send_type\":\"${KV_SEND_TYPE}\",\"nccl_num_channels\":\"16\",\"request_completion_timeout_s\":\"${REQUEST_COMPLETION_TIMEOUT_S}\",\"get_retry_timeout_s\":\"${GET_RETRY_TIMEOUT_S}\",\"get_retry_interval_s\":\"${GET_RETRY_INTERVAL_S}\",\"round_kv_reuse_path\":\"${ROUND_KV_REUSE_PATH}\"}}" \
    > "decode_prod_$((i + 1)).log" 2>&1 &
  PIDS+=("$!")
done

READY_TIMEOUT_PORTS=()

echo "Waiting prefill workers..."
for port in "${PREFILL_PORTS[@]}"; do
  if ! wait_for_http_ready "${port}" "${WAIT_TIMEOUT_SECONDS}"; then
    READY_TIMEOUT_PORTS+=("prefill:${port}")
  fi
done

echo "Waiting decode workers..."
for port in "${DECODE_PORTS[@]}"; do
  if ! wait_for_http_ready "${port}" "${WAIT_TIMEOUT_SECONDS}"; then
    READY_TIMEOUT_PORTS+=("decode:${port}")
  fi
done

echo ""
if [[ ${#READY_TIMEOUT_PORTS[@]} -gt 0 ]]; then
  echo "Warning: readiness check timed out for the following endpoints:"
  for item in "${READY_TIMEOUT_PORTS[@]}"; do
    echo "  - ${item}"
  done

  if [[ "${FAIL_ON_WAIT_TIMEOUT}" == "true" ]]; then
    echo "Configured with --fail-on-wait-timeout, exiting launcher."
    exit 1
  fi

  echo "Continuing to run existing processes; launcher will wait until Ctrl+C."
else
  echo "All services are ready."
fi
echo "Proxy endpoint: http://127.0.0.1:${PROXY_API_PORT}"
echo "Per-instance timestamps directory: ${SCRIPT_DIR}/continuum_exp"
echo ""
echo "Example test request:"
echo "curl http://127.0.0.1:${PROXY_API_PORT}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{\"model\":\"${MODEL_PATH}\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":64,\"stream\":false}'"
echo ""
echo "Press Ctrl+C to stop all services."

wait
