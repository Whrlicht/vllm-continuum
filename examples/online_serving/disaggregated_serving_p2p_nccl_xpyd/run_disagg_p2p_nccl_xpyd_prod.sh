#!/usr/bin/env bash

set -Eeuo pipefail

# Production launcher for P2P NCCL XpYd disaggregated serving.
# Example:
#   ./run_disagg_p2p_nccl_xpyd_prod.sh --prefill-gpus 0 --decode-gpus 1,2

MODEL_PATH="/data/huggingface/models--meta-llama--Llama-3.1-8B-Instruct"
PREFILL_GPUS="5"
DECODE_GPUS="6"

PROXY_DISCOVERY_HOST="0.0.0.0"
PROXY_DISCOVERY_PORT=30001
PROXY_API_HOST="0.0.0.0"
PROXY_API_PORT=10234

PROXY_IP_FOR_WORKERS="127.0.0.1"
PREFILL_HTTP_PORT_BASE=20003
DECODE_HTTP_PORT_BASE=20005
PREFILL_KV_PORT_BASE=21001
DECODE_KV_PORT_BASE=22001

PREFILL_GPU_MEMORY_UTILIZATION=0.95
DECODE_GPU_MEMORY_UTILIZATION=0.95

# DistServe-style direct block migration mode.
# Decode actively pops bridge metadata and migrates blocks from prefill.
KV_SEND_TYPE="BLOCK_MIGRATE"
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

WAIT_TIMEOUT_SECONDS=1200
SHUTDOWN_GRACE_SECONDS=20
CLIENT_STOP_GRACE_SECONDS=20
FAIL_ON_WAIT_TIMEOUT=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRACE_REPLAY_PATH="${REPO_ROOT}/trace_data/swe_bench_sample_100_with_timings.json"
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

if [[ ! -f "${PROXY_SCRIPT}" ]]; then
  echo "Proxy script not found: ${PROXY_SCRIPT}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${TRACE_REPLAY_PATH}" ]]; then
  echo "Trace replay file does not exist: ${TRACE_REPLAY_PATH}"
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
echo "  REQUEST_COMPLETION_TIMEOUT_S=${REQUEST_COMPLETION_TIMEOUT_S}"
echo "  GET_RETRY_TIMEOUT_S=${GET_RETRY_TIMEOUT_S}"
echo "  GET_RETRY_INTERVAL_S=${GET_RETRY_INTERVAL_S}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN} (0 means auto/model default)"
echo "  MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS} (0 means auto)"
echo "  LICHT=${LICHT}"
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

  prefill_output_dir="${SCRIPT_DIR}/continuum_exp/prefill_${http_port}"
  rm -rf "${prefill_output_dir}"
  mkdir -p "${prefill_output_dir}"
  EXPECTED_TIMESTAMP_FILES+=(
    "${prefill_output_dir}/scheduler_timestamps"
    "${prefill_output_dir}/monitoring_timestamps"
  )

  CUDA_VISIBLE_DEVICES="${gpu_id}" VLLM_USE_V1=1 VLLM_TRACE_REPLAY_PATH="${TRACE_REPLAY_PATH}" RUN_OUTPUT_DIR="${prefill_output_dir}" CONTINUUM_INSTANCE_TAG="prefill_${http_port}" setsid vllm serve "${MODEL_PATH}" \
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
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":\"${kv_port}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${PROXY_IP_FOR_WORKERS}\",\"proxy_port\":\"${PROXY_DISCOVERY_PORT}\",\"http_port\":\"${http_port}\",\"send_type\":\"${KV_SEND_TYPE}\",\"nccl_num_channels\":\"16\",\"request_completion_timeout_s\":\"${REQUEST_COMPLETION_TIMEOUT_S}\",\"get_retry_timeout_s\":\"${GET_RETRY_TIMEOUT_S}\",\"get_retry_interval_s\":\"${GET_RETRY_INTERVAL_S}\"}}" \
    > "prefill_prod_$((i + 1)).log" 2>&1 &
  PIDS+=("$!")
done

echo "Starting decode workers..."
DECODE_PORTS=()
for i in "${!DECODE_GPU_ARRAY[@]}"; do
  gpu_id="${DECODE_GPU_ARRAY[$i]}"
  http_port=$((DECODE_HTTP_PORT_BASE + i * 2))
  kv_port=$((DECODE_KV_PORT_BASE + i * 2))
  DECODE_PORTS+=("${http_port}")

  echo "  decode[$i]: gpu=${gpu_id}, http_port=${http_port}, kv_port=${kv_port}"
  DECODE_EXTRA_ARGS=()
  if (( MAX_MODEL_LEN > 0 )); then
    DECODE_EXTRA_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
  fi
  if (( MAX_NUM_SEQS > 0 )); then
    DECODE_EXTRA_ARGS+=(--max-num-batched-tokens "${MAX_NUM_SEQS}")
  fi
  if [[ "${LICHT}" == "true" ]]; then
    DECODE_EXTRA_ARGS+=(--licht)
  fi

  decode_output_dir="${SCRIPT_DIR}/continuum_exp/decode_${http_port}"
  rm -rf "${decode_output_dir}"
  mkdir -p "${decode_output_dir}"
  EXPECTED_TIMESTAMP_FILES+=(
    "${decode_output_dir}/scheduler_timestamps"
    "${decode_output_dir}/monitoring_timestamps"
  )

  CUDA_VISIBLE_DEVICES="${gpu_id}" VLLM_USE_V1=1 VLLM_TRACE_REPLAY_PATH="${TRACE_REPLAY_PATH}" RUN_OUTPUT_DIR="${decode_output_dir}" CONTINUUM_INSTANCE_TAG="decode_${http_port}" setsid vllm serve "${MODEL_PATH}" \
    --enforce-eager \
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
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_port\":\"${kv_port}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${PROXY_IP_FOR_WORKERS}\",\"proxy_port\":\"${PROXY_DISCOVERY_PORT}\",\"http_port\":\"${http_port}\",\"send_type\":\"${KV_SEND_TYPE}\",\"nccl_num_channels\":\"16\",\"request_completion_timeout_s\":\"${REQUEST_COMPLETION_TIMEOUT_S}\",\"get_retry_timeout_s\":\"${GET_RETRY_TIMEOUT_S}\",\"get_retry_interval_s\":\"${GET_RETRY_INTERVAL_S}\"}}" \
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
