#!/usr/bin/env bash
# 探针 B (2026-07-04): GPU 内存周期采样, 回答 "OOM 时刻是谁占的卡".
# 用法: ./gpu_mem_sampler.sh [间隔秒=30] [输出=continuum_exp/gpu_mem.log] &
# 跑批前后台启动, 结束 kill 即可. 每行: 时间戳 + 每 GPU 总/已用 + 每进程占用.
INTERVAL="${1:-30}"
OUT="${2:-continuum_exp/gpu_mem.log}"
mkdir -p "$(dirname "$OUT")"
echo "gpu_mem_sampler: interval=${INTERVAL}s -> $OUT" >&2
while true; do
    {
        echo "=== $(date '+%F %T') ==="
        nvidia-smi --query-gpu=index,memory.total,memory.used \
                   --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
                   --format=csv,noheader
    } >> "$OUT" 2>&1
    sleep "$INTERVAL"
done
