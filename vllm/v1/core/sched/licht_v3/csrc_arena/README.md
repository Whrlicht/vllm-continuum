# LICHT Round-KV Arena 跨进程原子原语扩展

独立的 C++ 扩展, 与 `../csrc/fused_scatter` 解耦.

## 提供能力

- **跨进程 mutex**: `pthread_mutex_t` with `PTHREAD_PROCESS_SHARED + PTHREAD_MUTEX_ROBUST`
  - 包住 alloc + evict + memcpy + commit 临界区
  - 持锁进程崩溃时可 recover
- **Slot 级 reader pin**: lock-free CAS (`try_pin` / `unpin`)
  - 高 16 位 atomic 计数, 容量 65535 并发
- **Slot 级 gen 计数器**: 48 位
  - reader/writer 跨进程一致性的核心机制
- **通用 64-bit 原子原语**: `load` / `store` / `fetch_add`

## 不提供的能力

- mmap / hdr layout 计算 → 在 Python 端做
- CUDA kernel → 这是给 arena_atomic 用的, 不是 GPU 工作
- xxhash → vendored 但 Stage 6 才用

## 构建

```bash
cd vllm/v1/core/sched/licht_v3/csrc_arena
pip install .
```

构建产物: `licht_arena_atomic.cpython-*-x86_64-linux-gnu.so`

## 测试

```bash
cd /data/whr/vllm-continuum
pytest vllm/v1/core/sched/licht_v3/tests/ -v
```

## 设计文档

`roadmap/pinned_arena_lru_redesign.md`
