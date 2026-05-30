# LICHT Round-KV Arena 测试

## 运行

```bash
cd /data/whr/vllm-continuum
pytest vllm/v1/core/sched/licht_v3/tests/ -v
```

## 阶段测试覆盖

- **Stage 0** (当前): `test_extension_importable` 占位
- **Stage 1**: 原子原语正确性 (mutex / pin / gen)
- **Stage 2**: slot-paged 分配器 + LRU 行为
- **Stage 3**: post-load fallback
- **Stage 4**: Phase 1/2 路径 + stale-pin 清理
- **Stage 6**: 内容寻址 + refcnt

## 测试设计原则

1. 单元测试 + 集成测试分开
2. 跨进程测试用 `multiprocessing.fork`, 在 fixture 里准备共享 mmap
3. ROBUST mutex 测试用 `os.kill(child, SIGKILL)` 模拟崩溃
4. 单元测试不依赖真实 GPU (atomic 原语不涉及 CUDA)
5. 集成测试可能依赖真实 GPU + vLLM 启动, 标 `@pytest.mark.integration`
