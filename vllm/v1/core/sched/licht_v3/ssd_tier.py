# SPDX-License-Identifier: Apache-2.0
"""LICHT Round-KV SSD tier (P0 骨架) —— 跨轮 KV 的最冷层.

分层模型 (与 round_kv_store.py 协作):
    GPU 显存 (工作台)  <->  CPU shm arena (货架)  <->  SSD arena (冷库, 本模块)

设计要点:
  - 账本复用 LruArenaStore: 第二个实例, storage_path 指向 /dev/shm 下的
    meta 目录 (hdr/manifest/.slot 全在内存 -> 高频原子操作零磁盘回写).
    LruArenaStore 本来就不碰数据 (搬运注入), 因此零改动复用.
  - 数据是 data_path 下的单个 posix_fallocate 大文件: slot_id 线性映射
    文件偏移 (slot n 在 n*slot_bytes 处), 无 per-slot 小文件.
  - 数据搬运由调用者注入/执行 (P1 demote = pwrite(shm slot 地址 -> 文件偏移),
    P2 promote = pread(文件偏移 -> shm slot 地址)); 本模块与 GPU 无关,
    CPU-only 单测可覆盖.
  - 跨进程: 多 worker (prefill/decode) open_or_create 同一 meta/data,
    互斥靠 LruArenaStore 现有 flock 创建协议; num_slots 由相同 env +
    slot_bytes 推出, 各进程一致. posix_fallocate 幂等 (二次调用无操作).
  - 重启语义: 启动脚本清 /dev/shm -> 账本清零 = SSD 缓存整体重置; 数据
    文件残留字节无账本引用, 会被自然覆写. 无 "账本与数据对不上" 的启动态.
  - best-effort: open_or_create 失败会抛异常, 由调用方 (round_kv_store
    的 _arena_init) try 包裹 -> 降级为无 SSD tier, 主路径不受影响.

P0 只有生命周期 (open/close/stats); demote/promote 在 P1/P2 加入.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import tempfile
from typing import Optional

from vllm.v1.core.sched.licht_v3.arena_lru_store import LruArenaStore

logger = logging.getLogger(__name__)

_GB = 1024 ** 3
_DATA_FILE = "arena.data"
_META_FILE = "_ssd_meta.json"


class SsdTier:
    """SSD 冷层: LruArenaStore 账本 (shm) + 单大数据文件 (SSD)."""

    def __init__(self, store: LruArenaStore, data_fd: int, data_file: str,
                 num_slots: int, slot_bytes: int, block_size: int):
        self._store = store
        self._data_fd = data_fd
        self._data_file = data_file
        self._num_slots = num_slots
        self._slot_bytes = slot_bytes
        self._block_size = block_size
        self._closed = False
        # P1 起会加: demote 队列/写线程/计数器
        self._stat_demote_blocks = 0
        self._stat_demote_skipped = 0
        self._stat_promote_blocks = 0

    # ============================================================
    # Lifecycle
    # ============================================================
    @classmethod
    def open_or_create(cls, meta_path: str, data_path: str, ssd_gb: float,
                       slot_bytes: int, block_size: int,
                       wait_timeout_s: float = 60.0) -> "SsdTier":
        """跨进程安全打开/创建 SSD tier. 失败抛异常 (调用方降级).

        meta_path: 账本目录 (应在 /dev/shm 下)
        data_path: 数据大文件所在目录 (应在 SSD 文件系统上)
        ssd_gb:    冷库容量 (GB); num_slots = 容量 // slot_bytes
        slot_bytes/block_size: 与 CPU arena 一致 (worker 侧布局推出后才可知)
        """
        if slot_bytes <= 0:
            raise ValueError(f"slot_bytes={slot_bytes} 非法")
        num_slots = max(int(ssd_gb * _GB) // int(slot_bytes), 1)
        os.makedirs(meta_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)

        # ★ 布局变更守卫: 残留账本 (换模型/改容量/改 LICHT_SSD_GB 后 shm 未清)
        # 的 num_slots/slot_bytes 与本次不一致时, 用不匹配的 num_slots 打开旧
        # hdr 会布局错乱 -> 整个 meta 目录重建. SSD 是缓存, 清空 = 冷启动,
        # 不丢正确性. 跨进程互斥: data_path 下的 init.lock flock 串行整段
        # 检查+重建 (各进程 num_slots 由相同 env+布局推出, 判定一致).
        _lk = os.open(os.path.join(data_path, ".ssd_init.lock"),
                      os.O_CREAT | os.O_RDWR, 0o600)
        store = None
        fd = -1
        data_file = os.path.join(data_path, _DATA_FILE)
        try:
            fcntl.flock(_lk, fcntl.LOCK_EX)
            old = cls.read_meta(meta_path)
            _hdr_exists = os.path.exists(
                os.path.join(meta_path, "_arena.hdr"))
            # 重建条件: (a) meta 在但布局不匹配 (换模型/容量后 shm 残留);
            # (b) meta 缺但 hdr 在 (上次初始化中途崩溃, meta 是锁内最后写的
            # "初始化完成" 标记). 两者都不能带病打开 -> 整目录重建冷启动.
            _stale = ((old is not None
                       and (old.get("num_slots") != num_slots
                            or old.get("slot_bytes") != int(slot_bytes)))
                      or (old is None and _hdr_exists))
            if _stale:
                logger.warning(
                    "SsdTier: 账本残留不可用 (旧 meta=%s, hdr_exists=%s, "
                    "新 num_slots=%d slot_bytes=%d) -> 重建 meta 目录 "
                    "(SSD 缓存冷启动)",
                    old, _hdr_exists, num_slots, int(slot_bytes))
                shutil.rmtree(meta_path, ignore_errors=True)
                os.makedirs(meta_path, exist_ok=True)

            # 账本: 复用 LruArenaStore 跨进程 flock 创建协议 (谁先到谁 init).
            store = LruArenaStore.open_or_create(
                meta_path, num_slots=num_slots, block_size=block_size,
                wait_timeout_s=wait_timeout_s)

            fd = os.open(data_file, os.O_RDWR | os.O_CREAT, 0o600)
            # 一次性预留全尺寸: 防边写边扩 (碎片/挤爆盘); 幂等, 并发安全.
            # ENOSPC 等在此立刻暴露 -> 调用方降级, 不带病运行.
            os.posix_fallocate(fd, 0, num_slots * slot_bytes)

            # meta json: 锁内最后写 (= "初始化完成" 的标记; 崩溃在此之前 ->
            # 下个进程重走守卫). scheduler 侧 (无 kv_caches, 不知 slot_bytes)
            # P2 做两层 lookup 拼接时靠它 lazy open. 原子写 (tmp+replace).
            _fd, _tmp = tempfile.mkstemp(dir=meta_path, prefix=".ssdmeta_",
                                         suffix=".tmp")
            with os.fdopen(_fd, "w") as f:
                json.dump({"num_slots": int(num_slots),
                           "slot_bytes": int(slot_bytes),
                           "block_size": int(block_size)}, f)
            os.replace(_tmp, os.path.join(meta_path, _META_FILE))
        except OSError:
            if fd >= 0:
                os.close(fd)
            if store is not None:
                store.close()
            raise
        finally:
            try:
                fcntl.flock(_lk, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(_lk)

        tier = cls(store, fd, data_file, num_slots, slot_bytes, block_size)
        logger.info(
            "SSD tier opened: %.1fGB (%d slots x %.2fMB), data=%s, meta=%s, "
            "content_addr=%s, free=%d",
            num_slots * slot_bytes / _GB, num_slots, slot_bytes / 1e6,
            data_file, meta_path, store.content_addr, store.free_count())
        return tier

    @staticmethod
    def read_meta(meta_path: str) -> Optional[dict]:
        """读 worker 写的 meta json (scheduler 侧 lazy open 用). 失败返 None."""
        try:
            with open(os.path.join(meta_path, _META_FILE)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._data_fd)
        except OSError:
            pass
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ============================================================
    # 属性 / 统计
    # ============================================================
    @property
    def store(self) -> LruArenaStore:
        return self._store

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def slot_bytes(self) -> int:
        return self._slot_bytes

    @property
    def data_file(self) -> str:
        return self._data_file

    def slot_offset(self, slot_id: int) -> int:
        """slot_id -> 数据文件内字节偏移 (线性映射)."""
        return slot_id * self._slot_bytes

    def stats(self) -> dict:
        try:
            free = self._store.free_count()
        except Exception:
            free = -1
        return {
            "num_slots": self._num_slots,
            "free": free,
            "demote_blocks": self._stat_demote_blocks,
            "demote_skipped": self._stat_demote_skipped,
            "promote_blocks": self._stat_promote_blocks,
        }
