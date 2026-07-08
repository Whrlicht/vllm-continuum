# SPDX-License-Identifier: Apache-2.0
"""跨进程无锁 SPSC 暂存环 (Q2: 把 SSD 写从 decode 进程剥离).

decode 进程 = 唯一生产者 (驱逐时 capture: memmove CPU 槽数据 -> 环数据槽,
发布 head); 独立写进程 = 唯一消费者 (读就绪槽 -> SSD 账本 write_inc, 推进
tail 释放槽). 单生产单消费 (SPSC) + head/tail 用 ACQ_REL 原子 -> 无需任何锁.

正确性 (SPSC 内存序):
  - 生产者: 先写数据+元数据, 再 atomic_store(head+1) [RELEASE] -> 数据对
    消费者可见后 head 才 +1;
  - 消费者: atomic_load(head) [ACQUIRE] 后才读数据 -> 看到 head 增量即看到数据;
    处理完 atomic_store(tail+1) [RELEASE] 释放槽;
  - 生产者 atomic_load(tail) [ACQUIRE] 判满 (head-tail>=N).
环满 -> 生产者丢 (背压, = 纯 CPU 丢弃, 绝不阻塞驱逐).

自描述: 头里存 n_slots/chunk_blocks/slot_bytes, 写进程 open 后即知布局, 无需
额外配置. best-effort: 任何异常调用方吞掉降级为无 SSD.
"""
from __future__ import annotations

import ctypes
import mmap as _mmap
import os
import struct

import licht_arena_atomic as _atomic

_MAGIC = 0x4C494348_53544731         # "LICHSTG1"
_HEADER_SIZE = 4096                  # 头区 (page 对齐)
_JOBID_BYTES = 64                    # job_id 定长区
_DIO_ALIGN = 4096                    # O_DIRECT 对齐: 每 slot 数据区须落 4096 边界,
                                     # 好让写进程对其做 O_DIRECT pwritev (直写设备,
                                     # 不经 page cache -> 不产脏页, 见 ssd_tier 复盘)
# 头字段偏移 (u64 各占 8B; head/tail 分处不同 cache line 防 false sharing)
_OFF_MAGIC = 0
_OFF_VERSION = 8
_OFF_NSLOTS = 16
_OFF_CHUNK = 24
_OFF_SLOTBYTES = 32
_OFF_METABYTES = 40
_OFF_STRIDE = 48
_OFF_HEAD = 64                       # 生产者写
_OFF_TAIL = 128                      # 消费者写 (隔 64B)


def _meta_bytes(chunk_blocks: int) -> int:
    # start_block(8) + count(8) + job_id(64) + hashes(8*chunk).
    # 上取 _DIO_ALIGN(4096): meta 与 stride(=meta+chunk*slot_bytes, slot_bytes
    # 为 2MB=4096 倍数) 都 4096 对齐 -> 每 slot 数据区起始地址落 4096 边界,
    # 写进程可对其 O_DIRECT 直写 (不产脏页). 每 slot 多几 KB, 相对 128MB 可忽略.
    raw = 16 + _JOBID_BYTES + 8 * chunk_blocks
    return (raw + _DIO_ALIGN - 1) & ~(_DIO_ALIGN - 1)


class StageRing:
    """无锁 SPSC 暂存环 (shm mmap)."""

    def __init__(self, path: str, mm, base_addr: int, cref,
                 n_slots: int, chunk_blocks: int, slot_bytes: int,
                 meta_bytes: int, stride: int):
        self._path = path
        self._mm = mm
        self._base = base_addr
        self._cref = cref               # 保活 from_buffer export
        self._n = n_slots
        self._chunk = chunk_blocks
        self._sb = slot_bytes
        self._meta = meta_bytes
        self._stride = stride
        self._head_addr = base_addr + _OFF_HEAD
        self._tail_addr = base_addr + _OFF_TAIL
        self._mv = memoryview(mm)

    # ---------- 生命周期 ----------
    @classmethod
    def create(cls, path: str, n_slots: int, chunk_blocks: int,
               slot_bytes: int) -> "StageRing":
        """生产者 (decode) 建环. 已存在则重建 (冷启动, 环是纯暂存)."""
        n_slots = max(2, int(n_slots))
        meta = _meta_bytes(chunk_blocks)
        stride = meta + chunk_blocks * slot_bytes
        size = _HEADER_SIZE + n_slots * stride
        # 原子重建: 写临时文件再 rename (防写进程读到半初始化的头)
        tmp = path + f".tmp.{os.getpid()}"
        fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        os.ftruncate(fd, size)
        os.close(fd)
        os.replace(tmp, path)
        fd = os.open(path, os.O_RDWR)
        mm = _mmap.mmap(fd, size, _mmap.MAP_SHARED,
                        _mmap.PROT_READ | _mmap.PROT_WRITE)
        os.close(fd)
        cref = (ctypes.c_char * size).from_buffer(mm)
        base = ctypes.addressof(cref)
        # 写头 (magic 最后写 = "就绪" 标记)
        struct.pack_into("<QQQQQ", mm, _OFF_NSLOTS,
                         n_slots, chunk_blocks, slot_bytes, meta, stride)
        _atomic.atomic_store_u64(base + _OFF_HEAD, 0)
        _atomic.atomic_store_u64(base + _OFF_TAIL, 0)
        struct.pack_into("<Q", mm, _OFF_VERSION, 1)
        _atomic.atomic_store_u64(base + _OFF_MAGIC, _MAGIC)  # release 发布头
        return cls(path, mm, base, cref, n_slots, chunk_blocks,
                   slot_bytes, meta, stride)

    @classmethod
    def open(cls, path: str) -> "StageRing":
        """消费者 (写进程) 打开已存在的环, 从头读布局. 头未就绪抛异常."""
        fd = os.open(path, os.O_RDWR)
        try:
            st = os.fstat(fd)
            mm = _mmap.mmap(fd, st.st_size, _mmap.MAP_SHARED,
                            _mmap.PROT_READ | _mmap.PROT_WRITE)
        finally:
            os.close(fd)
        cref = (ctypes.c_char * st.st_size).from_buffer(mm)
        base = ctypes.addressof(cref)
        if _atomic.atomic_load_u64(base + _OFF_MAGIC) != _MAGIC:
            raise ValueError(f"StageRing {path} 头未就绪/magic 不符")
        (n_slots, chunk_blocks, slot_bytes, meta, stride) = struct.unpack_from(
            "<QQQQQ", mm, _OFF_NSLOTS)
        exp = _HEADER_SIZE + n_slots * stride
        if st.st_size < exp:
            raise ValueError(f"StageRing {path} 文件过小 {st.st_size}<{exp}")
        return cls(path, mm, base, cref, int(n_slots), int(chunk_blocks),
                   int(slot_bytes), int(meta), int(stride))

    # ---------- 生产者 (decode capture) ----------
    def data_addr(self, idx: int) -> int:
        """槽 idx 的数据区起始【内存地址】(memmove 目标)."""
        return self._base + _HEADER_SIZE + idx * self._stride + self._meta

    def reserve(self):
        """判满 + 返回可写槽的 (环序号 h, 槽索引 idx). 满则返回 None (丢)."""
        h = _atomic.atomic_load_u64(self._head_addr)
        t = _atomic.atomic_load_u64(self._tail_addr)
        if h - t >= self._n:
            return None
        return h, h % self._n

    def publish(self, h: int, idx: int, job_id: str, start_block: int,
                count: int, hashes) -> None:
        """写元数据 + 发布 head (数据须已由调用方 memmove 进 data_addr(idx))."""
        base_off = _HEADER_SIZE + idx * self._stride
        jb = job_id.encode("utf-8")[:_JOBID_BYTES].ljust(_JOBID_BYTES, b"\0")
        struct.pack_into("<QQ", self._mm, base_off,
                         int(start_block), int(count))
        self._mm[base_off + 16:base_off + 16 + _JOBID_BYTES] = jb
        hoff = base_off + 16 + _JOBID_BYTES
        struct.pack_into("<%dQ" % count, self._mm, hoff,
                         *[int(hashes[i]) for i in range(count)])
        _atomic.atomic_store_u64(self._head_addr, h + 1)   # RELEASE 发布

    # ---------- 消费者 (写进程) ----------
    def pop(self):
        """取下一个就绪槽: 返回 (job_id, start_block, count, hashes, idx)
        或 None (空). 处理完须调 release()."""
        t = _atomic.atomic_load_u64(self._tail_addr)
        h = _atomic.atomic_load_u64(self._head_addr)   # ACQUIRE
        if t >= h:
            return None
        idx = t % self._n
        base_off = _HEADER_SIZE + idx * self._stride
        start_block, count = struct.unpack_from("<QQ", self._mm, base_off)
        jb = bytes(self._mm[base_off + 16:base_off + 16 + _JOBID_BYTES])
        job_id = jb.rstrip(b"\0").decode("utf-8", "replace")
        hoff = base_off + 16 + _JOBID_BYTES
        hashes = list(struct.unpack_from("<%dQ" % count, self._mm, hoff))
        return job_id, int(start_block), int(count), hashes, idx

    def block_mv(self, idx: int, blk: int) -> memoryview:
        """槽 idx 第 blk 块数据的 memoryview (pwrite 源, 零拷贝)."""
        off = _HEADER_SIZE + idx * self._stride + self._meta + blk * self._sb
        return self._mv[off:off + self._sb]

    def release(self) -> None:
        """推进 tail 释放当前槽 (RELEASE)."""
        t = _atomic.atomic_load_u64(self._tail_addr)
        _atomic.atomic_store_u64(self._tail_addr, t + 1)

    # ---------- 属性 ----------
    @property
    def chunk_blocks(self) -> int:
        return self._chunk

    @property
    def slot_bytes(self) -> int:
        return self._sb

    def depth(self) -> int:
        """当前在途槽数 (head-tail)."""
        return (_atomic.atomic_load_u64(self._head_addr)
                - _atomic.atomic_load_u64(self._tail_addr))

    def close(self) -> None:
        try:
            self._mv.release()
        except Exception:
            pass
        try:
            del self._cref
        except Exception:
            pass
        try:
            self._mm.close()
        except Exception:
            pass
