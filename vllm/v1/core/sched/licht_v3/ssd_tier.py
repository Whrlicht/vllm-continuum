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

import ctypes
import fcntl
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
from typing import Optional

import licht_arena_atomic as _atomic

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
        # ★ P2 修 (实测 2026-07-04): promote 读走 O_DIRECT —— 同等写压力下
        # 0.389 vs buffered 0.121 GB/s (3.2x). buffered 冷读要在 page cache
        # 抢页 (本机内存大头被 pinned shm arena 锁死, 直接回收停顿), O_DIRECT
        # 直达设备. 对齐天然满足 (偏移/长度 = slot_bytes 整数倍, 目标地址 =
        # 页对齐 mmap 的 slot 偏移). 打不开 (如 tmpfs 不支持) 退回 buffered.
        self._data_fd_direct = -1
        try:
            self._data_fd_direct = os.open(data_file,
                                           os.O_RDONLY | os.O_DIRECT)
        except OSError:
            pass
        # ★ 2026-07-07: 写也走 O_DIRECT (对称于上面的读). buffered pwrite 会在
        # 内核 page cache 生成脏页; 本机 400G pinned shm arena 锁死内存, 脏页撞
        # vm.dirty_ratio 触发全局 direct-reclaim/限流 -> prefill+decode 引擎
        # 【一起】卡到 0 (复盘: 大环放大写量后复现, 小环因 94% 丢弃写量小而无感).
        # O_DIRECT 直写设备不产脏页, 阻塞代价关在写进程内不外溢. 对齐由环 slot
        # 数据区 4096 对齐保证 (ssd_stage_ring._meta_bytes). 不支持则退回 buffered.
        self._data_fd_direct_w = -1
        try:
            self._data_fd_direct_w = os.open(data_file,
                                             os.O_WRONLY | os.O_DIRECT)
        except OSError:
            pass
        self._data_file = data_file
        self._num_slots = num_slots
        self._slot_bytes = slot_bytes
        self._block_size = block_size
        self._closed = False
        # P1: CPU arena 数据源 (bind_cpu_source 注入; memoryview 零拷贝切片)
        self._cpu_mv: Optional[memoryview] = None
        self._stat_demote_blocks = 0      # 真写盘的 block 数
        self._stat_demote_hit_blocks = 0  # dedup 命中零 I/O 的 block 数
        self._stat_demote_skipped = 0     # write_inc 失败放弃的 block 数
        self._stat_promote_blocks = 0
        # ★ 摊销 sync (真机修 A): 每 inc 一次 fdatasync 在慢盘上是吞吐灾难
        # (4 进程写同一文件, 每次 sync 互刷对方脏页 -> 同步风暴; 实测 demote
        # 只有 ~3 inc/s). 这是缓存不是持久存储 —— promote 读回走 page cache
        # 天然一致, 崩溃本来就是冷启动 (shm 账本同灭). sync 只为限制脏页
        # 积压 + 让 fadvise 真能丢页, 所以按写入量摊销: 每累计
        # LICHT_SSD_SYNC_MB (默认 512) 才 fdatasync+fadvise 一次.
        self._sync_bytes_cap = int(
            os.environ.get("LICHT_SSD_SYNC_MB", "512")) * 1024 * 1024
        self._bytes_since_sync = 0
        self._sync_lock = threading.Lock()   # 2+ 写线程并发计数
        # ★ 2026-07-06 capture-at-eviction 重构 (取代 demote-ahead+pin):
        #   驱逐【释放块前】把整 inc 数据 memcpy 进 blob (capture_inc), 后台
        #   写线程刷 SSD. 驱逐本身走纯 LRU、零 pin -> 和纯 CPU 一致. (旧路
        #   demote_scan 提前 pin 冷块, 逼驱逐淘热块 -> hit 崩塌, 2026-07-05
        #   复盘.) 字节预算封顶暂存 RAM: inflight+inc > budget 就丢 (那些块不
        #   进 SSD = 纯 CPU 丢弃, 绝不阻塞驱逐). 超大 inc(>max_blk) 也丢.
        # 暂存总预算 (RAM 上限). 相对 400G pinned arena 可忽略, 给足以流水.
        self._stage_budget = int(
            os.environ.get("LICHT_SSD_STAGE_MB", "512")) * (1 << 20)
        # 分块粒度: 大 inc (长 prompt 首轮可几千块) 切成 chunk 块一段逐段存,
        # 不再整段丢 (和 CPU 驱逐 bg_chunk 同哲学).
        self._stage_chunk = max(
            1, int(os.environ.get("LICHT_SSD_STAGE_CHUNK_BLK", "64")))
        # ★ 2026-07-06 Q2: 暂存改用【跨进程 SHM 环】(ssd_stage_ring). 生产端
        #   (decode) 建环, capture memmove 数据进环槽 + 发布; 消费端 (独立写
        #   进程) drain 环 -> SSD 账本 write_inc + pwrite. 整个写路径 (含
        #   fdatasync/fadvise) 搬出 decode 进程 -> decode 只剩 memmove(放 GIL),
        #   写线程的磁盘/内存副作用彻底和 decode 隔离. 环满即丢 (背压 = 纯 CPU 丢).
        #   环槽数 = 预算 / chunk字节 (和旧池同量 RAM, 只是挪进 shm).
        self._ring_slotbytes = self._stage_chunk * int(slot_bytes)
        self._ring_n = max(2, self._stage_budget // max(self._ring_slotbytes, 1))
        self._ring_dir = os.environ.get("LICHT_SSD_RING_DIR", "/dev/shm")
        self._ring = None                # 生产端 (decode bind 时建) / 消费端 open
        self._ring_path = None
        self._stat_capture_ok = 0        # 成功入环的 inc 数
        self._stat_capture_drop = 0      # 环满/超大丢弃的 block 数
        self._stat_capture_blocks = 0    # 成功 capture 的 block 数
        # ★ capture 拷贝改用 libc.memmove: ctypes 调 C 时【释放 GIL】, 2MB/块
        #   拷贝不再和 decode 计算抢 GIL (实测: memoryview 拷贝把计算线程打到
        #   -67%, memmove 只 -4%, 2026-07-06). argtypes 必设, 否则 64 位地址被
        #   当 c_int 截断 -> 段错误.
        self._libc = ctypes.CDLL(None)
        self._libc.memmove.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._libc.memmove.restype = ctypes.c_void_p
        self._cpu_base = 0               # CPU arena mmap 基址 (bind_cpu_source 设)

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
        # 关 SHM 环 (生产端删文件; 消费端只 close mmap)
        if self._ring is not None:
            try:
                self._ring.close()
            except Exception:
                pass
            if self._ring_path is not None:   # 生产端建的, 删文件
                try:
                    os.unlink(self._ring_path)
                except OSError:
                    pass
            self._ring = None
        if self._cpu_mv is not None:
            try:
                self._cpu_mv.release()   # 释放对 CPU arena mmap 的引用
            except Exception:
                pass
            self._cpu_mv = None
        try:
            os.close(self._data_fd)
        except OSError:
            pass
        if self._data_fd_direct >= 0:
            try:
                os.close(self._data_fd_direct)
            except OSError:
                pass
        if self._data_fd_direct_w >= 0:
            try:
                os.close(self._data_fd_direct_w)
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

    # ============================================================
    # P1: 降级数据面 (CPU arena -> SSD 文件, pwrite 直达零中间缓冲)
    # ============================================================
    def bind_cpu_source(self, cpu_buf, cpu_slot_bytes: int) -> None:
        """绑定 CPU arena 作为降级数据源 (worker 侧, _arena_init 后调).

        cpu_buf: 支持 buffer 协议的对象 (生产 = CPU arena 的 mmap; 单测 =
        任意 bytes-like). memoryview 切片零拷贝 -> pwrite 时内核直接从
        shm 物理页 DMA 到盘, 我方不申请任何中转内存.
        """
        if cpu_slot_bytes != self._slot_bytes:
            raise ValueError(
                f"CPU slot_bytes={cpu_slot_bytes} != SSD {self._slot_bytes}"
                " (两层 slot 布局必须一致)")
        self._cpu_mv = memoryview(cpu_buf)
        # CPU arena mmap 基址 (给 memmove 用). from_buffer 拿首字节地址后立即
        # 释放临时 export (del), 地址随 mmap 生命周期稳定不变. mmap 可写连续,
        # 与并存的 _cpu_mv 不冲突 (多 buffer export 允许).
        _tmp = (ctypes.c_char * 1).from_buffer(cpu_buf)
        self._cpu_base = ctypes.addressof(_tmp)
        del _tmp
        # SSD 账本的 data_writer: 只在 dedup MISS 时被调 (HIT 零 I/O).
        self._store.bind_data_writer(self._pwrite_from_cpu)
        # ★ 生产端: 建 SHM 环 (decode capture 往里推). 独立写进程 open 它来 drain.
        #   命名带 pid -> 每 decode 一个环 (SPSC 无锁); 写进程 glob 发现.
        #   不再起进程内写线程 -> 写路径全在独立进程, 和 decode 隔离.
        if self._ring is None:
            from vllm.v1.core.sched.licht_v3.ssd_stage_ring import StageRing
            os.makedirs(self._ring_dir, exist_ok=True)
            # pid + id(self): 保证同进程多实例 (测试) 与跨进程都唯一;
            # 写进程 glob licht_ssd_stage_*.ring 全收.
            self._ring_path = os.path.join(
                self._ring_dir,
                f"licht_ssd_stage_{os.getpid()}_{id(self) & 0xffffff}.ring")
            self._ring = StageRing.create(
                self._ring_path, n_slots=self._ring_n,
                chunk_blocks=self._stage_chunk, slot_bytes=self._slot_bytes)
            logger.info("SsdTier stage ring created %s "
                        "(slots=%d chunk=%d blk) — writes go to sidecar",
                        self._ring_path, self._ring_n, self._stage_chunk)

    def _pwrite_from_cpu(self, ssd_slot: int, blk_idx: int, src) -> None:
        """LruArenaStore data_writer 钩子: src[blk_idx] 是 CPU slot_id."""
        cpu_slot = src[blk_idx]
        off = cpu_slot * self._slot_bytes
        os.pwrite(self._data_fd,
                  self._cpu_mv[off:off + self._slot_bytes],
                  self.slot_offset(ssd_slot))

    def capture_inc(self, job_id: str, s: int, e: int, records) -> None:
        """★ 生产端 (decode): 驱逐释放块【前】把 inc 分块 memmove 进 SHM 环槽 +
        发布. best-effort 绝不抛. 环满即丢 (背压 = 纯 CPU 丢). 只做 memmove(放
        GIL) + 发布, 不碰盘/账本 -> 对 decode 计算冲击 ~-1% (实测).

        分块: 长 prompt 首轮 inc 可几千块, 切成 _stage_chunk 块一段逐段.
        records: [(cpu_slot, gen, hash), ...]. 块此刻仍分配 (arena 持 _evict_lock)."""
        if self._cpu_mv is None or self._closed or self._ring is None:
            return
        n = e - s
        if n <= 0 or len(records) != n:
            if n > 0:
                self._stat_capture_drop += n
            return
        sb = self._slot_bytes
        chunk = self._stage_chunk
        off = 0
        while off < n:
            c = min(chunk, n - off)
            r = self._ring.reserve()        # 判满 + 拿槽; 满则丢该段及之后
            if r is None:
                self._stat_capture_drop += (n - off)
                return
            h, idx = r
            try:
                dst = self._ring.data_addr(idx)
                hashes = []
                for i in range(c):
                    rec = records[off + i]
                    # memmove 释放 GIL; 数据进环槽 (arena 持锁, CPU 数据稳定).
                    self._libc.memmove(dst + i * sb,
                                       self._cpu_base + int(rec[0]) * sb, sb)
                    hashes.append(rec[2])
                self._ring.publish(h, idx, str(job_id),
                                   int(s + off), c, hashes)
                self._stat_capture_ok += 1
                self._stat_capture_blocks += c
            except Exception:
                self._stat_capture_drop += c
                # 未 publish 的槽自然不占 head, 无需归还.
            off += c

    def drain_ring(self, ring, max_items: int = 256) -> int:
        """★ 消费端 (独立写进程 / 测试): 从环取就绪槽 -> SSD 账本 write_inc
        (从环槽读, 零中间拷贝) -> release. 返回处理的 inc 数. 慢 I/O 全在此
        (和 decode 隔离). 摊销 sync 同旧写线程."""
        done = 0
        for _ in range(max_items):
            item = ring.pop()
            if item is None:
                break
            job_id, start, count, hashes, idx = item
            e = start + count
            try:
                def _dw(ssd_slot, blk, _src, _r=ring, _i=idx):
                    _off = self.slot_offset(ssd_slot)
                    _wfd = self._data_fd_direct_w
                    if _wfd >= 0:
                        try:
                            # 环 slot 数据区 4096 对齐 -> 可 O_DIRECT 直写设备.
                            os.pwritev(_wfd, [_r.block_mv(_i, blk)], _off)
                            return
                        except OSError as _e:   # 对齐/设备不支持 -> 永久退 buffered
                            self._data_fd_direct_w = -1
                            logger.warning(
                                "O_DIRECT 写失败(%s) -> 退回 buffered "
                                "(pinned-RAM 下有脏页停顿风险)", _e)
                    os.pwrite(self._data_fd, _r.block_mv(_i, blk), _off)
                _miss_before = self._store._stat_miss_blocks
                ok = self._store.write_inc(
                    job_id, start, e, token_ids=[],
                    source_obj=None, inc_hashes=hashes, data_writer=_dw)
                if ok:
                    _written = self._store._stat_miss_blocks - _miss_before
                    self._stat_demote_blocks += _written
                    self._stat_demote_hit_blocks += count - _written
                    if _written > 0:
                        # 多写线程共享摊销 sync 计数, 加锁 (每环一线程 -> N 线程)
                        _do_sync = False
                        with self._sync_lock:
                            self._bytes_since_sync += _written * self._slot_bytes
                            if self._bytes_since_sync >= self._sync_bytes_cap:
                                self._bytes_since_sync = 0
                                _do_sync = True
                        # O_DIRECT 写不产脏页, 且设备缓存对 O_DIRECT 读一致,
                        # 无需 fdatasync (它本身阻塞刷设备缓存 = 又一处停顿源);
                        # 仅 buffered 退路需靠摊销 sync 限制脏页积压.
                        if _do_sync and self._data_fd_direct_w < 0:
                            try:
                                os.fdatasync(self._data_fd)
                            except OSError:
                                pass
                else:
                    self._stat_demote_skipped += count
            except Exception as ex:  # pragma: no cover
                logger.warning("drain_ring write failed job=%s [%d,%d): %s",
                               str(job_id)[:32], start, e, ex)
            finally:
                ring.release()          # 无论成败都释放槽 (best-effort)
            done += 1
        return done

    def demote_inc(self, job_id: str, s: int, e: int, records) -> bool:
        """降级一个 inc (LruArenaStore 降级写线程回调).

        records: [(cpu_slot, gen, hash), ...] —— 调用方已整 inc pin 住
        (数据不会被动/被改), 返回后由调用方 unpin.

        走 SSD 账本的标准 write_inc (hash 注入): 内容寻址命中 -> refcnt++
        零字节 I/O ("每份内容终生只写一次盘"); miss -> alloc SSD slot +
        _pwrite_from_cpu 落盘 + 两段式发布. SSD 满时 write_inc 内部走
        自己的 LRU 淘汰 (drop-only). 失败返 False (调用方不标干净,
        该 inc 退化为直接丢弃).
        """
        if self._cpu_mv is None or self._closed:
            return False
        cpu_slots = [r[0] for r in records]
        hashes = [r[2] for r in records]
        _miss_before = self._store._stat_miss_blocks
        ok = self._store.write_inc(job_id, s, e, token_ids=[],
                                   source_obj=cpu_slots,
                                   inc_hashes=hashes)
        if not ok:
            self._stat_demote_skipped += len(records)
            return False
        _written = self._store._stat_miss_blocks - _miss_before
        self._stat_demote_blocks += _written
        self._stat_demote_hit_blocks += len(records) - _written
        if _written > 0:
            # 摊销 sync+丢缓存 (见 __init__ 注释): 攒够 LICHT_SSD_SYNC_MB
            # 才刷一次. buffered 写脏页要先回写才可被 fadvise 丢.
            _do_sync = False
            with self._sync_lock:
                self._bytes_since_sync += _written * self._slot_bytes
                if self._bytes_since_sync >= self._sync_bytes_cap:
                    self._bytes_since_sync = 0
                    _do_sync = True
            if _do_sync:
                try:
                    os.fdatasync(self._data_fd)
                    os.posix_fadvise(self._data_fd, 0, 0,
                                     os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
        return True

    # ============================================================
    # P2: 升级数据面 (SSD 文件 -> CPU arena, pread 直达零中间缓冲)
    # ============================================================
    def _pread_to_cpu(self, cpu_slot: int, blk_idx: int, src) -> None:
        """CPU 账本 write_inc 的 data_writer 覆盖: src[blk_idx] 是 SSD
        slot_id, 把它的字节 pread 直读进 CPU arena 新分配的 cpu_slot.
        优先 O_DIRECT (见 __init__ 注释), 不可用退回 buffered."""
        ssd_slot = src[blk_idx]
        off = cpu_slot * self._slot_bytes
        fd = (self._data_fd_direct if self._data_fd_direct >= 0
              else self._data_fd)
        os.preadv(fd,
                  [self._cpu_mv[off:off + self._slot_bytes]],
                  self.slot_offset(ssd_slot))

    def promote_inc(self, cpu_store, job_id: str, start_block: int,
                    end_block: int, records) -> Optional[list]:
        """把 SSD 上的段 [start_block, end_block) 搬回 CPU arena (P2 升级).

        调用于 worker 引擎线程 (admit 后同步基线): 卡引擎时长 = pread 时间,
        上限由 claim 期的 LICHT_SSD_PROMOTE_MAX_MB 保证.

        records: [(ssd_slot, gen, hash), ...] 与块区间逐块对齐 (scheduler
        侧 resolve_range 的产物, claim 后 SSD 账本已 mark_inflight 防淘).

        流程 (P1 demote 的镜像):
          1. try_pin 全部 SSD 源槽 (gen 校验; 任一失败 = claim 后仍被淘的
             残余竞态 -> 整段放弃, 调用方走 fail-closed);
          2. CPU 账本标准 write_inc: hash 注入 + data_writer=pread 直读
             (dedup HIT 的块零 I/O; 两段式发布保证半读不可见);
          3. unpin; 用同批 hash 重探 CPU 表拿 (slot, gen) 返回, 喂显式 load.

        返回 [(cpu_slot, gen), ...] 或 None (失败, 不 raise)."""
        if self._cpu_mv is None or self._closed or not records:
            return None
        n = end_block - start_block
        if n <= 0 or len(records) != n:
            return None
        # 1) pin SSD 源槽
        pinned: list = []
        for (slot, gen, _h) in records:
            addr = self._store._hdr.slot_state_addr(slot)
            if not _atomic.try_pin(addr, gen):
                for a in pinned:
                    _atomic.unpin(a)
                # ★ 取证探针 (2026-07-04 复盘): 修复后 pin 失败理论不可能
                # (当步现探 + claim 即 inflight), 真发生时把现场吐全 —— 到底
                # 是 gen 变了(被淘复用) / 槽空闲(被释放) / inflight 没挡住.
                try:
                    _cur = _atomic.get_gen(addr)
                    _free = bool(self._store._allocator.is_free(slot))
                    _infl = self._store.is_inflight(str(job_id))
                except Exception:
                    _cur, _free, _infl = -1, None, None
                logger.error(
                    "promote_inc: SSD slot pin FAILED job=%s [%d,%d) "
                    "slot=%d expected_gen=%d cur_gen=%s is_free=%s "
                    "job_inflight=%s -> 整段放弃 (三道防陈旧闸有漏, 需查)",
                    str(job_id)[:32], start_block, end_block,
                    slot, gen, _cur, _free, _infl)
                return None
            pinned.append(addr)
        try:
            ssd_slots = [r[0] for r in records]
            hashes = [r[2] for r in records]
            # 2) 写进 CPU 账本 (pread 直读, dedup 幂等)
            ok = cpu_store.write_inc(
                job_id, start_block, end_block, token_ids=[],
                source_obj=ssd_slots, inc_hashes=hashes,
                data_writer=self._pread_to_cpu)
            if not ok:
                return None
            # 3) 重探 CPU 表拿新 (slot, gen) —— dedup HIT 块拿到已有槽,
            # 同样正确; 探不到 (极端并发) -> 整段失败 fail-closed.
            sg = cpu_store.probe_slots(hashes)
            if sg is None:
                logger.warning(
                    "promote_inc: post-promote probe miss job=%s [%d,%d)",
                    str(job_id)[:32], start_block, end_block)
                return None
            self._stat_promote_blocks += n
            # (P2 修) 不再 fadvise: O_DIRECT 读不经 page cache, fadvise 无
            # 意义; buffered 退路下保留缓存反而救 thrash 二次读 (实测 fadvise
            # 后重读慢 8x).
            return sg
        except Exception as e:  # pragma: no cover
            logger.warning("promote_inc failed job=%s [%d,%d): %s",
                           str(job_id)[:32], start_block, end_block, e)
            return None
        finally:
            for a in pinned:
                _atomic.unpin(a)

    def stats(self) -> dict:
        try:
            free = self._store.free_count()
        except Exception:
            free = -1
        return {
            "num_slots": self._num_slots,
            "free": free,
            "demote_blocks": self._stat_demote_blocks,
            "demote_hit_blocks": self._stat_demote_hit_blocks,
            "demote_skipped": self._stat_demote_skipped,
            "promote_blocks": self._stat_promote_blocks,
            # capture-at-eviction 计数
            "capture_ok": self._stat_capture_ok,
            "capture_blocks": self._stat_capture_blocks,
            "capture_drop": self._stat_capture_drop,
            # 环在途槽数 × chunk 字节 = 待写进程 drain 的暂存量 (MB)
            "stage_inflight_mb": (
                (self._ring.depth() * self._ring_slotbytes) >> 20)
                if self._ring is not None else 0,
        }
