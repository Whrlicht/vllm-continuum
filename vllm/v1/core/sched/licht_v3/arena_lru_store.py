# SPDX-License-Identifier: Apache-2.0
"""LICHT Round-KV Arena LRU store (Stage 2 顶层类).

设计哲学:
  - 本类不直接接触 GPU; 只管 slot 分配 / .slot 文件 / mutex / pin / LRU
  - 数据搬运 (gather/scatter) 由调用者注入: write_inc 接收已 gather 的数据,
    load_request 返回 (slot_ids, gens) 让调用者自己搬
  - 这样既能在 CPU only 单元测试里跑, 也能在 round_kv_store.py 里桥接到 GPU

关键 API:
  - LruArenaStore(storage_path, num_slots, block_size)
  - .bind_data_writer(fn)  / .bind_data_reader(fn)   # 注入搬数据钩子
  - .write_inc(job_id, start, end, token_ids, gathered_blocks)
  - .lookup(job_id, prompt_token_ids) -> (matched_tokens, matched_blocks)
  - .load_request(job_id, dst_block_ids, src_block_offset)
       -> Optional[LoadHandle]  (handle 含 slot 列表 + gens, 调用者 scatter 后 unpin)
  - .delete_job(job_id), .mark_finished_job(job_id)
  - .free_count() / .num_slots / .num_jobs

跨进程协议:
  - 一个进程 create() 时初始化 hdr + mutex
  - 其他进程 open() 仅 mmap, 跨进程共享 mutex
  - 所有 write 路径在 alloc_mutex 临界区内执行
  - lookup 是 lock-free (读文件 + atomic 校验 gen)
  - load_request 通过 try_pin 保护
"""
from __future__ import annotations

import json
import logging
import os
import struct
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import licht_arena_atomic as _atomic

from vllm.v1.core.sched.licht_v3.arena_allocator import ArenaAllocator
from vllm.v1.core.sched.licht_v3.arena_block_hash import SEED0, block_hashes
from vllm.v1.core.sched.licht_v3.arena_hdr import ArenaHdr
from vllm.v1.core.sched.licht_v3.arena_slot_file import (
    SlotFileV1,
    parse_slot_filename,
    read_slot_file_v1,
    read_slot_file_v2,
    read_slot_file_version,
    slot_filename,
    write_slot_file_v1,
    write_slot_file_v2,
)


logger = logging.getLogger(__name__)

# Stage 6 perf: 整条 lookup_resolve 是否有 C 实现 (旧 .so 没有则回退 Python).
_HAS_C_LOOKUP = hasattr(_atomic, "lookup_resolve")

_MANIFEST = "manifest.json"
_RESERVED_DIR_PREFIXES = ("_", ".")  # 不当作 job 处理的目录前缀


# ============================================================
# Helpers
# ============================================================
def _safe_job_id(job_id: str) -> str:
    """把 job_id 转成安全的目录名."""
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in str(job_id)]
    return "".join(keep)[:200]


def align_blocks(num_tokens: int, block_size: int) -> int:
    if num_tokens <= 0 or block_size <= 0:
        return 0
    return num_tokens // block_size


# ============================================================
# Load handle (load_request 返回, 用于 unpin)
# ============================================================
@dataclass
class LoadHandle:
    """load_request 成功时返回. 调用者搬完数据必须调 .release()."""
    slot_state_addrs: List[int]   # 已 pin 的 slot_state 地址列表
    slot_ids: List[int]            # 对应 slot id (调用者用来读 arena_view)
    gens: List[int]                # 对应 gen (post-load 二次校验用)
    dst_block_ids: List[int]       # 目标 paged block 号 (调用者用来 scatter)
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        for addr in self.slot_state_addrs:
            _atomic.unpin(addr)
        self._released = True

    def post_load_validate(self) -> bool:
        """load 结束后再校验一次 gen.

        返回 True: 期间没被 evict, 数据有效
        返回 False: 期间被 evict 覆盖 (race), 应走 fallback

        注意: 即使返回 False, slot 的 pin 仍由 release() 减回.
        实际上 pin > 0 时 evict 不应触发, 所以这条几乎不会触发
        (留着兜底 + 监控).
        """
        for addr, expected_gen in zip(self.slot_state_addrs, self.gens):
            cur_gen = _atomic.get_gen(addr)
            if cur_gen != expected_gen:
                return False
        return True


@dataclass
class BatchLoadHandle:
    """load_batch_pin 返回. 一整波 admit 请求的展平 slot/dst/gen + pin 地址.

    调用者把 (slot_ids, dst_block_ids) 喂直读 kernel, 完成后必须 release().
    """
    per_item_ok: List[bool]        # 每个请求是否成功解析+pin
    slot_ids: List[int]             # 所有成功请求的 block 的 arena slot (展平)
    dst_block_ids: List[int]        # 对应目标 paged block 号 (同序)
    gens: List[int]                 # 对应 gen (post-load 校验)
    slot_state_addrs: List[int]     # 已 pin 的 slot_state 地址
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        for addr in self.slot_state_addrs:
            _atomic.unpin(addr)
        self._released = True

    def post_load_validate(self) -> bool:
        """整波 load 完后再校验所有 pin 的 slot 的 gen (race 检测)."""
        for addr, expected_gen in zip(self.slot_state_addrs, self.gens):
            if _atomic.get_gen(addr) != expected_gen:
                return False
        return True


# ============================================================
# Main class
# ============================================================
class LruArenaStore:
    def __init__(self,
                 storage_path: str,
                 num_slots: int,
                 block_size: int):
        self._storage_path = storage_path
        self._num_slots = num_slots
        self._block_size = block_size
        self._hdr: Optional[ArenaHdr] = None
        self._allocator: Optional[ArenaAllocator] = None

        # in-memory caches
        self._last_stored: dict[str, int] = {}
        # in-memory LRU: job_id -> None, OrderedDict 保序 (最老在前, 最新在后).
        # evict 选 victim 直接从这里取最老, O(1), 避免在临界区内扫文件系统
        # (os.listdir + 每 job os.stat manifest mtime, O(jobs), 写满后会塌方).
        # 本进程 store 过的 job 都在这里; 跨进程对方写的 job 不在 (各 evict 各的,
        # bitmap 共享所以不会重分配, gen 失效保证 lookup 正确性).
        self._job_lru: "OrderedDict[str, None]" = OrderedDict()

        # 数据搬运钩子 (由调用者注入)
        # data_writer(slot_id, block_idx_in_gathered, gathered): 把 gathered[block_idx]
        #     的 block 数据写到 arena slot_id 位置
        # data_reader(slot_id) -> 无返回, 由调用者根据 slot_id 列表自行处理
        # 我们不直接调 reader, 因为 reader 通常是批量 H2D
        self._data_writer: Optional[Callable[[int, int, object], None]] = None

        # 诊断日志 (LICHT_ARENA_DEBUG=1): 打 write_inc/evict/load 的 slot/gen
        # 细节, 用于验证 Phase 1/2 等路径下 LRU 真在工作且 slot/gen 对得上.
        self._debug = os.environ.get("LICHT_ARENA_DEBUG", "0") == "1"

        # ★ Stage 6 内容寻址总开关 (LICHT_ARENA_CONTENT_ADDR=1).
        # 开: store 走 dedup (probe hash 表, 命中 refcnt++ 不新分配), .slot 写 v2;
        # 关: 完全等价旧路径 (alloc 全新 slot, .slot 写 v1), 字节级不变.
        self._content_addr = (
            os.environ.get("LICHT_ARENA_CONTENT_ADDR", "0") == "1")
        # 临时探针: dedup write_inc 分段计时 (hash/cs1/data/slot/cs2). 默认关.
        self._write_profile = (
            os.environ.get("LICHT_ROUND_KV_WRITE_PROFILE", "0") == "1")
        # 埋点计数 (LICHT_ARENA_DEBUG): dedup 命中 / 新分配 block 数,
        # evict 真释放 (refcnt->0) vs 仅减引用 (refcnt>0 数据留给别 job)
        self._stat_hit_blocks = 0
        self._stat_miss_blocks = 0
        self._stat_evict_freed = 0
        self._stat_evict_decref = 0

    # ============================================================
    # Lifecycle
    # ============================================================
    @classmethod
    def create(cls, storage_path: str, num_slots: int,
               block_size: int) -> "LruArenaStore":
        """首次创建: ftruncate hdr, init mutex, 标 bitmap 全 free."""
        store = cls(storage_path, num_slots, block_size)
        os.makedirs(storage_path, exist_ok=True)
        hdr_path = os.path.join(storage_path, "_arena.hdr")
        store._hdr = ArenaHdr.create(hdr_path, num_slots=num_slots)
        store._allocator = ArenaAllocator(store._hdr)
        store._allocator.init_all_free()
        # ★ Stage 6: 内容寻址开时初始化 hash 表 (零页非 EMPTY, 必须 clear)
        if store._content_addr:
            store._hdr.content_addr_init()
        return store

    @classmethod
    def open(cls, storage_path: str, num_slots: int,
             block_size: int) -> "LruArenaStore":
        """打开已有 hdr (其他进程已 create)."""
        store = cls(storage_path, num_slots, block_size)
        hdr_path = os.path.join(storage_path, "_arena.hdr")
        store._hdr = ArenaHdr.open(hdr_path, num_slots=num_slots)
        store._allocator = ArenaAllocator(store._hdr)
        # bitmap 已被 create 进程初始化, sync 当前真实计数
        store._allocator.sync_from_bitmap()
        return store

    @classmethod
    def open_or_create(cls, storage_path: str, num_slots: int,
                       block_size: int,
                       wait_timeout_s: float = 60.0) -> "LruArenaStore":
        """跨进程安全的初始化: 谁先到谁创建+init bitmap, 后到的 mmap+sync.

        与 prefill/decode 启动顺序无关. 推荐用这个代替 create()/open() 分别处理.
        flock 协议保证 creator 在临界区内完成 bitmap init, 不会有半态被后到进程
        看到.
        """
        store = cls(storage_path, num_slots, block_size)
        hdr_path = os.path.join(storage_path, "_arena.hdr")

        # ArenaHdr.open_or_create 的 on_create 回调里做 allocator.init_all_free,
        # 这样 bitmap init 也在 flock 临界区内完成
        def _on_create(hdr):
            tmp_allocator = ArenaAllocator(hdr)
            tmp_allocator.init_all_free()
            # ★ Stage 6: 内容寻址开时, 在 flock 临界区内 clear hash 表,
            # 保证后到进程看到的是已 init 的完整状态 (与 bitmap init 同步).
            if store._content_addr:
                hdr.content_addr_init()

        store._hdr = ArenaHdr.open_or_create(
            hdr_path, num_slots=num_slots,
            wait_timeout_s=wait_timeout_s,
            on_create=_on_create)
        store._allocator = ArenaAllocator(store._hdr)
        # bitmap 已被 creator init, sync 当前真实计数
        store._allocator.sync_from_bitmap()
        return store

    def close(self) -> None:
        if self._hdr is not None:
            self._hdr.close()
            self._hdr = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def bind_data_writer(self,
                         fn: Callable[[int, int, object], None]) -> None:
        """注入数据写入钩子.

        签名: fn(slot_id, block_idx_in_source, source_object) -> None
              用户实现把 source_object 的第 block_idx 个 block 拷贝到 arena slot_id
        """
        self._data_writer = fn

    # ============================================================
    # 属性
    # ============================================================
    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def storage_path(self) -> str:
        return self._storage_path

    @property
    def content_addr(self) -> bool:
        return self._content_addr

    @property
    def _ht_base(self) -> int:
        return self._hdr.hash_table_addr

    @property
    def _ht_cap(self) -> int:
        return self._hdr.hash_table_capacity

    def free_count(self) -> int:
        """快照空闲 slot 数. 仅在持 mutex 时严格准确."""
        return self._allocator.free_count

    def num_jobs(self) -> int:
        """当前 arena 中有 manifest 的 job 数."""
        return len(self._list_jobs())

    # ============================================================
    # 内部 helper: 路径 + 文件操作
    # ============================================================
    def _job_dir(self, job_id: str) -> str:
        return os.path.join(self._storage_path, _safe_job_id(job_id))

    def _manifest_path(self, job_id: str) -> str:
        return os.path.join(self._job_dir(job_id), _MANIFEST)

    def _slot_path(self, job_id: str, start: int, end: int) -> str:
        return os.path.join(self._job_dir(job_id), slot_filename(start, end))

    def _list_jobs(self) -> List[str]:
        """扫 storage_path 下所有 job 子目录."""
        try:
            entries = os.listdir(self._storage_path)
        except OSError:
            return []
        return [e for e in entries
                if not e.startswith(_RESERVED_DIR_PREFIXES)
                and os.path.isdir(os.path.join(self._storage_path, e))]

    def _list_incs(self, job_id: str) -> List[Tuple[int, int, str]]:
        """返回该 job 的 inc 列表, 按 start_block 升序: [(start, end, path), ...]."""
        d = self._job_dir(job_id)
        try:
            names = os.listdir(d)
        except OSError:
            return []
        out: List[Tuple[int, int, str]] = []
        for name in names:
            parsed = parse_slot_filename(name)
            if parsed is None:
                continue
            s, e = parsed
            out.append((s, e, os.path.join(d, name)))
        out.sort()
        return out

    def _read_slot_sg(self, path: str) -> Optional[List[Tuple[int, int]]]:
        """读 .slot 返回 [(slot_id, gen), ...], v1/v2 自适应 (lookup/load 用).

        v2 (content-addr) 多带 hash 字段, 这里丢弃只取 (slot, gen).
        损坏/不存在返回 None.
        """
        ver = read_slot_file_version(path)
        if ver == 2:
            sf2 = read_slot_file_v2(path)
            if sf2 is None:
                return None
            return [(s, g) for (s, g, _h) in sf2.records]
        sf1 = read_slot_file_v1(path)
        if sf1 is None:
            return None
        return sf1.records

    def _read_manifest(self, job_id: str) -> Optional[dict]:
        try:
            with open(self._manifest_path(job_id), "r") as f:
                return json.load(f)
        except (FileNotFoundError, ValueError):
            return None

    def _write_manifest(self, job_id: str, total_blocks: int,
                        token_ids: List[int]) -> None:
        d = self._job_dir(job_id)
        os.makedirs(d, exist_ok=True)
        payload = {"total_blocks": int(total_blocks),
                   "token_ids": [int(t) for t in token_ids]}
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".man_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self._manifest_path(job_id))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ============================================================
    # WRITE INC
    # ============================================================
    def write_inc(self,
                  job_id: str,
                  start_block: int,
                  end_block: int,
                  token_ids: List[int],
                  source_obj: object,
                  gpu_write_fn=None) -> bool:
        """把 [start_block, end_block) 这段 inc 写入 arena.

        参数:
            source_obj: 任何对象, 会被原样传给 data_writer.
                第 i 个 block (i ∈ [0, end-start)) 通过 data_writer(slot, i, source_obj) 写入.

        返回:
            True 成功, False 失败 (空间不够 / writer 未绑定)

        临界区设计 (修复死锁):
            mutex 临界区只做 alloc/evict (快, ~10us). 慢操作 (GPU memcpy +
            .slot/manifest 文件 IO) 全在锁外做. 把 4 个 store 线程 + prefill
            进程从争一把持锁 ~200ms 的锁, 缩到争 ~10us.

            正确性: alloc 后 slot 被 bitmap 标 used (其他进程 alloc 不会再拿),
            但 gen 还没 bump -> reader try_pin 会因 gen 不匹配而 miss, 看不到
            半写状态. 数据写完 + .slot 落盘后, 第二段临界区 bump gen 发布,
            此刻 reader 才能 pin 成功. manifest 最后写 (它是 lookup 的 LCP 依据,
            必须在 .slot 之后, 否则 lookup 命中但 .slot 还没落盘).
        """
        if end_block <= start_block:
            return True  # 无 inc 需要写
        n_blocks = end_block - start_block
        if self._data_writer is None:
            logger.warning("write_inc: data_writer not bound, skipping")
            return False

        # ★ Stage 6: 内容寻址 dedup 路径 (probe hash 表, 命中复用不新分配)
        if self._content_addr:
            return self._write_inc_dedup(
                job_id, start_block, end_block, token_ids, source_obj,
                gpu_write_fn=gpu_write_fn)

        # ---- 临界区 1: alloc (+ evict if needed), 拿到 slot_ids ----
        rc = _atomic.mutex_lock(self._hdr.mutex_addr)
        if rc == _atomic.errno_eownerdead():
            _atomic.mutex_recover(self._hdr.mutex_addr)
            self._allocator.sync_from_bitmap()
        elif rc != 0:
            logger.warning("write_inc: mutex_lock failed rc=%d", rc)
            return False
        try:
            slot_ids = self._allocator.alloc_n(n_blocks)
            if slot_ids is None:
                # 不够 n_blocks 个 free slot -> evict. need 用准确的实时 bitmap
                # 计数 (跨进程双 writer 下本地 free_count 缓存不准, 会把 need
                # 算错导致 evict 不足).
                accurate_free = self._allocator.count_free_accurate()
                need = n_blocks - accurate_free
                if not self._evict_until_free_locked(need,
                                                    exclude_job_id=job_id):
                    return False
                slot_ids = self._allocator.alloc_n(n_blocks)
                if slot_ids is None:
                    logger.warning(
                        "write_inc: alloc failed even after evict (job=%s n=%d "
                        "accurate_free=%d need=%d)",
                        job_id, n_blocks, accurate_free, need)
                    return False
        finally:
            _atomic.mutex_unlock(self._hdr.mutex_addr)

        # slot_ids 此刻已被 bitmap 标 used (其他进程不会再 alloc 到), 但 gen
        # 未 bump (reader 看不到). 下面慢操作全在锁外做.
        try:
            # ---- 锁外: 写数据 (GPU memcpy, 慢) ----
            for i, slot_id in enumerate(slot_ids):
                self._data_writer(slot_id, i, source_obj)

            # ---- 锁外: 准备 records (gen = 当前 gen + 1, 但还没发布) ----
            records: List[Tuple[int, int]] = []
            for slot_id in slot_ids:
                cur_gen = _atomic.get_gen(self._hdr.slot_state_addr(slot_id))
                records.append((slot_id, cur_gen + 1))

            # ---- 锁外: 写 .slot 文件 (IO) ----
            slot_path = self._slot_path(job_id, start_block, end_block)
            write_slot_file_v1(slot_path, records)
        except Exception as e:
            # 写失败: 回滚 — 把 slot 还回 free pool (临界区内), 不发布 gen
            logger.warning("write_inc: data/slot write failed job=%s: %s",
                           job_id, e)
            rc = _atomic.mutex_lock(self._hdr.mutex_addr)
            if rc in (0, _atomic.errno_eownerdead()):
                if rc == _atomic.errno_eownerdead():
                    _atomic.mutex_recover(self._hdr.mutex_addr)
                    self._allocator.sync_from_bitmap()
                try:
                    self._allocator.free_n(slot_ids)
                finally:
                    _atomic.mutex_unlock(self._hdr.mutex_addr)
            return False

        # ---- 临界区 2: 发布 gen (让 reader 可见), 快 ----
        rc = _atomic.mutex_lock(self._hdr.mutex_addr)
        if rc == _atomic.errno_eownerdead():
            _atomic.mutex_recover(self._hdr.mutex_addr)
            self._allocator.sync_from_bitmap()
        elif rc != 0:
            logger.warning("write_inc: mutex_lock(publish) failed rc=%d", rc)
            return False
        try:
            for (slot_id, new_gen) in records:
                _atomic.publish_slot(self._hdr.slot_state_addr(slot_id),
                                     new_gen)
            # in-memory LRU 状态在临界区内更新 (alloc_mutex 保护): 同进程多
            # store 线程争同一 mutex, 保证 _last_stored / _job_lru 不竞争.
            # (evict 的 self-heal 也在 alloc_mutex 内改这俩, 顺序一致.)
            self._last_stored[job_id] = end_block
            self._touch_job_lru(job_id)   # 移到 LRU 末尾 (最新)
        finally:
            _atomic.mutex_unlock(self._hdr.mutex_addr)

        # ---- 锁外: 重写 manifest (文件 IO, 不进临界区).
        # manifest 必须在 gen 发布之后写: lookup 用 manifest 的 token_ids 算
        # LCP, 然后校验每 inc 的 gen. gen 已发布, 所以 lookup 命中即可用. ----
        self._write_manifest(
            job_id, end_block, token_ids[:end_block * self._block_size])
        if self._debug:
            logger.info(
                "LRU-DBG write_inc job=%s [%d,%d) nslots=%d slot0=%d gen0=%d "
                "acc_free=%d",
                str(job_id)[:32], start_block, end_block, len(slot_ids),
                slot_ids[0], records[0][1],
                self._allocator.count_free_accurate())
        return True

    # ============================================================
    # ★ Stage 6: 内容寻址 dedup store
    # ============================================================
    def _write_inc_dedup(self,
                         job_id: str,
                         start_block: int,
                         end_block: int,
                         token_ids: List[int],
                         source_obj: object,
                         gpu_write_fn=None) -> bool:
        """内容寻址版 write_inc.

        与非 dedup 版同样的三段式临界区, 但每块先 probe hash 表:
          - HIT: refcnt++ 复用已有 slot, 不分配/不搬数据
          - MISS: alloc 新 slot + 插表 + refcnt=1, 锁外搬数据, CS2 publish

        正确性要点:
          - insert 必须在 CS1 内 (跨进程同 mutex 串行), 并发同内容写者出 CS1
            即看到 entry -> HIT, 不会给同内容重复分配 slot.
          - HIT refcnt++ 必须在 evict 之前做: 否则 evict 可能把某 HIT slot 的
            refcnt 减到 0 释放掉 (若它属于受害 job), 之后我们 inc 到一个已释放
            slot. 先 inc 保证 evict 时它 refcnt>=2 不会被淘.
          - gen 只在 MISS slot 上 publish; HIT slot 的 gen 已由原 owner 发布,
            不动 (其他引用者还在用).
        """
        n_blocks = end_block - start_block
        _prof = self._write_profile
        _tp = time.time() if _prof else 0.0
        _seg = {}
        # 链式 hash 需要整个前缀 [0,end) 来算 [start,end) 段
        all_hashes = block_hashes(token_ids, self._block_size, end_block)
        if _prof:
            _seg['hash'] = (time.time() - _tp) * 1000.0; _tp = time.time()
        if len(all_hashes) < end_block:
            logger.warning(
                "dedup write_inc: token_ids 不足 job=%s end=%d have=%d",
                str(job_id)[:32], end_block, len(all_hashes))
            return False
        inc_hashes = all_hashes[start_block:end_block]
        ht_base, ht_cap = self._ht_base, self._ht_cap

        # 每块计划: [kind('H'/'M'), slot_id, hash]
        plan: List[list] = [['?', -1, h] for h in inc_hashes]
        # ★ 淘汰 self-heal 延迟收集 (job_id -> 最小 s): 全量 manifest 重写是淘汰里
        # 最贵的一段 (O(token) JSON, 锁内 ×N inc = 22s 主因). 收集到这里, CS1 解锁
        # 后锁外按 job 各重写一次, 把它移出跨进程临界区.
        evict_deferred: dict = {}

        # ---- 临界区 1: probe + HIT refcnt++ + MISS alloc/insert ----
        rc = _atomic.mutex_lock(self._hdr.mutex_addr)
        if _prof:
            _seg['cs1_lockwait'] = (time.time() - _tp) * 1000.0; _tp = time.time()
        if rc == _atomic.errno_eownerdead():
            _atomic.mutex_recover(self._hdr.mutex_addr)
            self._allocator.sync_from_bitmap()
        elif rc != 0:
            logger.warning("dedup write_inc: mutex_lock failed rc=%d", rc)
            return False
        hit_inced: List[int] = []   # 已 refcnt++ 的 HIT slot (回滚用)
        try:
            miss_is: List[int] = []
            for i, h in enumerate(inc_hashes):
                slot, _g = _atomic.ht_probe(ht_base, ht_cap, h)
                if slot >= 0:
                    plan[i][0] = 'H'
                    plan[i][1] = slot
                else:
                    plan[i][0] = 'M'
                    miss_is.append(i)
            # HIT 先 refcnt++ (保护其不被下面 evict 淘掉)
            for i in range(n_blocks):
                if plan[i][0] == 'H':
                    _atomic.refcnt_inc(
                        self._hdr.slot_refcnt_addr(plan[i][1]))
                    hit_inced.append(plan[i][1])
            # MISS alloc (+ evict)
            n_miss = len(miss_is)
            if n_miss > 0:
                slot_ids = self._allocator.alloc_n(n_miss)
                if slot_ids is None:
                    accurate_free = self._allocator.count_free_accurate()
                    need = n_miss - accurate_free
                    if not self._evict_until_free_locked(
                            need, exclude_job_id=job_id,
                            deferred=evict_deferred):
                        for s in hit_inced:
                            _atomic.refcnt_dec(self._hdr.slot_refcnt_addr(s))
                        return False
                    slot_ids = self._allocator.alloc_n(n_miss)
                    if slot_ids is None:
                        for s in hit_inced:
                            _atomic.refcnt_dec(self._hdr.slot_refcnt_addr(s))
                        logger.warning(
                            "dedup write_inc: alloc failed after evict "
                            "(job=%s n_miss=%d)", str(job_id)[:32], n_miss)
                        return False
                # MISS: 绑 slot + refcnt=1 + 插表 (插表在 CS1, 防并发重复分配)
                for k, i in enumerate(miss_is):
                    s = slot_ids[k]
                    plan[i][1] = s
                    _atomic.refcnt_set(self._hdr.slot_refcnt_addr(s), 1)
                    _atomic.ht_insert(ht_base, ht_cap, plan[i][2], s)
        finally:
            _atomic.mutex_unlock(self._hdr.mutex_addr)
        if _prof:
            _seg['cs1_work'] = (time.time() - _tp) * 1000.0; _tp = time.time()
        # ★ 锁外: 批量 self-heal (淘汰收集的 job->min_s), 每 job 重写一次 manifest.
        # 已出临界区, 不再阻塞别的 store/alloc. 内存 _last_stored 已在锁内更新过.
        for _hj, _hs in evict_deferred.items():
            self._rewrite_manifest_for_self_heal(_hj, _hs)

        # ---- 锁外: 只对 MISS 搬数据 + 建 v2 records (慢) ----
        try:
            if gpu_write_fn is not None:
                # ★ store-direct: GPU kernel 直写 (paged -> arena), 一次批量, 内部
                # wait_event(forward) + sync, 省掉 D2H gather + 逐块 CPU memcpy.
                # 传 MISS 的 (arena slot, inc 内位置), 调用方据位置映射 paged block id.
                _miss_slots = [plan[i][1] for i in range(n_blocks)
                               if plan[i][0] == 'M']
                _miss_pos = [i for i in range(n_blocks)
                             if plan[i][0] == 'M']
                gpu_write_fn(_miss_slots, _miss_pos)
            else:
                for i in range(n_blocks):
                    if plan[i][0] == 'M':
                        self._data_writer(plan[i][1], i, source_obj)
            if _prof:
                _seg['data'] = (time.time() - _tp) * 1000.0; _tp = time.time()
            # records: (slot, gen, hash). HIT gen=当前; MISS gen=当前+1(待发布)
            records: List[Tuple[int, int, int]] = []
            miss_pub: List[Tuple[int, int, int]] = []  # (slot,new_gen,hash) 待发布
            for i in range(n_blocks):
                kind, slot, h = plan[i][0], plan[i][1], plan[i][2]
                cur_gen = _atomic.get_gen(self._hdr.slot_state_addr(slot))
                if kind == 'M':
                    new_gen = cur_gen + 1
                    records.append((slot, new_gen, h))
                    miss_pub.append((slot, new_gen, h))
                else:
                    records.append((slot, cur_gen, h))
            slot_path = self._slot_path(job_id, start_block, end_block)
            write_slot_file_v2(slot_path, records)
            if _prof:
                _seg['slot'] = (time.time() - _tp) * 1000.0; _tp = time.time()
        except Exception as e:
            logger.warning("dedup write_inc: data/slot write failed job=%s: %s",
                           str(job_id)[:32], e)
            self._rollback_dedup(hit_inced, plan)
            return False

        # ---- 临界区 2: 只 publish MISS slot, 更新 LRU ----
        rc = _atomic.mutex_lock(self._hdr.mutex_addr)
        if rc == _atomic.errno_eownerdead():
            _atomic.mutex_recover(self._hdr.mutex_addr)
            self._allocator.sync_from_bitmap()
        elif rc != 0:
            logger.warning("dedup write_inc: mutex_lock(publish) failed rc=%d",
                           rc)
            return False
        try:
            for (slot, new_gen, h) in miss_pub:
                _atomic.publish_slot(self._hdr.slot_state_addr(slot), new_gen)
                # ★ 把发布 gen 写进 hash entry, 供跨 job load 的 try_pin 校验
                _atomic.ht_set_gen(ht_base, ht_cap, h, new_gen)
            self._last_stored[job_id] = end_block
            self._touch_job_lru(job_id)
        finally:
            _atomic.mutex_unlock(self._hdr.mutex_addr)
        if _prof:
            _seg['cs2'] = (time.time() - _tp) * 1000.0; _tp = time.time()

        # ---- 锁外: 重写 manifest ----
        self._write_manifest(
            job_id, end_block, token_ids[:end_block * self._block_size])
        if _prof:
            _seg['manifest'] = (time.time() - _tp) * 1000.0
            _nm = len(miss_pub)
            logger.info(
                "WRITE-PROF job=%s nblk=%d miss=%d | hash=%.0f lockwait=%.0f "
                "cs1=%.0f data=%.0f(%.2fms/blk) slot=%.0f cs2=%.0f man=%.0f",
                str(job_id)[:24], n_blocks, _nm, _seg.get('hash', 0),
                _seg.get('cs1_lockwait', 0), _seg.get('cs1_work', 0),
                _seg.get('data', 0),
                (_seg.get('data', 0) / _nm) if _nm else 0.0,
                _seg.get('slot', 0), _seg.get('cs2', 0),
                _seg.get('manifest', 0))

        n_hit = n_blocks - len(miss_pub)
        self._stat_hit_blocks += n_hit
        self._stat_miss_blocks += len(miss_pub)
        if self._debug:
            logger.info(
                "LRU-DBG dedup write_inc job=%s [%d,%d) hit=%d miss=%d "
                "acc_free=%d (cum hit=%d miss=%d)",
                str(job_id)[:32], start_block, end_block, n_hit, len(miss_pub),
                self._allocator.count_free_accurate(),
                self._stat_hit_blocks, self._stat_miss_blocks)
        return True

    def _rollback_dedup(self, hit_inced: List[int], plan: List[list]) -> None:
        """dedup write_inc 锁外阶段失败的回滚 (临界区内): HIT refcnt--,
        MISS 删表 + free slot (gen 未发布, 无需动)."""
        rc = _atomic.mutex_lock(self._hdr.mutex_addr)
        if rc not in (0, _atomic.errno_eownerdead()):
            return
        if rc == _atomic.errno_eownerdead():
            _atomic.mutex_recover(self._hdr.mutex_addr)
            self._allocator.sync_from_bitmap()
        try:
            for s in hit_inced:
                _atomic.refcnt_dec(self._hdr.slot_refcnt_addr(s))
            miss_slots = [p[1] for p in plan if p[0] == 'M' and p[1] >= 0]
            for i, p in enumerate(plan):
                if p[0] == 'M' and p[1] >= 0:
                    _atomic.ht_remove(self._ht_base, self._ht_cap, p[2])
                    _atomic.refcnt_set(self._hdr.slot_refcnt_addr(p[1]), 0)
            if miss_slots:
                self._allocator.free_n(miss_slots)
        finally:
            _atomic.mutex_unlock(self._hdr.mutex_addr)

    # ============================================================
    # LOOKUP (lock-free)
    # ============================================================
    def lookup(self, job_id: str,
               prompt_token_ids: List[int]) -> Optional[Tuple[int, int]]:
        """Lookup 这个 job 的 manifest + .slot 文件, 校验 gen 后返回可复用前缀.

        返回 (matched_tokens, matched_blocks); 无命中返回 None.
        """
        manifest = self._read_manifest(job_id)
        if not manifest:
            return None
        stored = manifest.get("token_ids") or []
        if not stored:
            return None
        n_cmp = min(len(stored), len(prompt_token_ids))
        lcp = 0
        for i in range(n_cmp):
            if stored[i] != prompt_token_ids[i]:
                break
            lcp += 1
        matched_blocks_by_token = align_blocks(lcp, self._block_size)
        if matched_blocks_by_token <= 0:
            return None
        matched_blocks_by_token = min(matched_blocks_by_token,
                                      int(manifest.get("total_blocks", 0)))
        if matched_blocks_by_token <= 0:
            return None

        # 按 inc 顺序校验 gen
        valid_blocks = 0
        for (s, e, path) in self._list_incs(job_id):
            if s != valid_blocks:
                break   # coverage gap (跨 inc 不连续)
            recs = self._read_slot_sg(path)
            if recs is None or len(recs) != (e - s):
                break
            ok = True
            for (slot_id, expected_gen) in recs:
                cur_gen = _atomic.get_gen(
                    self._hdr.slot_state_addr(slot_id))
                if cur_gen != expected_gen:
                    ok = False
                    break
            if not ok:
                break
            valid_blocks = e
            if valid_blocks >= matched_blocks_by_token:
                break

        usable_blocks = min(valid_blocks, matched_blocks_by_token)
        if usable_blocks <= 0:
            return None
        return usable_blocks * self._block_size, usable_blocks

    # ============================================================
    # LOAD (lock-free, pin-based)
    # ============================================================
    def load_request(self,
                     job_id: str,
                     dst_block_ids: List[int],
                     src_block_offset: int) -> Optional[LoadHandle]:
        """获取 [src_block_offset, src_block_offset + len(dst_block_ids)) 这段
        block 对应的 slot, 通过 pin 锁定使其在 load 期间不被 evict.

        成功: 返回 LoadHandle, 调用者用 slot_ids 搬数据后必须 release().
        miss / race: 返回 None.
        """
        n = len(dst_block_ids)
        if n == 0:
            return LoadHandle([], [], [], [])
        lo = int(src_block_offset)
        hi = lo + n
        if lo < 0:
            return None

        # 收集 [lo, hi) 范围内的 (slot_id, gen) 记录
        all_records: List[Tuple[int, int]] = []
        all_dst: List[int] = []
        covered = lo
        for (s, e, path) in self._list_incs(job_id):
            if e <= lo or s >= hi:
                continue
            if s > covered:
                return None   # coverage gap
            recs = self._read_slot_sg(path)
            if recs is None or len(recs) != (e - s):
                return None
            a = max(s, lo)
            b = min(e, hi)
            # inc 内 block 下标 [a-s, b-s)
            for blk_in_inc in range(a - s, b - s):
                all_records.append(recs[blk_in_inc])
            for blk_in_dst in range(a - lo, b - lo):
                all_dst.append(dst_block_ids[blk_in_dst])
            covered = b
        if covered < hi:
            return None

        # 逐 slot try_pin; 任一失败回滚
        state_addrs: List[int] = []
        slot_ids: List[int] = []
        gens: List[int] = []
        pinned_addrs: List[int] = []
        for (slot_id, expected_gen) in all_records:
            addr = self._hdr.slot_state_addr(slot_id)
            if not _atomic.try_pin(addr, expected_gen):
                # 回滚
                for a in pinned_addrs:
                    _atomic.unpin(a)
                return None
            pinned_addrs.append(addr)
            state_addrs.append(addr)
            slot_ids.append(slot_id)
            gens.append(expected_gen)

        return LoadHandle(
            slot_state_addrs=state_addrs,
            slot_ids=slot_ids,
            gens=gens,
            dst_block_ids=all_dst,
        )

    def load_batch_pin(self, items: list) -> "BatchLoadHandle":
        """Batch 版 load: 解析一整波 admit 请求, 把所有 block 的
        (slot_id, gen, dst_block) 收集到一起并逐 slot try_pin.

        items: [(job_id, dst_block_ids, src_block_offset), ...]

        返回 BatchLoadHandle, 含:
          - per_item_ok: [bool] 每个请求是否成功 (miss/race -> False)
          - slot_ids / dst_block_ids / gens: 所有成功请求的 block 展平 (同序)
          - state_addrs: 已 pin 的 slot_state 地址 (release 用)
        调用者用 slot_ids + dst_block_ids 喂直读 kernel, 完成后 release().

        设计: 一次解析整波 -> 一次性把 src_slots/dst 交给 GPU kernel, 避免逐
        请求的 Python/CUDA 往返. pin 保证 load 期间 evict 不动这些 slot.
        """
        n_items = len(items)
        per_item_ok = [False] * n_items
        all_slot_ids: List[int] = []
        all_dst: List[int] = []
        all_gens: List[int] = []
        all_addrs: List[int] = []
        for k, item in enumerate(items):
            # item 可为 3 元组 (job_id, dst, src_offset) [own-job, 查 .slot],
            # 或 4 元组 (..., slot_gen_list) [★ 跨 job, lookup_resolve 已解析好的
            # (slot,gen) 列表, 直接 load_pin_explicit].
            job_id, dst_block_ids, src_block_offset = item[0], item[1], item[2]
            slot_gen = item[3] if len(item) > 3 else None
            if slot_gen:
                handle = self.load_pin_explicit(
                    list(slot_gen), list(dst_block_ids))
            else:
                handle = self.load_request(
                    str(job_id), list(dst_block_ids), int(src_block_offset))
            if handle is None:
                continue
            per_item_ok[k] = True
            all_slot_ids.extend(handle.slot_ids)
            all_dst.extend(handle.dst_block_ids)
            all_gens.extend(handle.gens)
            all_addrs.extend(handle.slot_state_addrs)
        if self._debug:
            hit = sum(1 for ok in per_item_ok if ok)
            logger.info(
                "LRU-DBG load_batch_pin reqs=%d hit=%d miss=%d pinned_blocks=%d",
                n_items, hit, n_items - hit, len(all_slot_ids))
        return BatchLoadHandle(
            per_item_ok=per_item_ok,
            slot_ids=all_slot_ids,
            dst_block_ids=all_dst,
            gens=all_gens,
            slot_state_addrs=all_addrs,
        )

    # ============================================================
    # ★ Stage 6c: 跨 job 表驱动 lookup + 显式 slot load
    # ============================================================
    def lookup_resolve(self, prompt_token_ids: List[int]
                       ) -> Optional[Tuple[int, int, List[Tuple[int, int]]]]:
        """跨 job 表驱动 lookup (job 无关, 无锁).

        对 prompt 逐块链式 hash → ht_probe → 校验后连续命中计数. 因为任何 job
        store 时都把块插了表, 这条同时覆盖 own-job 与 cross-job (谁存的都能命中).

        校验一块 (slot, egen) 有效:
          - slot >= 0 (probe 命中)
          - egen != UNPUB(0)  (已发布, 非半写)
          - slot_state.gen == egen  (slot 未被淘汰复用)
          - refcnt > 0  (仍被引用)

        返回 (matched_tokens, matched_blocks, slot_gen_list) 或 None.
        slot_gen_list 是匹配前缀逐块的 (slot_id, gen), 供 load_pin_explicit
        直接 pin+scatter (跨进程下由调用方经元数据传给 worker, 免得 worker 重算).

        注意: 本函数不 pin (只读探测). lookup→load 之间 slot 可能被淘, 由
        load_pin_explicit 的 try_pin(gen) fail-closed 兜底.
        """
        bs = self._block_size
        n_full = len(prompt_token_ids) // bs
        if n_full <= 0:
            return None

        # ★ C 快路径: 整条循环 (链式 hash + probe + gen/refcnt 校验 + 连续截断)
        # 一个 C 调用跑完, 替掉 Python 逐块 (800 块=几千次跨语言调用 ~37ms → ~1ms).
        if _HAS_C_LOOKUP:
            n_tok = n_full * bs
            tok_bytes = struct.pack("<%dI" % n_tok,
                                    *prompt_token_ids[:n_tok])
            nb, slots, gens = _atomic.lookup_resolve(
                self._ht_base, self._ht_cap,
                self._hdr.slot_state_addr(0),
                self._hdr.slot_refcnt_addr(0),
                self._num_slots, tok_bytes, bs, SEED0)
            if nb <= 0:
                return None
            return nb * bs, nb, list(zip(slots, gens))

        # Python fallback (旧 .so / 无 C lookup_resolve)
        hashes = block_hashes(prompt_token_ids, bs, n_full)
        slot_gen: List[Tuple[int, int]] = []
        for i in range(n_full):
            slot, egen = _atomic.ht_probe(self._ht_base, self._ht_cap, hashes[i])
            if slot < 0 or egen == 0:
                break
            if _atomic.get_gen(self._hdr.slot_state_addr(slot)) != egen:
                break   # slot 被淘汰/复用, gen 不匹配
            if _atomic.refcnt_get(self._hdr.slot_refcnt_addr(slot)) == 0:
                break   # 已无引用
            slot_gen.append((slot, egen))
        nb = len(slot_gen)
        if nb <= 0:
            return None
        return nb * bs, nb, slot_gen

    def load_pin_explicit(self, slot_gen_list: List[Tuple[int, int]],
                          dst_block_ids: List[int]) -> Optional[LoadHandle]:
        """按显式 (slot, gen) 列表 pin (跨 job load).

        不查 .slot / 不查表 (lookup_resolve 已解析好), 直接对每个 slot
        try_pin(gen). 任一失败 (gen 变 = 被淘) 回滚返回 None. 成功返回 LoadHandle,
        调用者 scatter 完必须 release().

        slot_gen_list 取前 len(dst_block_ids) 个 (与 dst 一一对应).
        """
        n = len(dst_block_ids)
        if n == 0:
            return LoadHandle([], [], [], [])
        if len(slot_gen_list) < n:
            return None
        state_addrs: List[int] = []
        slot_ids: List[int] = []
        gens: List[int] = []
        pinned: List[int] = []
        for k in range(n):
            slot, egen = slot_gen_list[k]
            addr = self._hdr.slot_state_addr(slot)
            if not _atomic.try_pin(addr, egen):
                for a in pinned:
                    _atomic.unpin(a)
                return None
            pinned.append(addr)
            state_addrs.append(addr)
            slot_ids.append(slot)
            gens.append(egen)
        return LoadHandle(
            slot_state_addrs=state_addrs,
            slot_ids=slot_ids,
            gens=gens,
            dst_block_ids=list(dst_block_ids),
        )

    # ============================================================
    # EVICT (LRU + tail-first + self-heal)
    # ============================================================
    def _evict_until_free_locked(self, need: int,
                                 exclude_job_id: Optional[str] = None,
                                 deferred: Optional[dict] = None) -> bool:
        """在 alloc_mutex 内调用. 释放至少 need 个 slot.

        exclude_job_id: 写新 inc 的 job 自己不应被淘 (避免自淘)
        deferred: 非 None 时把 self-heal (job->min_s) 收集到此, 由调用方锁外批量
                  执行 (把全量 manifest 重写移出临界区). None 则锁内 inline (旧行为).
        返回: True 成功 / False 失败 (即便挖光 LRU 也不够)
        """
        if need <= 0:
            return True
        gained = 0
        # 单遍: 内存 LRU 快照 (最老在前) 逐 victim, O(jobs) (旧版每轮重扫 ~O(jobs²)).
        # 快照 list(): _evict_job_tail_first 可能 _drop_job_lru 改动 _job_lru.
        seen: set[str] = set()
        for victim in list(self._job_lru):
            if gained >= need:
                break
            if exclude_job_id is not None and victim == exclude_job_id:
                continue
            seen.add(victim)
            gained += self._evict_job_tail_first(victim, need - gained,
                                                 deferred)
        # 单遍不够 (victim 多被 pin / 内存 LRU 不全, 罕见): 兜底反复挑 (含文件系统)
        attempted: set[str] = set(seen)
        if exclude_job_id is not None:
            attempted.add(exclude_job_id)
        while gained < need:
            victim = self._pick_lru_victim(exclude=attempted)
            if victim is None:
                logger.warning("evict_until_free: no victim job found, "
                               "gained=%d need=%d", gained, need)
                return False
            attempted.add(victim)
            gained += self._evict_job_tail_first(victim, need - gained,
                                                 deferred)
        return True

    def _pick_lru_victim(self,
                        exclude: Iterable[str]) -> Optional[str]:
        """返回最老的 job (不在 exclude 内).

        优先用内存 LRU (_job_lru, OrderedDict, 最老在前) — O(待排除数), 不碰
        文件系统. 这是临界区内 evict 的热路径, 必须快 (旧版扫 os.listdir +
        每 job os.stat manifest mtime, 写满后 O(jobs) 文件 IO 在锁内塌方).

        内存 LRU 选不出时 (本进程没 store 过任何 job, 但 bitmap 满 — 例如
        全是对方进程写的), 兜底扫一次文件系统 mtime.
        """
        exclude_set = set(exclude)
        # 热路径: 内存 LRU, 最老在前
        for jid in self._job_lru:
            if jid not in exclude_set:
                return jid
        # 兜底: 文件系统扫描 (罕见 — 本进程内存 LRU 为空但仍需 evict)
        candidates: List[Tuple[float, str]] = []
        for job in self._list_jobs():
            if job in exclude_set:
                continue
            try:
                mtime = os.stat(self._manifest_path(job)).st_mtime
            except FileNotFoundError:
                continue
            candidates.append((mtime, job))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def _touch_job_lru(self, job_id: str) -> None:
        """write_inc 成功后调用: 把 job 移到 LRU 末尾 (最新)."""
        self._job_lru[job_id] = None
        self._job_lru.move_to_end(job_id)

    def _drop_job_lru(self, job_id: str) -> None:
        self._job_lru.pop(job_id, None)

    def _evict_job_tail_first(self, job_id: str,
                              max_need: int,
                              deferred: Optional[dict] = None) -> int:
        """从该 job 尾巴 inc 开始释放 slot, 直到释放够 max_need 或 inc 用完.

        返回实际释放数.

        并发安全 (配合缩小后的 write_inc 临界区): write_inc 先写 .slot 文件,
        最后才写 manifest. 一个 end > manifest.total_blocks 的 inc 说明它正在
        被某个 writer 写 (还没提交). 我们用 manifest.total_blocks 作为"已提交
        边界", 只 evict end <= committed 的 inc, 跳过未提交的 inc, 避免淘到正在
        写的 slot.
        """
        incs = self._list_incs(job_id)
        if not incs:
            return 0
        # 已提交边界: 没有 manifest 视为 0 (整个 job 都是未提交的, 全跳过)
        manifest = self._read_manifest(job_id)
        committed = int(manifest.get("total_blocks", 0)) if manifest else 0
        released = 0
        # 反向遍历 (tail first)
        for (s, e, path) in reversed(incs):
            if released >= max_need:
                break
            if e > committed:
                # 未提交的 inc (正在被 writer 写), 跳过, 不能淘
                continue
            if self._content_addr:
                # ---- ★ content-addr: refcnt--, 到 0 才真销毁 ----
                released += self._evict_inc_content(job_id, s, e, path,
                                                    deferred)
            else:
                # ---- 原始 plain 路径 (非 content-addr), 行为不变 ----
                sf = read_slot_file_v1(path)
                if sf is None:
                    # 损坏的 .slot, 直接删
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                    continue
                slots_freed_here: List[int] = []
                slots_left_pinned: List[int] = []
                for (slot_id, _gen) in sf.records:
                    # 已经 free 的 slot 跳过 (防止跨进程/stale LRU 重复 free 把
                    # free_count 加错: 对方进程可能已 evict 过这个 job 的 slot).
                    if self._allocator.is_free(slot_id):
                        continue
                    addr = self._hdr.slot_state_addr(slot_id)
                    if _atomic.can_evict(addr):
                        _atomic.evict_slot(addr)
                        slots_freed_here.append(slot_id)
                    else:
                        slots_left_pinned.append(slot_id)
                if slots_freed_here:
                    self._allocator.free_n(slots_freed_here)
                    released += len(slots_freed_here)
                    # 同步回退 _last_stored (self-heal)
                    cur_last = self._last_stored.get(job_id, e)
                    if s < cur_last:
                        self._last_stored[job_id] = s
                        # 同步回退 manifest 的 total_blocks
                        self._rewrite_manifest_for_self_heal(job_id, s)
                    # 如果 inc 内还有 pinned slot, 不删 .slot 文件
                    # (next lookup 校验 gen 时自然发现 gap)
                    if not slots_left_pinned:
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
        # 如果该 job 已被淘空 (没有 .slot 文件了), 从内存 LRU + 文件系统清理.
        # 关键: 无条件检查 (即使 released==0). 跨进程场景下对方可能已把这个
        # job evict 空了, 本进程 _job_lru 仍有它 (空壳). 选到空壳时 released=0,
        # 但仍必须清理, 否则空壳永远留在 _job_lru 最前 (最老), 每次 evict 又
        # 选到它 -> _list_incs/_read_manifest 文件 IO 空转 -> O(空壳) 性能退化.
        if not self._list_incs(job_id):
            self._drop_job_lru(job_id)
            self._last_stored.pop(job_id, None)
            try:
                mp = self._manifest_path(job_id)
                if os.path.exists(mp):
                    os.unlink(mp)
                os.rmdir(self._job_dir(job_id))
            except OSError:
                pass
        if self._debug and released > 0:
            logger.info("LRU-DBG evict victim=%s freed=%d acc_free=%d",
                        str(job_id)[:32], released,
                        self._allocator.count_free_accurate())
        return released

    def _evict_inc_content(self, job_id: str, s: int, e: int,
                           path: str, deferred: Optional[dict] = None) -> int:
        """★ content-addr 版单 inc 淘汰 (在 alloc_mutex 内调用).

        对该 inc 的每块 refcnt--; 减到 0 且 pin==0 才真销毁 (bump gen + 删表 +
        free). refcnt>0 的块仅摘本 job 引用, 数据留给别的 job. pin>0 的块跳过
        留下轮. 返回真正释放的 slot 数 (用于 evict 的 gained 计数).

        self-heal + unlink: 当本 job 对该 inc 的引用被完全摘除 (无 pinned 残留)
        即触发, 即使 freed==0 (全是 refcnt-- 未到 0) —— 因为 manifest/.slot 被
        删后该 job 逻辑上不再持有这段, 下轮须从 s 续 store.
        """
        sf2 = read_slot_file_v2(path)
        if sf2 is None:
            try:
                os.unlink(path)
            except OSError:
                pass
            return 0
        freed = 0
        left_pinned = 0
        for (slot_id, _gen, h) in sf2.records:
            if self._allocator.is_free(slot_id):
                continue   # 已被释放 (别的引用者把 refcnt 减到 0)
            addr = self._hdr.slot_state_addr(slot_id)
            if not _atomic.can_evict(addr):
                left_pinned += 1
                continue   # pinned, 不能动, 留下轮
            # 校验该 slot 仍持有这个 hash 的内容 (防 slot 被淘后复用给别 hash
            # 时误减别人的 refcnt). 正常 refcnt 记账下恒成立, 此为纵深防御.
            probe_slot, _g = _atomic.ht_probe(self._ht_base, self._ht_cap, h)
            if probe_slot != slot_id:
                continue
            new_rc = _atomic.refcnt_dec(self._hdr.slot_refcnt_addr(slot_id))
            if new_rc == 0:
                _atomic.evict_slot(addr)
                _atomic.ht_remove(self._ht_base, self._ht_cap, h)
                self._allocator.free_n([slot_id])
                freed += 1
            else:
                # refcnt>0, 数据留给别的 job, 仅摘本 job 引用
                self._stat_evict_decref += 1
        self._stat_evict_freed += freed
        if self._debug and (freed or left_pinned):
            logger.info(
                "LRU-DBG dedup evict-inc job=%s [%d,%d) freed=%d pinned=%d "
                "(cum freed=%d decref=%d)",
                str(job_id)[:32], s, e, freed, left_pinned,
                self._stat_evict_freed, self._stat_evict_decref)
        if left_pinned == 0:
            cur_last = self._last_stored.get(job_id, e)
            if s < cur_last:
                self._last_stored[job_id] = s   # 内存进度 (锁内, 便宜) 立即更新
                if deferred is not None:
                    # 延迟: 收集 job 最小 s, 调用方锁外按 job 重写一次 manifest
                    prev = deferred.get(job_id)
                    deferred[job_id] = s if prev is None else min(prev, s)
                else:
                    self._rewrite_manifest_for_self_heal(job_id, s)
            try:
                os.unlink(path)
            except OSError:
                pass
        return freed

    def _rewrite_manifest_for_self_heal(self, job_id: str,
                                         new_total_blocks: int) -> None:
        """Self-heal: 回退 manifest.total_blocks. token_ids 也截到对应 token."""
        manifest = self._read_manifest(job_id)
        if manifest is None:
            return
        cur_total = int(manifest.get("total_blocks", 0))
        if new_total_blocks >= cur_total:
            return  # 不需要回退
        new_tok_len = new_total_blocks * self._block_size
        token_ids = manifest.get("token_ids") or []
        new_tokens = token_ids[:new_tok_len]
        self._write_manifest(job_id, new_total_blocks, new_tokens)

    # ============================================================
    # DELETE / FINISH
    # ============================================================
    def delete_job(self, job_id: str) -> None:
        """彻底删除 job: 释放所有 slot, 删目录."""
        rc = _atomic.mutex_lock(self._hdr.mutex_addr)
        if rc == _atomic.errno_eownerdead():
            _atomic.mutex_recover(self._hdr.mutex_addr)
            self._allocator.sync_from_bitmap()
        elif rc != 0:
            return
        try:
            for (_s, _e, path) in self._list_incs(job_id):
                sf = read_slot_file_v1(path)
                if sf is not None:
                    slot_ids_freed = []
                    for (slot_id, _) in sf.records:
                        # 跳过已 free 的 (防重复 free)
                        if self._allocator.is_free(slot_id):
                            continue
                        addr = self._hdr.slot_state_addr(slot_id)
                        if _atomic.can_evict(addr):
                            _atomic.evict_slot(addr)
                            slot_ids_freed.append(slot_id)
                    if slot_ids_freed:
                        self._allocator.free_n(slot_ids_freed)
                try:
                    os.unlink(path)
                except OSError:
                    pass
            d = self._job_dir(job_id)
            try:
                if os.path.exists(self._manifest_path(job_id)):
                    os.unlink(self._manifest_path(job_id))
                os.rmdir(d)
            except OSError:
                pass
            self._last_stored.pop(job_id, None)
            self._drop_job_lru(job_id)
        finally:
            _atomic.mutex_unlock(self._hdr.mutex_addr)

    def mark_finished_job(self, job_id: str) -> None:
        """Stage 2 简化: 直接 delete. (不做 Stage A 那种 finished 集合)"""
        self.delete_job(job_id)
