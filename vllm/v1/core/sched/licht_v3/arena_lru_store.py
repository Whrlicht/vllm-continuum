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
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import licht_arena_atomic as _atomic

from vllm.v1.core.sched.licht_v3.arena_allocator import ArenaAllocator
from vllm.v1.core.sched.licht_v3.arena_hdr import ArenaHdr
from vllm.v1.core.sched.licht_v3.arena_slot_file import (
    SlotFileV1,
    parse_slot_filename,
    read_slot_file_v1,
    slot_filename,
    write_slot_file_v1,
)


logger = logging.getLogger(__name__)

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
                  source_obj: object) -> bool:
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
                need = n_blocks - self._allocator.free_count
                if not self._evict_until_free_locked(need,
                                                    exclude_job_id=job_id):
                    return False
                slot_ids = self._allocator.alloc_n(n_blocks)
                if slot_ids is None:
                    logger.warning(
                        "write_inc: alloc failed even after evict (job=%s n=%d)",
                        job_id, n_blocks)
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
        finally:
            _atomic.mutex_unlock(self._hdr.mutex_addr)

        # ---- 锁外: 重写 manifest (mtime 自动更新 → LRU 提权).
        # manifest 必须在 gen 发布之后写: lookup 用 manifest 的 token_ids 算
        # LCP, 然后校验每 inc 的 gen. gen 已发布, 所以 lookup 命中即可用. ----
        self._write_manifest(
            job_id, end_block, token_ids[:end_block * self._block_size])
        self._last_stored[job_id] = end_block
        self._touch_job_lru(job_id)   # 移到 LRU 末尾 (最新)
        return True

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
            sf = read_slot_file_v1(path)
            if sf is None or len(sf.records) != (e - s):
                break
            ok = True
            for (slot_id, expected_gen) in sf.records:
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
            sf = read_slot_file_v1(path)
            if sf is None or len(sf.records) != (e - s):
                return None
            a = max(s, lo)
            b = min(e, hi)
            # inc 内 block 下标 [a-s, b-s)
            for blk_in_inc in range(a - s, b - s):
                all_records.append(sf.records[blk_in_inc])
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

    # ============================================================
    # EVICT (LRU + tail-first + self-heal)
    # ============================================================
    def _evict_until_free_locked(self, need: int,
                                 exclude_job_id: Optional[str] = None) -> bool:
        """在 alloc_mutex 内调用. 释放至少 need 个 slot.

        exclude_job_id: 写新 inc 的 job 自己不应被淘 (避免自淘)
        返回: True 成功 / False 失败 (即便挖光 LRU 也不够)
        """
        if need <= 0:
            return True
        gained = 0
        # 反复挑 victim, 直到够 need
        attempted: set[str] = set()
        while gained < need:
            victim = self._pick_lru_victim(
                exclude={exclude_job_id} | attempted
                if exclude_job_id else attempted)
            if victim is None:
                logger.warning("evict_until_free: no victim job found, "
                               "gained=%d need=%d", gained, need)
                return False
            attempted.add(victim)
            v_gained = self._evict_job_tail_first(victim,
                                                  need - gained)
            gained += v_gained
            if v_gained == 0:
                # 这个 victim 全部被 pin 住, 跳到下一个
                continue
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
                              max_need: int) -> int:
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
        # 关键: 不清理的话, 空壳 job 留在 _job_lru 最前 (最老), 下次 evict 又
        # 选到它, _list_incs + _read_manifest 文件 IO 空转 -> O(空壳) 退化.
        if released > 0 and not self._list_incs(job_id):
            self._drop_job_lru(job_id)
            self._last_stored.pop(job_id, None)
            try:
                mp = self._manifest_path(job_id)
                if os.path.exists(mp):
                    os.unlink(mp)
                os.rmdir(self._job_dir(job_id))
            except OSError:
                pass
        return released

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
