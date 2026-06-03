# SPDX-License-Identifier: Apache-2.0
"""LICHT Round-KV Arena Stage 6 内容寻址基础设施测试.

测试矩阵:

Group A - 链式 block hash:
  - 确定性 / 区分前缀 / 共享前缀逐块相等且首分叉块 diverge / 跨进程一致 (纯函数)
Group B - slot refcnt (atomic uint16, 经真 ArenaHdr):
  - inc/dec/get/set / 跨 slot 不串扰 / 初始为 0
Group C - hash 表 (open-addressing + seqlock, 经真 ArenaHdr):
  - clear / probe miss / insert+probe hit / remove+tombstone 探测链 / 更新 / 表满
Group D - 跨进程 (fork, 两进程共享同一 hdr 文件):
  - 子进程 insert / refcnt++, 父进程 probe / refcnt_get 可见
"""
import os
import struct

import pytest

try:
    import licht_arena_atomic as A
except ImportError:
    pytest.skip("licht_arena_atomic not built", allow_module_level=True)

from vllm.v1.core.sched.licht_v3 import arena_block_hash as BH
from vllm.v1.core.sched.licht_v3.arena_hdr import ArenaHdr


@pytest.fixture
def tmp_hdr_path(tmp_path):
    return str(tmp_path / "test_arena.hdr")


# ============================================================
# Group A - 链式 block hash (纯函数)
# ============================================================

class TestGroupABlockHash:
    def test_deterministic(self):
        toks = list(range(64))
        h1 = BH.block_hashes(toks, block_size=16)
        h2 = BH.block_hashes(toks, block_size=16)
        assert h1 == h2
        assert len(h1) == 4

    def test_only_full_blocks(self):
        # 70 token, bs=16 -> 4 个完整 block, 尾 6 token 忽略
        toks = list(range(70))
        assert len(BH.block_hashes(toks, block_size=16)) == 4

    def test_chain_position_matters(self):
        # 同样的 block 内容, 不同前缀 -> hash 不同 (链式)
        a = BH.block_hashes([0] * 16 + [5] * 16, block_size=16)
        b = BH.block_hashes([9] * 16 + [5] * 16, block_size=16)
        assert a[0] != b[0]          # block0 内容不同
        assert a[1] != b[1]          # block1 内容相同但前缀不同 -> hash 不同

    def test_shared_prefix_diverge_at_first_diff(self):
        # 两请求共享前 3 个 block, 第 4 个分叉
        base = list(range(48))       # 3 blocks
        req1 = base + [100] * 16
        req2 = base + [200] * 16
        h1 = BH.block_hashes(req1, block_size=16)
        h2 = BH.block_hashes(req2, block_size=16)
        assert h1[:3] == h2[:3]      # 共享前缀逐块相等
        assert h1[3] != h2[3]        # 首个分叉块 diverge

    def test_prefix_hash_resume(self):
        toks = list(range(80))       # 5 blocks
        full = BH.block_hashes(toks, block_size=16)
        # prefix_hash(upto=3) 应等于 block 2 的 hash (续链起点)
        assert BH.prefix_hash(toks, 16, 3) == full[2]
        assert BH.prefix_hash(toks, 16, 0) == BH.SEED0

    def test_seed_nonzero(self):
        assert BH.SEED0 != 0


# ============================================================
# Group B - slot refcnt
# ============================================================

class TestGroupBRefcnt:
    def test_inc_dec_get_set(self, tmp_hdr_path):
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=100)
        try:
            addr = hdr.slot_refcnt_addr(7)
            assert A.refcnt_get(addr) == 0
            A.refcnt_set(addr, 1)
            assert A.refcnt_get(addr) == 1
            assert A.refcnt_inc(addr) == 2
            assert A.refcnt_inc(addr) == 3
            assert A.refcnt_dec(addr) == 2
            assert A.refcnt_get(addr) == 2
        finally:
            hdr.close()

    def test_distinct_per_slot(self, tmp_hdr_path):
        N = 100
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=N)
        try:
            addrs = [hdr.slot_refcnt_addr(i) for i in range(N)]
            assert len(set(addrs)) == N
            for i in range(N - 1):
                assert addrs[i + 1] - addrs[i] == 2   # uint16
            # 改一个不串扰邻居
            A.refcnt_set(hdr.slot_refcnt_addr(50), 9)
            assert A.refcnt_get(hdr.slot_refcnt_addr(49)) == 0
            assert A.refcnt_get(hdr.slot_refcnt_addr(51)) == 0
            assert A.refcnt_get(hdr.slot_refcnt_addr(50)) == 9
        finally:
            hdr.close()

    def test_initial_zero(self, tmp_hdr_path):
        N = 64
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=N)
        try:
            for i in range(N):
                assert A.refcnt_get(hdr.slot_refcnt_addr(i)) == 0
        finally:
            hdr.close()


# ============================================================
# Group C - hash 表
# ============================================================

class TestGroupCHashTable:
    def test_clear_then_probe_miss(self, tmp_hdr_path):
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=100)
        try:
            hdr.content_addr_init()
            base, cap = hdr.hash_table_addr, hdr.hash_table_capacity
            assert A.ht_probe(base, cap, 0x1234) == -1
        finally:
            hdr.close()

    def test_clear_required_zero_page_not_empty(self, tmp_hdr_path):
        """未 clear 时零页 slot_id=0 会被误判命中 -> 证明 clear 必要."""
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=100)
        try:
            base, cap = hdr.hash_table_addr, hdr.hash_table_capacity
            # 不调 content_addr_init: 零页. hash=0 落到的 entry slot_id=0,hash=0
            # -> probe(0) 会命中 slot 0 (错误!). clear 后才正确 miss.
            assert A.ht_probe(base, cap, 0) == 0       # 零页伪命中
            hdr.content_addr_init()
            assert A.ht_probe(base, cap, 0) == -1       # clear 后正确 miss
        finally:
            hdr.close()

    def test_insert_probe_remove(self, tmp_hdr_path):
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=100)
        try:
            hdr.content_addr_init()
            base, cap = hdr.hash_table_addr, hdr.hash_table_capacity
            h = 0xDEADBEEFCAFE
            assert A.ht_probe(base, cap, h) == -1
            assert A.ht_insert(base, cap, h, 42) == 1
            assert A.ht_probe(base, cap, h) == 42
            assert A.ht_remove(base, cap, h) == 1
            assert A.ht_probe(base, cap, h) == -1
            assert A.ht_remove(base, cap, h) == 0       # 再删返回 0
        finally:
            hdr.close()

    def test_update_existing_hash(self, tmp_hdr_path):
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=100)
        try:
            hdr.content_addr_init()
            base, cap = hdr.hash_table_addr, hdr.hash_table_capacity
            h = 0xABCD
            A.ht_insert(base, cap, h, 10)
            A.ht_insert(base, cap, h, 20)               # 同 hash 更新
            assert A.ht_probe(base, cap, h) == 20
        finally:
            hdr.close()

    def test_tombstone_probe_chain(self, tmp_hdr_path):
        """删中间元素后, 探测链不能断 (后面元素仍找得到)."""
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=100)
        try:
            hdr.content_addr_init()
            base, cap = hdr.hash_table_addr, hdr.hash_table_capacity
            # 构造同槽冲突链: h, h+cap, h+2cap 落到同一起始 bucket
            h0 = 5
            chain = [h0 + k * cap for k in range(3)]
            for i, hh in enumerate(chain):
                assert A.ht_insert(base, cap, hh, 100 + i) == 1
            # 删中间那个
            assert A.ht_remove(base, cap, chain[1]) == 1
            # 链尾仍可探测到 (tombstone 不能截断)
            assert A.ht_probe(base, cap, chain[2]) == 102
            assert A.ht_probe(base, cap, chain[0]) == 100
            assert A.ht_probe(base, cap, chain[1]) == -1
        finally:
            hdr.close()

    def test_tombstone_reused_on_insert(self, tmp_hdr_path):
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=100)
        try:
            hdr.content_addr_init()
            base, cap = hdr.hash_table_addr, hdr.hash_table_capacity
            h = 7
            A.ht_insert(base, cap, h, 1)
            A.ht_remove(base, cap, h)
            assert A.ht_insert(base, cap, h, 2) == 1     # 复用墓碑
            assert A.ht_probe(base, cap, h) == 2
        finally:
            hdr.close()

    def test_table_full_returns_zero(self, tmp_hdr_path):
        """填满 cap 个不同 hash 后, 再 insert 返回 0 (表满)."""
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=4)
        try:
            hdr.content_addr_init()
            base, cap = hdr.hash_table_addr, hdr.hash_table_capacity
            # 用 cap 个不同 hash 填满
            ok = 0
            for i in range(cap):
                if A.ht_insert(base, cap, 1000 + i, i) == 1:
                    ok += 1
            assert ok == cap
            assert A.ht_insert(base, cap, 999999, 0) == 0   # 满了
        finally:
            hdr.close()

    def test_many_insert_probe(self, tmp_hdr_path):
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=2000)
        try:
            hdr.content_addr_init()
            base, cap = hdr.hash_table_addr, hdr.hash_table_capacity
            hashes = {A.block_hash(0, struct.pack("<I", i)): i
                      for i in range(1500)}
            for h, sid in hashes.items():
                assert A.ht_insert(base, cap, h, sid % 2000) == 1
            for h, sid in hashes.items():
                assert A.ht_probe(base, cap, h) == sid % 2000
        finally:
            hdr.close()


# ============================================================
# Group D - 跨进程 (fork)
# ============================================================

class TestGroupDCrossProcess:
    def test_child_insert_parent_probe(self, tmp_hdr_path):
        """父创建 hdr+clear, 子进程 (独立 open) insert, 父进程 probe 可见."""
        N = 500
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=N)
        hdr.content_addr_init()
        base, cap = hdr.hash_table_addr, hdr.hash_table_capacity
        h = 0x11112222

        pid = os.fork()
        if pid == 0:
            # 子进程: 独立 open 同一文件, 在 mutex 内 insert
            os._exit(_child_insert(tmp_hdr_path, N, h, 77))
        _, status = os.waitpid(pid, 0)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0

        # 父进程在自己的 mmap 上 probe -> 看得到子进程写的
        assert A.ht_probe(base, cap, h) == 77
        hdr.close()

    def test_cross_process_refcnt(self, tmp_hdr_path):
        """子进程 refcnt++ 三次, 父进程读到 3."""
        N = 100
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=N)
        slot = 13
        pid = os.fork()
        if pid == 0:
            os._exit(_child_refcnt(tmp_hdr_path, N, slot, 3))
        _, status = os.waitpid(pid, 0)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        assert A.refcnt_get(hdr.slot_refcnt_addr(slot)) == 3
        hdr.close()


def _child_insert(path: str, n: int, h: int, slot_id: int) -> int:
    try:
        child = ArenaHdr.open(path, num_slots=n)
        rc = A.mutex_lock(child.mutex_addr)
        if rc != 0:
            return 2
        try:
            ok = A.ht_insert(child.hash_table_addr,
                             child.hash_table_capacity, h, slot_id)
        finally:
            A.mutex_unlock(child.mutex_addr)
        child.close()
        return 0 if ok == 1 else 3
    except Exception:
        return 1


def _child_refcnt(path: str, n: int, slot: int, times: int) -> int:
    try:
        child = ArenaHdr.open(path, num_slots=n)
        rc = A.mutex_lock(child.mutex_addr)
        if rc != 0:
            return 2
        try:
            for _ in range(times):
                A.refcnt_inc(child.slot_refcnt_addr(slot))
        finally:
            A.mutex_unlock(child.mutex_addr)
        child.close()
        return 0
    except Exception:
        return 1
