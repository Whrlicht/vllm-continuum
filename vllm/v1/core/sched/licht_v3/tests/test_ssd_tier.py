# SPDX-License-Identifier: Apache-2.0
"""SsdTier P0 骨架测试 (CPU only, 不依赖 GPU / 真 SSD).

测试矩阵:

Group A - 生命周期: open_or_create / 数据文件尺寸 / meta json / close 幂等
Group B - 复用账本可用性: 第二实例的 LruArenaStore 能 write_inc/lookup
Group C - 跨进程: 两个进程先后 open_or_create 同一 tier 不打架
Group D - 失败降级: 不可写路径 -> 抛 OSError (由调用方捕获禁用)
"""
import json
import os

import pytest

try:
    import licht_arena_atomic  # noqa: F401
except ImportError:
    pytest.skip("licht_arena_atomic not built", allow_module_level=True)

from vllm.v1.core.sched.licht_v3.ssd_tier import SsdTier

# 测试尺寸: slot 4KB, 容量 0.001GB -> 256 slots -> 数据文件 1MB (tmp 友好)
SLOT_BYTES = 4096
SSD_GB = 0.001
BLOCK_SIZE = 16
EXPECT_SLOTS = int(SSD_GB * (1024 ** 3)) // SLOT_BYTES  # 256


@pytest.fixture
def tier_paths(tmp_path):
    meta = str(tmp_path / "ssd_meta")
    data = str(tmp_path / "ssd_data")
    return meta, data


def _open(meta, data) -> SsdTier:
    return SsdTier.open_or_create(
        meta_path=meta, data_path=data, ssd_gb=SSD_GB,
        slot_bytes=SLOT_BYTES, block_size=BLOCK_SIZE)


# ============================================================
# Group A - 生命周期
# ============================================================
class TestLifecycle:

    def test_create_data_file_full_size(self, tier_paths):
        meta, data = tier_paths
        with _open(meta, data) as tier:
            assert tier.num_slots == EXPECT_SLOTS
            st = os.stat(tier.data_file)
            assert st.st_size == EXPECT_SLOTS * SLOT_BYTES
            # fallocate 真实占块 (非稀疏): st_blocks*512 >= 文件尺寸
            assert st.st_blocks * 512 >= EXPECT_SLOTS * SLOT_BYTES

    def test_meta_json_written(self, tier_paths):
        meta, data = tier_paths
        with _open(meta, data):
            m = SsdTier.read_meta(meta)
            assert m == {"num_slots": EXPECT_SLOTS,
                         "slot_bytes": SLOT_BYTES,
                         "block_size": BLOCK_SIZE}

    def test_read_meta_missing(self, tmp_path):
        assert SsdTier.read_meta(str(tmp_path)) is None

    def test_slot_offset_linear(self, tier_paths):
        meta, data = tier_paths
        with _open(meta, data) as tier:
            assert tier.slot_offset(0) == 0
            assert tier.slot_offset(7) == 7 * SLOT_BYTES

    def test_close_idempotent(self, tier_paths):
        meta, data = tier_paths
        tier = _open(meta, data)
        tier.close()
        tier.close()  # 二次 close 不炸

    def test_reopen_same_process(self, tier_paths):
        """重开 (模拟晚到的进程): 账本/尺寸一致, fallocate 幂等."""
        meta, data = tier_paths
        t1 = _open(meta, data)
        size1 = os.stat(t1.data_file).st_size
        t2 = _open(meta, data)
        assert t2.num_slots == t1.num_slots
        assert os.stat(t2.data_file).st_size == size1
        t2.close()
        t1.close()

    def test_stats_shape(self, tier_paths):
        meta, data = tier_paths
        with _open(meta, data) as tier:
            s = tier.stats()
            assert s["num_slots"] == EXPECT_SLOTS
            assert s["free"] == EXPECT_SLOTS       # P0 无写入, 全空闲
            assert s["demote_blocks"] == 0
            assert s["promote_blocks"] == 0


# ============================================================
# Group B - 账本可用性 (第二个 LruArenaStore 实例真能干活)
# ============================================================
class TestLedger:

    def test_write_inc_and_lookup(self, tier_paths):
        """SSD tier 的账本走 LruArenaStore 标准回路 (数据搬运用假 writer)."""
        meta, data = tier_paths
        with _open(meta, data) as tier:
            written = {}

            def fake_writer(slot_id, blk_idx, src):
                written[slot_id] = src[blk_idx]

            tier.store.bind_data_writer(fake_writer)
            n_blocks = 4
            tokens = list(range(n_blocks * BLOCK_SIZE))
            blocks = [f"blk{i}".encode() for i in range(n_blocks)]
            ok = tier.store.write_inc("jobA", 0, n_blocks, tokens, blocks)
            assert ok
            assert len(written) == n_blocks
            matched_tokens, matched_blocks = tier.store.lookup("jobA", tokens)
            assert matched_blocks == n_blocks
            assert tier.stats()["free"] == EXPECT_SLOTS - n_blocks


# ============================================================
# Group C - 跨进程 open_or_create
# ============================================================
def _child_open(meta, data, q):
    try:
        tier = SsdTier.open_or_create(
            meta_path=meta, data_path=data, ssd_gb=SSD_GB,
            slot_bytes=SLOT_BYTES, block_size=BLOCK_SIZE)
        q.put(("ok", tier.num_slots, tier.stats()["free"]))
        tier.close()
    except Exception as e:  # pragma: no cover
        q.put(("err", repr(e), None))


class TestCrossProcess:

    def test_two_process_open(self, tier_paths):
        import multiprocessing as mp
        meta, data = tier_paths
        parent = _open(meta, data)
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        proc = ctx.Process(target=_child_open, args=(meta, data, q))
        proc.start()
        status, slots, free = q.get(timeout=30)
        proc.join(timeout=30)
        assert status == "ok"
        assert slots == parent.num_slots
        assert free == EXPECT_SLOTS
        parent.close()

    def test_race_open(self, tier_paths):
        """两个子进程同时创建 (无先建者): flock 协议保证只 init 一次."""
        import multiprocessing as mp
        meta, data = tier_paths
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        procs = [ctx.Process(target=_child_open, args=(meta, data, q))
                 for _ in range(2)]
        for p in procs:
            p.start()
        results = [q.get(timeout=30) for _ in procs]
        for p in procs:
            p.join(timeout=30)
        assert all(r[0] == "ok" for r in results), results
        assert all(r[1] == EXPECT_SLOTS for r in results)


# ============================================================
# Group E - 账本残留守卫 (布局变更 / 半初始化)
# ============================================================
class TestStaleGuard:

    def test_layout_change_rebuilds_meta(self, tier_paths):
        """slot_bytes 变了 (如换模型) -> 旧账本整体重建, 不带病打开."""
        meta, data = tier_paths
        t1 = _open(meta, data)
        t1.close()
        t2 = SsdTier.open_or_create(
            meta_path=meta, data_path=data, ssd_gb=SSD_GB,
            slot_bytes=SLOT_BYTES * 2, block_size=BLOCK_SIZE)
        assert t2.num_slots == EXPECT_SLOTS // 2
        m = SsdTier.read_meta(meta)
        assert m["slot_bytes"] == SLOT_BYTES * 2
        assert t2.stats()["free"] == EXPECT_SLOTS // 2   # 全新账本
        t2.close()

    def test_halfinit_residue_rebuilds(self, tier_paths):
        """meta 缺失但 hdr 残留 (上次初始化中途崩溃) -> 重建."""
        meta, data = tier_paths
        t1 = _open(meta, data)
        t1.close()
        os.unlink(os.path.join(meta, "_ssd_meta.json"))  # 模拟崩溃残留
        t2 = _open(meta, data)                            # 应重建而非带病开
        assert t2.stats()["free"] == EXPECT_SLOTS
        assert SsdTier.read_meta(meta) is not None
        t2.close()


# ============================================================
# Group D - 失败降级
# ============================================================
class TestFailure:

    def test_unwritable_data_path_raises(self, tmp_path):
        if os.geteuid() == 0:
            pytest.skip("root 无视权限位")
        ro = tmp_path / "ro"
        ro.mkdir()
        os.chmod(ro, 0o500)   # 只读目录
        try:
            with pytest.raises(OSError):
                SsdTier.open_or_create(
                    meta_path=str(tmp_path / "meta"),
                    data_path=str(ro / "sub"),
                    ssd_gb=SSD_GB, slot_bytes=SLOT_BYTES,
                    block_size=BLOCK_SIZE)
        finally:
            os.chmod(ro, 0o700)

    def test_bad_slot_bytes_raises(self, tier_paths):
        meta, data = tier_paths
        with pytest.raises(ValueError):
            SsdTier.open_or_create(
                meta_path=meta, data_path=data, ssd_gb=SSD_GB,
                slot_bytes=0, block_size=BLOCK_SIZE)
