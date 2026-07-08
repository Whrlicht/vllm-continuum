# SPDX-License-Identifier: Apache-2.0
"""SPSC 暂存环 (ssd_stage_ring) 单测: 数据/元数据 roundtrip, 满→丢, 空, 绕环."""
import ctypes

import pytest

try:
    import licht_arena_atomic  # noqa: F401
except ImportError:
    pytest.skip("licht_arena_atomic not built", allow_module_level=True)

from vllm.v1.core.sched.licht_v3.ssd_stage_ring import StageRing

SB = 4096          # 测试用小 slot
CHUNK = 4
_libc = ctypes.CDLL(None)
_libc.memmove.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_libc.memmove.restype = ctypes.c_void_p


def _pattern(tag: int) -> bytes:
    return bytes([(tag * 7 + i) % 256 for i in range(SB)])


def _push(ring, src_buf, job, start, count, hashes):
    """模拟 decode capture: 借槽 -> memmove 数据 -> 发布. 返回是否成功."""
    r = ring.reserve()
    if r is None:
        return False
    h, idx = r
    dst = ring.data_addr(idx)
    saddr = ctypes.addressof((ctypes.c_char * len(src_buf)).from_buffer(src_buf))
    for i in range(count):
        _libc.memmove(dst + i * SB, saddr + i * SB, SB)
    ring.publish(h, idx, job, start, count, hashes)
    return True


def test_roundtrip(tmp_path):
    p = str(tmp_path / "r.ring")
    prod = StageRing.create(p, n_slots=4, chunk_blocks=CHUNK, slot_bytes=SB)
    cons = StageRing.open(p)               # 写进程侧打开同一环
    assert cons.chunk_blocks == CHUNK and cons.slot_bytes == SB
    # 准备 3 块数据
    src = bytearray(_pattern(1) + _pattern(2) + _pattern(3))
    hashes = [111, 222, 333]
    assert _push(prod, src, "job::abc", 5, 3, hashes)
    # 消费者取出
    got = cons.pop()
    assert got is not None
    job, start, count, hs, idx = got
    assert job == "job::abc" and start == 5 and count == 3 and hs == hashes
    # 逐块字节比对
    for b in range(3):
        assert bytes(cons.block_mv(idx, b)) == _pattern(b + 1)
    cons.release()
    assert cons.pop() is None               # 空
    prod.close(); cons.close()


def test_full_drops(tmp_path):
    p = str(tmp_path / "r2.ring")
    prod = StageRing.create(p, n_slots=2, chunk_blocks=1, slot_bytes=SB)
    src = bytearray(_pattern(9))
    assert _push(prod, src, "j", 0, 1, [1])
    assert _push(prod, src, "j", 1, 1, [2])
    # 满 (2 槽都占) -> 第三个丢
    assert prod.reserve() is None
    assert not _push(prod, src, "j", 2, 1, [3])
    prod.close()


def test_wraparound(tmp_path):
    p = str(tmp_path / "r3.ring")
    prod = StageRing.create(p, n_slots=2, chunk_blocks=1, slot_bytes=SB)
    cons = StageRing.open(p)
    # 推-取-推-取 循环远超 n_slots, 验证绕环 + 数据始终正确
    for k in range(10):
        src = bytearray(_pattern(k))
        assert _push(prod, src, f"j{k}", k, 1, [k * 10])
        got = cons.pop()
        assert got is not None
        job, start, count, hs, idx = got
        assert job == f"j{k}" and start == k and hs == [k * 10]
        assert bytes(cons.block_mv(idx, 0)) == _pattern(k)
        cons.release()
    assert cons.pop() is None
    prod.close(); cons.close()


def test_spsc_interleaved(tmp_path):
    """生产者连推 2 个, 消费者再逐个取 (SPSC 顺序 FIFO)."""
    p = str(tmp_path / "r4.ring")
    prod = StageRing.create(p, n_slots=4, chunk_blocks=1, slot_bytes=SB)
    cons = StageRing.open(p)
    for k in range(3):
        src = bytearray(_pattern(k + 20))
        assert _push(prod, src, f"x{k}", k, 1, [k])
    for k in range(3):                       # FIFO 顺序取出
        job, start, count, hs, idx = cons.pop()
        assert job == f"x{k}" and start == k
        assert bytes(cons.block_mv(idx, 0)) == _pattern(k + 20)
        cons.release()
    assert cons.pop() is None
    prod.close(); cons.close()
