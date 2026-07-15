# SPDX-License-Identifier: Apache-2.0
"""跨进程 capture bloom (诊断专用, LICHT_SSD_HOLE_PROBE=1 才用).

目的: resolve 撞洞 (ht_miss) 时, 分清这块是
  ① 被 capture 尝试过、但没写上盘 (环满丢 / 跨段失败) —— 真·洞, 连续性 capture 能救
  ⑥ 从没被 demote 过 (= 这轮新内容, SSD 本就没货) —— 不是洞, 正常

decode 侧 capture_inc 把【所有尝试 capture 的 hash】(含丢弃的) 打进 bloom;
prefill 侧 resolve 撞洞时查 bloom: 命中 = ①, 不命中 = ⑥.

跨进程: shm mmap 位数组, 2-hash bloom. best-effort 绝不抛. 竞态下丢个 set 只是
把 ① 少算一点 (退成 ⑥), 对粗分占比无碍. 每 run 由 launcher rm 重置 (避免陈旧位)."""
from __future__ import annotations

import os
import mmap as _mmap

_DEFAULT_MB = 128                       # 1Gbit, 容 ~数百万 hash FP<0.01%


class CaptureBloom:
    def __init__(self, mm, nbits: int):
        self._mm = mm
        self._nbits = nbits

    @classmethod
    def open_or_create(cls, path: str, mb: int = _DEFAULT_MB):
        """打开 (或首个创建) 共享 bloom. 失败返 None (调用方降级为不分)."""
        try:
            nbytes = int(mb) * 1024 * 1024
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if os.fstat(fd).st_size < nbytes:
                    os.ftruncate(fd, nbytes)
                mm = _mmap.mmap(fd, nbytes, _mmap.MAP_SHARED,
                                _mmap.PROT_READ | _mmap.PROT_WRITE)
            finally:
                os.close(fd)
            return cls(mm, nbytes * 8)
        except Exception:
            return None

    def _bits(self, h: int):
        h &= 0xFFFFFFFFFFFFFFFF
        i1 = h % self._nbits
        i2 = (h ^ (h >> 33)) % self._nbits   # 第二 hash, 降 FP
        return i1, i2

    def add(self, h: int) -> None:
        try:
            for b in self._bits(int(h)):
                byte = b >> 3
                self._mm[byte] = self._mm[byte] | (1 << (b & 7))
        except Exception:
            pass

    def test(self, h: int) -> bool:
        try:
            for b in self._bits(int(h)):
                if not (self._mm[b >> 3] & (1 << (b & 7))):
                    return False
            return True
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass
