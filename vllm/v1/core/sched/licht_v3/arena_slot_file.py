# SPDX-License-Identifier: Apache-2.0
"""LICHT Round-KV Arena `.slot` 文件格式 reader/writer.

新格式 (Stage 2):
    header (8 字节):    n = num_blocks_in_inc (uint64 LE)
    records (16n 字节): 对每 block i ∈ [0, n):
                          slot_id      : int64 LE
                          gen          : int64 LE

Stage 6 扩展格式 (向前兼容):
    records 改 24 字节: 加 content_hash : int64 LE
    通过 header 里多存一个版本号或 record size 字段表达

为简洁, Stage 2 用一个固定 magic + version 头:
    bytes  0..4  : magic = b"LSLT"
    bytes  4..6  : version (uint16): 1 = Stage 2, 2 = Stage 6+
    bytes  6..8  : reserved
    bytes  8..16 : n (uint64)
    bytes 16..   : records (per-block records)

总长 16 + n * record_size 字节. record_size = 16 (v1) 或 24 (v2).
"""
from __future__ import annotations

import os
import struct
import tempfile
from dataclasses import dataclass
from typing import List, Tuple


_MAGIC = b"LSLT"
VERSION_V1 = 1  # Stage 2: (slot_id, gen) per block
VERSION_V2 = 2  # Stage 6: (slot_id, gen, content_hash) per block
_HEADER_SIZE = 16


@dataclass(frozen=True)
class SlotFileV1:
    """Stage 2 .slot 文件内容: 每 block 一条 (slot_id, gen)."""
    records: List[Tuple[int, int]]   # [(slot_id, gen), ...]

    @property
    def n(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class SlotFileV2:
    """Stage 6 .slot 文件内容: 每 block 一条 (slot_id, gen, content_hash)."""
    records: List[Tuple[int, int, int]]   # [(slot_id, gen, hash), ...]

    @property
    def n(self) -> int:
        return len(self.records)


def write_slot_file_v1(path: str,
                       records: List[Tuple[int, int]]) -> None:
    """原子写: mkstemp + write + atomic rename.

    records: [(slot_id, gen), ...] 长度 = inc 的 block 数
    """
    n = len(records)
    body = struct.pack("<4sHH", _MAGIC, VERSION_V1, 0)
    body += struct.pack("<Q", n)
    for (slot_id, gen) in records:
        body += struct.pack("<qq", slot_id, gen)

    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".slot_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_slot_file_v1(path: str) -> SlotFileV1 | None:
    """读 v1 格式. 不存在或损坏返回 None."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if len(raw) < _HEADER_SIZE:
        return None
    magic, version, _reserved = struct.unpack("<4sHH", raw[:8])
    if magic != _MAGIC:
        return None
    if version != VERSION_V1:
        # Stage 6 文件用 v2; 本函数不读 v2
        return None
    (n,) = struct.unpack("<Q", raw[8:16])
    expected_size = _HEADER_SIZE + n * 16
    if len(raw) < expected_size:
        return None
    records: List[Tuple[int, int]] = []
    off = _HEADER_SIZE
    for _ in range(n):
        slot_id, gen = struct.unpack("<qq", raw[off:off + 16])
        records.append((slot_id, gen))
        off += 16
    return SlotFileV1(records=records)


def write_slot_file_v2(path: str,
                       records: List[Tuple[int, int, int]]) -> None:
    """★ Stage 6 原子写: 每 block (slot_id, gen, content_hash).

    records: [(slot_id, gen, hash), ...] 长度 = inc 的 block 数.
    hash 是 uint64 (无符号打包); slot_id/gen 为 int64.
    """
    n = len(records)
    body = struct.pack("<4sHH", _MAGIC, VERSION_V2, 0)
    body += struct.pack("<Q", n)
    for (slot_id, gen, h) in records:
        body += struct.pack("<qqQ", slot_id, gen, h & 0xFFFFFFFFFFFFFFFF)

    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".slot_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_slot_file_v2(path: str) -> SlotFileV2 | None:
    """读 v2 格式. 不存在/损坏/非 v2 返回 None."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except (FileNotFoundError, OSError):
        return None
    if len(raw) < _HEADER_SIZE:
        return None
    magic, version, _reserved = struct.unpack("<4sHH", raw[:8])
    if magic != _MAGIC or version != VERSION_V2:
        return None
    (n,) = struct.unpack("<Q", raw[8:16])
    expected_size = _HEADER_SIZE + n * 24
    if len(raw) < expected_size:
        return None
    records: List[Tuple[int, int, int]] = []
    off = _HEADER_SIZE
    for _ in range(n):
        slot_id, gen, h = struct.unpack("<qqQ", raw[off:off + 24])
        records.append((slot_id, gen, h))
        off += 24
    return SlotFileV2(records=records)


def read_slot_file_version(path: str) -> int | None:
    """只读 header 里的 version (1/2). 不存在/损坏返回 None.

    用于 evict/load 在两种格式间分派 (content-addr 模式下是 v2).
    """
    try:
        with open(path, "rb") as f:
            head = f.read(_HEADER_SIZE)
    except (FileNotFoundError, OSError):
        return None
    if len(head) < 8:
        return None
    magic, version, _reserved = struct.unpack("<4sHH", head[:8])
    if magic != _MAGIC:
        return None
    return int(version)


def slot_filename(start_block: int, end_block: int) -> str:
    """统一文件名格式: inc_{start:09d}_{end:09d}.slot

    保证按文件名字典序 = inc 顺序排列 (前补 0).
    """
    return f"inc_{start_block:09d}_{end_block:09d}.slot"


def parse_slot_filename(filename: str) -> Tuple[int, int] | None:
    """从 inc_XXX_YYY.slot 解析 (start, end). 失败返回 None."""
    if not filename.startswith("inc_") or not filename.endswith(".slot"):
        return None
    core = filename[len("inc_"):-len(".slot")]
    try:
        s_str, e_str = core.split("_")
        return int(s_str), int(e_str)
    except (ValueError, AttributeError):
        return None
