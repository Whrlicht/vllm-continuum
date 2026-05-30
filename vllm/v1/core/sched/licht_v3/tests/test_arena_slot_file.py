# SPDX-License-Identifier: Apache-2.0
"""`.slot` 文件格式 reader/writer 单元测试."""
import os
import struct

import pytest

from vllm.v1.core.sched.licht_v3.arena_slot_file import (
    SlotFileV1,
    VERSION_V1,
    VERSION_V2,
    parse_slot_filename,
    read_slot_file_v1,
    slot_filename,
    write_slot_file_v1,
)


@pytest.fixture
def tmp_slot_path(tmp_path):
    return str(tmp_path / "inc_000000000_000000010.slot")


# ============================================================
# Group A - 基本 round-trip
# ============================================================

class TestGroupARoundTrip:
    def test_empty_records(self, tmp_slot_path):
        write_slot_file_v1(tmp_slot_path, [])
        result = read_slot_file_v1(tmp_slot_path)
        assert result is not None
        assert result.records == []
        assert result.n == 0

    def test_single_record(self, tmp_slot_path):
        write_slot_file_v1(tmp_slot_path, [(42, 7)])
        result = read_slot_file_v1(tmp_slot_path)
        assert result.records == [(42, 7)]

    def test_many_records(self, tmp_slot_path):
        records = [(i, i * 2 + 1) for i in range(100)]
        write_slot_file_v1(tmp_slot_path, records)
        result = read_slot_file_v1(tmp_slot_path)
        assert result.records == records

    def test_large_values(self, tmp_slot_path):
        # gen 接近 48 位上限, slot_id 接近 32 位
        records = [(2 ** 31 - 1, 2 ** 48 - 1)]
        write_slot_file_v1(tmp_slot_path, records)
        result = read_slot_file_v1(tmp_slot_path)
        assert result.records == records


# ============================================================
# Group B - 错误处理
# ============================================================

class TestGroupBErrors:
    def test_missing_file(self, tmp_path):
        result = read_slot_file_v1(str(tmp_path / "nonexistent.slot"))
        assert result is None

    def test_truncated_header(self, tmp_slot_path):
        with open(tmp_slot_path, "wb") as f:
            f.write(b"LSLT\x01\x00")  # 6 字节, 不够 16
        result = read_slot_file_v1(tmp_slot_path)
        assert result is None

    def test_wrong_magic(self, tmp_slot_path):
        body = struct.pack("<4sHHQ", b"XXXX", VERSION_V1, 0, 0)
        with open(tmp_slot_path, "wb") as f:
            f.write(body)
        result = read_slot_file_v1(tmp_slot_path)
        assert result is None

    def test_wrong_version(self, tmp_slot_path):
        """V1 reader 拒绝 V2 文件."""
        body = struct.pack("<4sHHQ", b"LSLT", VERSION_V2, 0, 0)
        with open(tmp_slot_path, "wb") as f:
            f.write(body)
        result = read_slot_file_v1(tmp_slot_path)
        assert result is None

    def test_truncated_records(self, tmp_slot_path):
        """header 说 n=5 但 body 只有 3 条 record."""
        body = struct.pack("<4sHHQ", b"LSLT", VERSION_V1, 0, 5)
        # 只写 3 条 record
        for i in range(3):
            body += struct.pack("<qq", i, i)
        with open(tmp_slot_path, "wb") as f:
            f.write(body)
        result = read_slot_file_v1(tmp_slot_path)
        assert result is None


# ============================================================
# Group C - 文件名 helper
# ============================================================

class TestGroupCFilename:
    def test_filename_format(self):
        assert slot_filename(0, 50) == "inc_000000000_000000050.slot"
        assert slot_filename(123, 456) == "inc_000000123_000000456.slot"

    def test_filename_sorts_correctly(self):
        """文件名按字典序应等于 (start, end) 数值序."""
        names = [
            slot_filename(80, 120),
            slot_filename(0, 50),
            slot_filename(50, 80),
        ]
        assert sorted(names) == [
            slot_filename(0, 50),
            slot_filename(50, 80),
            slot_filename(80, 120),
        ]

    def test_parse_valid(self):
        assert parse_slot_filename("inc_000000000_000000050.slot") == (0, 50)
        assert parse_slot_filename("inc_000000123_000000456.slot") == (123, 456)

    def test_parse_invalid(self):
        assert parse_slot_filename("manifest.json") is None
        assert parse_slot_filename("inc_abc_def.slot") is None
        assert parse_slot_filename("inc_0_1.slot") == (0, 1)  # 不要求 09d


# ============================================================
# Group D - 原子写
# ============================================================

class TestGroupDAtomicWrite:
    def test_atomic_rename_no_partial(self, tmp_path):
        """写过程中崩溃, 不应留下半完整文件 (因为原子 rename)."""
        path = str(tmp_path / "x.slot")
        write_slot_file_v1(path, [(1, 2), (3, 4)])
        # 文件应该完整可读
        assert os.path.exists(path)
        result = read_slot_file_v1(path)
        assert result.records == [(1, 2), (3, 4)]
        # 没有 .tmp 残留
        assert not any(f.endswith(".tmp") for f in os.listdir(tmp_path))

    def test_overwrite_existing(self, tmp_path):
        path = str(tmp_path / "x.slot")
        write_slot_file_v1(path, [(1, 1)])
        write_slot_file_v1(path, [(2, 2), (3, 3)])
        result = read_slot_file_v1(path)
        assert result.records == [(2, 2), (3, 3)]
