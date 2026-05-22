# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill-side snapshot writer + decode-side reader.

Uses an atomic-rename JSON file per prefill instance:
    {SNAPSHOT_DIR}/v3_snapshot_{instance_tag}.json

instance_tag comes from env var CONTINUUM_INSTANCE_TAG (set by the run
script for both prefill_* and decode_*).  Decode side scans the
directory for any v3_snapshot_*.json that does NOT match its own tag,
and picks one based on `select_snapshot_path()`.

JSON layout mirrors LichtV3PrefillSnapshot field-for-field.  We pay the
JSON tax intentionally: it is human-readable for debugging, parsing
cost (<1ms for ~hundreds of requests) is negligible against the v3
prediction cost, and it avoids any new IPC infrastructure.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.sched.licht_v3_snapshot import LichtV3PrefillSnapshot

logger = init_logger(__name__)


DEFAULT_SNAPSHOT_DIR = "/tmp/vllm_licht_v3"

# Module-level write-rate limiter: keyed by tag → last write monotonic ts.
# Wall-time cadence is fine here because the decode-side consumer also
# polls at a similar rate (~100ms).  Adjust via env LICHT_V3_SNAPSHOT_MIN_S.
_last_write_ts: dict[str, float] = {}


def _min_write_interval_s() -> float:
    try:
        return float(os.environ.get("LICHT_V3_SNAPSHOT_MIN_S", "0.1"))
    except ValueError:
        return 0.1


def get_snapshot_dir() -> str:
    return os.environ.get("LICHT_V3_SNAPSHOT_DIR", DEFAULT_SNAPSHOT_DIR)


def _ensure_dir() -> str:
    d = get_snapshot_dir()
    Path(d).mkdir(parents=True, exist_ok=True)
    return d


def get_instance_tag() -> str:
    """Tag used in snapshot file names.  Falls back to PID if unset."""
    tag = os.environ.get("CONTINUUM_INSTANCE_TAG", "").strip()
    if not tag:
        tag = f"pid{os.getpid()}"
    return tag


def snapshot_path_for_tag(tag: str) -> str:
    return os.path.join(get_snapshot_dir(), f"v3_snapshot_{tag}.json")


def write_snapshot(snap: LichtV3PrefillSnapshot,
                   tag: Optional[str] = None) -> Optional[str]:
    """Atomically write `snap` to /{SNAPSHOT_DIR}/v3_snapshot_{tag}.json.

    Returns the file path on success, None on failure (errors are
    swallowed and logged at warning).  The function never raises so
    that a stale-disk situation cannot wedge the prefill scheduler.
    """
    if tag is None:
        tag = get_instance_tag()
    now_mono = time.monotonic()
    last = _last_write_ts.get(tag, 0.0)
    if now_mono - last < _min_write_interval_s():
        return None
    _last_write_ts[tag] = now_mono
    try:
        d = _ensure_dir()
        final_path = snapshot_path_for_tag(tag)
        payload = asdict(snap)
        with tempfile.NamedTemporaryFile(
                "w", dir=d, prefix=f".v3_snap_{tag}_", suffix=".json",
                delete=False, encoding="utf-8") as tmp:
            json.dump(payload, tmp, separators=(",", ":"))
            tmp_path = tmp.name
        os.replace(tmp_path, final_path)
        return final_path
    except Exception as e:  # pragma: no cover - non-fatal
        logger.warning("LICHTV3 snapshot write failed: %s", e)
        return None


def select_snapshot_path(exclude_tag: Optional[str] = None,
                         max_age_s: float = 5.0) -> Optional[str]:
    """Pick a fresh-enough prefill snapshot file to read.

    Strategy: among files matching v3_snapshot_*.json in the snapshot
    dir, exclude `exclude_tag` (decode's own tag) and any file whose
    mtime is older than `max_age_s`; return the most recently modified
    survivor.  Returns None when nothing usable is found.
    """
    d = get_snapshot_dir()
    if not os.path.isdir(d):
        return None
    candidates: list[tuple[float, str]] = []
    now = time.time()
    for name in os.listdir(d):
        if not name.startswith("v3_snapshot_") or not name.endswith(".json"):
            continue
        tag = name[len("v3_snapshot_"):-len(".json")]
        if exclude_tag is not None and tag == exclude_tag:
            continue
        path = os.path.join(d, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if now - mtime > max_age_s:
            continue
        candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def read_snapshot(path: str) -> Optional[dict]:
    """Read a snapshot JSON and return it as a plain dict.

    Returns None on parse error (the file may be mid-write on a buggy
    writer; with atomic-rename this should not happen).  We return a
    dict, not the dataclass, to avoid coupling consumers to the
    dataclass module - tests and predictor code can subset what they
    need.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        logger.warning("LICHTV3 snapshot read failed for %s: %s", path, e)
        return None
