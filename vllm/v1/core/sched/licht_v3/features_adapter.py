# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime feature-extraction adapter for the tool_call_time predictor.

Bridges between vLLM's `Request` object (with raw token IDs) and the
offline `tool_call_time.features.TrajectoryState` API (which expects an
OpenAI-format tool_call dict and maintains per-trajectory state).

Pipeline (per decode-round finish):
  1. Detokenize `request.output_token_ids` to text.
  2. Extract the latest tool_call dict from the assistant text using
     a robust multi-format parser.
  3. Use TrajectoryTracker (keyed by `job_id`) to call
     `TrajectoryState.extract(tool_call)` and `observe(...)`.
  4. The caller wraps the resulting dict into a pandas DataFrame and
     feeds it to `Predictor.predict_df`.

Limitations / design notes:
  - We use a self-bootstrapping `observe(predicted_t)` because real
    actual-execution times are not available at decode-finish (the
    tool hasn't run yet).  The history features (E1-E3) therefore
    reflect predicted-not-actual durations.  For SWE-bench style
    workloads this is the best signal available without round-trip
    timing telemetry.
  - The tokenizer is loaded lazily and cached process-globally so
    multiple decode_managers in the same process share it.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from typing import Any, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# tool_call_time package import (resolves regardless of CWD)
# ---------------------------------------------------------------------------

def _ensure_tool_call_time_on_path() -> None:
    candidates: list[os.PathLike | str] = []
    predictor_dir = os.environ.get("LICHT_V3_TOOL_PREDICTOR_DIR")
    if predictor_dir:
        p = os.path.abspath(predictor_dir)
        candidates.append(p)
        parent = p
        while True:
            next_parent = os.path.dirname(parent)
            if next_parent == parent:
                break
            candidates.append(next_parent)
            parent = next_parent

    here = os.path.abspath(__file__)
    parent = here
    while True:
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            break
        candidates.append(next_parent)
        parent = next_parent

    for candidate in candidates:
        root = os.fspath(candidate)
        if os.path.isfile(os.path.join(root, "tool_call_time", "features.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return


# ---------------------------------------------------------------------------
# Lazy tokenizer load (process-singleton keyed by model path)
# ---------------------------------------------------------------------------

_tokenizer_cache: dict[tuple[str, str], Any] = {}
_tokenizer_lock = threading.Lock()


def get_tokenizer(model_name_or_path: str,
                  tokenizer_mode: str = "auto",
                  trust_remote_code: bool = False,
                  tokenizer_revision: Optional[str] = None) -> Any:
    key = (model_name_or_path, tokenizer_revision or "")
    with _tokenizer_lock:
        tok = _tokenizer_cache.get(key)
        if tok is not None:
            return tok
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=tokenizer_revision,
            use_fast=(tokenizer_mode != "slow"),
        )
        _tokenizer_cache[key] = tok
        return tok


# ---------------------------------------------------------------------------
# Tool-call extraction
# ---------------------------------------------------------------------------

# Hermes / qwen style:   <tool_call>{"name": ..., "arguments": {...}}</tool_call>
_HERMES_RE = re.compile(
    r"<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>",
    re.DOTALL)
# OpenAI-style direct JSON: {"name": ..., "arguments": ...}
# Llama 3.1 native tool-call: {"name": ..., "parameters": "..."}
# Same structure, different key name.  Accept either.
_NAMED_JSON_RE = re.compile(
    r"\{\s*\"name\"\s*:\s*\"(?P<name>[a-zA-Z_][\w-]*)\"\s*,\s*"
    r"\"(?:arguments|parameters)\"\s*:\s*"
    r"(?P<args>\"(?:[^\"\\]|\\.)*\"|\{.*?\})\s*\}",
    re.DOTALL)
# Same but key order reversed (some templates emit name LAST).
_NAMED_JSON_REV_RE = re.compile(
    r"\{\s*\"(?:arguments|parameters)\"\s*:\s*"
    r"(?P<args>\"(?:[^\"\\]|\\.)*\"|\{.*?\})\s*,\s*"
    r"\"name\"\s*:\s*\"(?P<name>[a-zA-Z_][\w-]*)\"\s*\}",
    re.DOTALL)
# Function-call XML/tag style:
#   <function=NAME>{...}</function>
_TAG_RE = re.compile(
    r"<function=(?P<name>[a-zA-Z_][\w-]*)>(?P<args>\{.*?\})\s*</function>",
    re.DOTALL)
# Plain "Action: bash\nAction Input: <json>"  (ReAct legacy)
_REACT_RE = re.compile(
    r"Action\s*:\s*(?P<name>[a-zA-Z_][\w-]*)\s*\n+\s*Action\s+Input\s*:\s*"
    r"(?P<args>\{.*?\})",
    re.DOTALL)


# ---------------------------------------------------------------------------
# Trace-aware tool_call extraction (preferred path when trace_replay enabled)
# ---------------------------------------------------------------------------

def extract_tool_call_from_trace(request) -> Optional[dict]:
    """Look up the canonical (training-format) tool_call dict for this
    request directly from the TraceStore.  Works when:
      - request.trace_replay_enabled is True
      - request.traj_id is set
      - request.agent_round identifies the N-th assistant message

    Returns the same dict shape `tool_call_time.bucket.classify` expects:
      {"function": {"name": "...", "arguments": "..."}}
    (Plus other fields like "id", "type", "execution_time_seconds" left
    as-is so callers can also use the true actual_t for `observe`.)
    """
    if not getattr(request, "trace_replay_enabled", False):
        return None
    traj_id = getattr(request, "traj_id", None)
    if not traj_id:
        return None
    try:
        from vllm.trace_replay.store import get_trace_store
        store = get_trace_store()
        messages = store.get_messages(str(traj_id))
    except Exception:
        return None
    if not messages:
        return None
    target_round = max(int(getattr(request, "agent_round", 0) or 0), 0)
    seen = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if seen != target_round:
            seen += 1
            continue
        tcs = msg.get("tool_calls")
        if not isinstance(tcs, list) or not tcs:
            return None
        tc = tcs[0] if isinstance(tcs[0], dict) else None
        return tc
    return None


# Trace-replay block tags allow optional message_type:
#   <tool:observation> / </tool>
#   <user:obs>         / </user>
_TOOL_BLOCK_RE = re.compile(
    r"<tool(?::\w+)?>\s*\n(?P<body>.*?)\s*\n\s*</tool>", re.DOTALL)
_USER_BLOCK_RE = re.compile(
    r"<user(?::\w+)?>\s*\n(?P<body>.*?)\s*\n\s*</user>", re.DOTALL)


def extract_observation_text(prompt_tail_text: str) -> str:
    """Extract the most recent OBSERVATION text from a decoded
    prompt-tail.

    Trace-replay format (`TraceStore._extract_trace_text`):
        [N]<tool:observation>
        {"text": "OBSERVATION:\\n...", "type": "text"}
        toolu_<call_id>
        </tool>

    The body is multi-line: one or more JSON dicts (each with a
    `text` field) plus tool_call_id string lines.  We concatenate
    all parseable `text` fields; if no JSON dict has a `text` field,
    fall back to the raw body so the FileCache parser can do its own
    line-by-line work on it.
    """
    if not prompt_tail_text:
        return ""
    matches = _TOOL_BLOCK_RE.findall(prompt_tail_text)
    if not matches:
        matches = _USER_BLOCK_RE.findall(prompt_tail_text)
    if not matches:
        return ""
    body = matches[-1].strip()
    text_parts: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    t = obj.get("text")
                    if isinstance(t, str):
                        text_parts.append(t)
                        continue
            except json.JSONDecodeError:
                pass
        # Also handle top-level JSON list form (some traces).
        if line.startswith("[") and line.endswith("]"):
            try:
                arr = json.loads(line)
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, dict):
                            t = item.get("text")
                            if isinstance(t, str):
                                text_parts.append(t)
                continue
            except json.JSONDecodeError:
                pass
    if text_parts:
        return "\n".join(text_parts)
    return body


def _find_function_dict(text: str) -> Optional[dict]:
    """Scan `text` for an embedded tool_call dict and return an
    OpenAI-format dict ({"function": {"name", "arguments"}}).

    Handles three on-the-wire shapes:
      A. OpenAI canonical (SWE-bench trace dump):
           {"function": {"arguments": ..., "name": ...}, ...}
      B. Llama 3.1 native tool_call:
           {"name": ..., "parameters": ...}
      C. OpenAI inline:
           {"name": ..., "arguments": ...}

    Algorithm: walk every `{` in the text, run a balanced-brace scan
    from it, `json.loads` the slice, and keep the FIRST dict that
    matches one of the shapes above.  Returns a normalised
    `{"function": {"name", "arguments"}}` dict regardless of input
    shape so callers can treat it uniformly.
    """
    n = len(text)
    fast_substr_filters = ('"function"', '"name"', "'function'", "'name'")
    for j in range(n):
        if text[j] != "{":
            continue
        end = _balanced_brace_end(text, j)
        if end < 0:
            continue
        slice_ = text[j:end]
        if not any(needle in slice_ for needle in fast_substr_filters):
            continue
        try:
            obj = json.loads(slice_)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        # Shape A: nested "function" with "name".
        fn = obj.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return obj
        # Shape B/C: top-level "name" + "arguments" or "parameters".
        if isinstance(obj.get("name"), str):
            args = obj.get("arguments")
            if args is None:
                args = obj.get("parameters")
            if args is not None:
                return {"function": {"name": obj["name"],
                                       "arguments": args}}
    return None


def _balanced_brace_end(text: str, start: int) -> int:
    """Return index one-past the `}` that matches `text[start] == '{'`.
    Respects JSON string literals (so `"}"` inside a string doesn't
    decrement depth).  Returns -1 if no match before end of string."""
    if start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_str = False
    escape = False
    for k in range(start, len(text)):
        c = text[k]
        if in_str:
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    return -1


def extract_tool_call(text: str) -> Optional[dict]:
    """Return an OpenAI-format tool_call dict ({function: {name, arguments}})
    or None if no recognised tool_call appears in `text`.

    Tried in order:
      1. Balanced JSON scan for `{"function": {...}}` — the format the
         trace_replay text materialiser produces (sort_keys=True).
      2. Hermes / qwen-tool: `<tool_call>{"name", "arguments"}</tool_call>`.
      3. XML tag: `<function=NAME>{...}</function>`.
      4. OpenAI inline JSON: `{"name": ..., "arguments": ...}`.
      5. ReAct legacy: `Action: ... Action Input: ...`.
    """
    if not text:
        return None

    # 1) Trace-replay / OpenAI-canonical form: {"function": {...}}.
    obj = _find_function_dict(text)
    if obj is not None:
        fn = obj["function"]
        name = fn.get("name")
        args = fn.get("arguments")
        if name:
            return _normalise(name, args)

    # 2) Hermes / qwen-tool style.
    m = _HERMES_RE.search(text)
    if m:
        try:
            body = json.loads(m.group("body"))
            name = body.get("name")
            args = body.get("arguments")
            if name:
                return _normalise(name, args)
        except json.JSONDecodeError:
            pass

    # 3) XML-tag style.
    m = _TAG_RE.search(text)
    if m:
        return _normalise(m.group("name"), m.group("args"))

    # 4) Inline named JSON.
    m = _NAMED_JSON_RE.search(text)
    if m:
        args_raw = m.group("args")
        if args_raw.startswith('"'):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = args_raw[1:-1]
        else:
            args = args_raw
        return _normalise(m.group("name"), args)

    # 5) ReAct legacy.
    m = _REACT_RE.search(text)
    if m:
        return _normalise(m.group("name"), m.group("args"))

    return None


def _normalise(name: str, arguments) -> dict:
    """Coerce `arguments` to a string when possible (the bucket
    classifier's `_parse_args` does `json.loads(str)` and falls back
    cleanly when given a non-JSON string)."""
    if isinstance(arguments, dict):
        try:
            arguments = json.dumps(arguments)
        except (TypeError, ValueError):
            arguments = "{}"
    elif arguments is None:
        arguments = "{}"
    elif not isinstance(arguments, str):
        arguments = str(arguments)
    return {"function": {"name": name, "arguments": arguments}}


# ---------------------------------------------------------------------------
# TrajectoryTracker
# ---------------------------------------------------------------------------

class TrajectoryTracker:
    """One TrajectoryState per agent conversation (keyed by `job_id`).

    Self-bootstraps: at predict() time we call extract() then observe()
    using the predicted p50 as the actual time (we never see the real
    tool execution time on decode side).  This keeps the history
    features sane for subsequent rounds without round-trip telemetry.
    """

    def __init__(self,
                 global_bucket_log_means: dict[str, float],
                 timeout_threshold_s: float = 60.0,
                 max_states: int = 4096):
        _ensure_tool_call_time_on_path()
        # Imported inside __init__ so module import does not require
        # tool_call_time on PYTHONPATH (unit tests, etc.).
        from features import TrajectoryState as _TS  # type: ignore
        from bucket import classify as _classify  # type: ignore
        from bucket import family as _family  # type: ignore
        self._TrajectoryState = _TS
        self._classify = _classify
        self._family = _family
        self._mu = global_bucket_log_means
        self._timeout_thresh = timeout_threshold_s
        self._max_states = max_states
        self._lock = threading.Lock()
        # job_id → TrajectoryState
        self._states: dict[str, Any] = {}
        # LRU ordering: most-recently-used at the end.
        self._lru: list[str] = []

    def _get_state(self, job_id: str):
        with self._lock:
            st = self._states.get(job_id)
            if st is None:
                if len(self._states) >= self._max_states:
                    # Evict the oldest job_id.
                    evict = self._lru.pop(0) if self._lru else None
                    if evict is not None:
                        self._states.pop(evict, None)
                st = self._TrajectoryState(
                    self._mu, self._timeout_thresh)
                self._states[job_id] = st
                self._lru.append(job_id)
            else:
                # Move to end (most recently used).
                try:
                    self._lru.remove(job_id)
                except ValueError:
                    pass
                self._lru.append(job_id)
            return st

    def feature_row(self, job_id: str, tool_call: dict) -> dict:
        st = self._get_state(job_id)
        feats = st.extract(tool_call)
        row = dict(feats)
        row["_family"] = self._family(feats["bucket"])
        return row

    def observe(self, job_id: str, tool_call: dict,
                actual_t: float, obs_text: str = "") -> None:
        st = self._states.get(job_id)
        if st is None:
            return
        st.observe(tool_call, float(actual_t), obs_text)

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._states.pop(job_id, None)
            try:
                self._lru.remove(job_id)
            except ValueError:
                pass

    def stats(self) -> dict:
        with self._lock:
            return {"n_states": len(self._states)}
