# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from pathlib import Path
from typing import Any, Optional

from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


class TraceStore:

    def __init__(self, trace_path: Optional[str] = None):
        self.trace_path = Path(trace_path) if trace_path else self._resolve_path()
        self.by_traj_id: dict[str, list[int]] = {}
        self.entry_by_data_id: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._load_if_needed()

    def _force_reload(self) -> None:
        self.by_traj_id.clear()
        self.entry_by_data_id.clear()
        self._loaded = False
        self._load_if_needed()

    def _resolve_path(self) -> Optional[Path]:
        configured = os.getenv("VLLM_TRACE_REPLAY_PATH")
        if configured:
            return Path(configured)

        default_path = Path.cwd() / "trace_data" / \
            "swe_bench_sample_100_with_timings.json"
        if default_path.exists():
            return default_path
        return None

    def _load_if_needed(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if self.trace_path is None:
            logger.warning(
                "Trace replay enabled but no trace file configured. Set "
                "VLLM_TRACE_REPLAY_PATH to a JSON file containing traj_id and "
                "token id sequences.")
            return

        if not self.trace_path.exists():
            raise FileNotFoundError(
                f"Trace replay file not found: {self.trace_path}")

        with self.trace_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        entries = raw if isinstance(raw, list) else [raw]
        loaded = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            data_ids = [
                data_id for data_id in
                (entry.get("traj_id"), entry.get("instance_id")) if data_id
            ]
            if not data_ids:
                continue
            for data_id in data_ids:
                self.entry_by_data_id[str(data_id)] = entry

            token_ids = self._extract_token_ids(entry)
            if token_ids is None:
                continue
            for data_id in data_ids:
                self.by_traj_id[str(data_id)] = token_ids
            loaded += 1

        logger.info("Trace replay loaded %d trajectories from %s", loaded,
                    self.trace_path)

    @staticmethod
    def _is_int_list(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(x, int) for x in value)

    def _extract_token_ids(self, entry: dict[str, Any]) -> Optional[list[int]]:
        direct_keys = (
            "trace_token_ids",
            "forced_token_ids",
            "output_token_ids",
            "token_ids",
            "tokens",
        )
        for key in direct_keys:
            value = entry.get(key)
            if self._is_int_list(value):
                return list(value)

        # Fallback: search nested objects for token-id lists.
        def dfs(obj: Any) -> Optional[list[int]]:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_l = key.lower()
                    if ("token" in key_l and "id" in key_l
                            and self._is_int_list(value)):
                        return list(value)
                    found = dfs(value)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = dfs(item)
                    if found is not None:
                        return found
            elif isinstance(obj, str) and obj and obj[0] in "[{":
                try:
                    parsed = json.loads(obj)
                except json.JSONDecodeError:
                    return None
                return dfs(parsed)
            return None

        return dfs(entry)

    def has_trace(self, traj_id: str) -> bool:
        self._load_if_needed()
        return traj_id in self.by_traj_id or traj_id in self.entry_by_data_id

    def get_entry(self, traj_id: str) -> dict[str, Any]:
        self._load_if_needed()
        entry = self.entry_by_data_id.get(traj_id)
        if entry is None and self.trace_path is not None and self.trace_path.exists():
            # Handle in-place dataset updates without requiring server restart.
            self._force_reload()
            entry = self.entry_by_data_id.get(traj_id)
        if entry is None:
            raise KeyError(f"traj_id={traj_id!r} not found in trace replay store")
        return entry

    def get_messages(self, traj_id: str) -> list[dict[str, Any]]:
        entry = self.get_entry(traj_id)
        return self._extract_messages(entry)

    def get_trace(self, traj_id: str) -> list[int]:
        self._load_if_needed()
        if traj_id not in self.by_traj_id and self.trace_path is not None and self.trace_path.exists():
            # Handle in-place dataset updates without requiring server restart.
            self._force_reload()
        if traj_id not in self.by_traj_id:
            raise KeyError(
                f"traj_id={traj_id!r} not found in trace replay store. "
                "Ensure the dataset contains both traj_id and token ids.")
        return self.by_traj_id[traj_id]

    def materialize_trace_token_ids(
        self,
        traj_id: str,
        tokenizer: Optional[AnyTokenizer],
    ) -> list[int]:
        self._load_if_needed()
        if traj_id in self.by_traj_id:
            return self.by_traj_id[traj_id]

        entry = self.entry_by_data_id.get(traj_id)
        if entry is None:
            raise KeyError(
                f"traj_id={traj_id!r} not found in trace replay store")
        if tokenizer is None:
            raise ValueError(
                "Trace entry does not contain token ids and tokenizer is "
                "unavailable. Disable skip_tokenizer_init or provide token ids "
                "in trace data.")

        trace_text = self._extract_trace_text(entry)
        if not trace_text:
            raise ValueError(
                f"traj_id={traj_id!r} has no parseable text for tokenization")

        token_ids = self._encode_text(tokenizer, trace_text)

        if not token_ids:
            raise ValueError(
                f"traj_id={traj_id!r} produced empty token ids after tokenization")

        self.by_traj_id[traj_id] = token_ids
        return token_ids

    @staticmethod
    def _extract_messages(entry: dict[str, Any]) -> list[dict[str, Any]]:
        messages_obj = entry.get("messages")
        if isinstance(messages_obj, list):
            return [m for m in messages_obj if isinstance(m, dict)]
        if isinstance(messages_obj, str):
            try:
                parsed = json.loads(messages_obj)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return [m for m in parsed if isinstance(m, dict)]
        return []

    @staticmethod
    def _extract_trace_text(entry: dict[str, Any]) -> str:
        messages = TraceStore._extract_messages(entry)
        if messages:
            message_chunks: list[str] = []
            for idx, msg in enumerate(messages):
                role = str(msg.get("role", "unknown"))
                msg_type = msg.get("message_type")
                header = f"[{idx}]<{role}>"
                if msg_type:
                    header = f"[{idx}]<{role}:{msg_type}>"

                body_parts: list[str] = []
                for key in ("content", "thought", "action", "tool_calls",
                            "tool_call_ids", "cache_control"):
                    if key not in msg:
                        continue
                    body_parts.extend(
                        TraceStore._flatten_value_to_text(msg.get(key)))

                if body_parts:
                    body = "\n".join(
                        part for part in body_parts if part is not None and part != "")
                    if body:
                        message_chunks.append(
                            f"{header}\n{body}\n</{role}>")

            return "\n".join(message_chunks)

        # Fallback: use raw messages field if available.
        raw_messages = entry.get("messages")
        if isinstance(raw_messages, str):
            return raw_messages
        if raw_messages is not None:
            return TraceStore._stringify(raw_messages)
        return ""

    @staticmethod
    def _flatten_value_to_text(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (int, float, bool)):
            return [str(value)]
        if isinstance(value, list):
            chunks: list[str] = []
            for item in value:
                chunks.extend(TraceStore._flatten_value_to_text(item))
            return chunks
        if isinstance(value, dict):
            # Preserve all fields for multi-turn/tool traces by canonical JSON.
            return [TraceStore._stringify(value)]
        return [str(value)]

    @staticmethod
    def _stringify(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _encode_text(tokenizer: AnyTokenizer, text: str) -> list[int]:
        # Most tokenizers in vLLM support encode(..., add_special_tokens=False).
        if hasattr(tokenizer, "encode"):
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(text)
            if isinstance(encoded, list):
                return [int(x) for x in encoded]

        # Fallback for tokenizer call interface.
        encoded_obj = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded_obj, dict) and "input_ids" in encoded_obj:
            ids = encoded_obj["input_ids"]
            if isinstance(ids, list):
                return [int(x) for x in ids]

        raise TypeError("Unsupported tokenizer interface for trace replay")

    def get_token(self, traj_id: str, pos: int) -> int:
        return self.get_trace(traj_id)[pos]

    def get_length(self, traj_id: str) -> int:
        return len(self.get_trace(traj_id))


_TRACE_STORE: Optional[TraceStore] = None


def get_trace_store() -> TraceStore:
    global _TRACE_STORE
    if _TRACE_STORE is None:
        _TRACE_STORE = TraceStore()
    return _TRACE_STORE