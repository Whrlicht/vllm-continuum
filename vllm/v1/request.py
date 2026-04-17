# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList
from vllm.trace_replay.store import get_trace_store

TRACE_STORE = get_trace_store()

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


class Request:

    @staticmethod
    def _parse_non_negative_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, float):
            if value.is_integer() and value >= 0:
                return int(value)
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
        return None

    @classmethod
    def _extract_agent_round(cls, extra_args: Mapping[str, Any]) -> Optional[int]:
        # Accept multiple keys for compatibility with existing clients.
        for key in ("agent_round", "assistant_round",
                    "trace_replay_assistant_round"):
            parsed = cls._parse_non_negative_int(extra_args.get(key))
            if parsed is not None:
                return parsed
        return None

    def __init__(
        self,
        job_id: Optional[str] = None,
        proxy_request_id: Optional[str] = None,
        request_id: str = "",
        prompt_token_ids: list[int] = [],
        sampling_params: Optional[SamplingParams] = None,
        pooling_params: Optional[PoolingParams] = None,
        eos_token_id: Optional[int] = None,
        client_index: int = 0,
        arrival_time: Optional[float] = None,
        mm_features: Optional[list[MultiModalFeatureSpec]] = None,
        lora_request: Optional["LoRARequest"] = None,
        structured_output_request: Optional["StructuredOutputRequest"] = None,
        cache_salt: Optional[str] = None,
        priority: int = 0,
        trace_headers: Optional[Mapping[str, str]] = None,
        block_hasher: Optional[Callable[["Request"],
                                        list["BlockHash"]]] = None,
        last_func_call: Optional[str] = None,
        is_last_step: Optional[bool] = None,
        this_func_call: Optional[str] = None,
        agent_round: Optional[int] = None,
        trace_replay_token_ids: Optional[list[int]] = None,
    ) -> None:
    # TODO (Hanchen) need to input job_id, last_func_call, is_last_step, this_func_call from the API request
        self.job_id = job_id
        self.proxy_request_id = proxy_request_id
        # NOTE (Hanchen) this i s used for emulation, we have other ways to get this information in real experiment

        self.last_func_call = last_func_call
        self.is_last_step = is_last_step
        self.this_func_call = this_func_call
        self.agent_round = agent_round if agent_round is not None else 0
        
        # trace replay
        self.trace_replay_enabled = bool(
            sampling_params.trace_replay if sampling_params is not None else False
        )
        self.traj_id = sampling_params.traj_id if sampling_params is not None else None
        self.trace_pos = 0
        self.trace_finished = False

        if self.trace_replay_enabled:
            if trace_replay_token_ids is not None:
                self.trace_token_ids = list(trace_replay_token_ids)
            else:
                if not self.traj_id:
                    raise ValueError(
                        "trace_replay=true requires a non-empty traj_id")
                self.trace_token_ids = TRACE_STORE.get_trace(self.traj_id)
        else:
            self.trace_token_ids = None
        
        self.request_id = request_id  
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = structured_output_request
        self.arrival_time = arrival_time if arrival_time is not None else \
            time.time()

        self.status = RequestStatus.WAITING
        self.use_structured_output = False
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: Union[int, str, None] = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: Optional[dict[str, Any]] = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if sampling_params.guided_decoding is not None:
                self.status = RequestStatus.WAITING_FOR_FSM
                self.use_structured_output = True

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = \
                    sampling_params.extra_args.get("kv_transfer_params")
        else:
            raise ValueError(
                "sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()
        self.num_output_placeholders = 0  # Used in async scheduling.
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: Optional[str] = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []
        self.num_encoder_inputs = len(self.mm_features)
        self.has_encoder_inputs = self.num_encoder_inputs > 0
        # TODO(sfeng33): Remove these legacy fields after clearing out all
        # references in scheduler and model runner
        self.mm_positions = [f.mm_position for f in self.mm_features]
        self.mm_kwargs = [f.data for f in self.mm_features]
        self.mm_hashes = [f.identifier for f in self.mm_features]

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        self.block_hashes: list[BlockHash] = []
        self.get_hash_new_full_blocks: Optional[Callable[
            [], list[BlockHash]]] = None
        if block_hasher is not None:
            self.get_hash_new_full_blocks = partial(block_hasher, self)
            self.block_hashes = self.get_hash_new_full_blocks()

    @classmethod
    def from_engine_core_request(
        cls, request: EngineCoreRequest,
        block_hasher: Optional[Callable[["Request"], list["BlockHash"]]]
    ) -> "Request":
        # Extract optional job/function-call metadata from sampling_params.extra_args
        job_id = None
        proxy_request_id = None
        last_func_call = None
        is_last_step = None
        this_func_call = None
        agent_round = None
        if request.sampling_params is not None and \
                request.sampling_params.extra_args is not None:
            extra_args = request.sampling_params.extra_args
            job_id = extra_args.get("job_id")
            proxy_request_id = extra_args.get("proxy_request_id")
            last_func_call = extra_args.get("last_func_call")
            is_last_step = extra_args.get("is_last_step")
            this_func_call = extra_args.get("this_func_call")
            agent_round = cls._extract_agent_round(extra_args)
            trace_replay_token_ids = extra_args.get("trace_replay_token_ids")
        else:
            trace_replay_token_ids = None

        return cls(
            job_id=job_id,
            proxy_request_id=proxy_request_id,
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            structured_output_request=StructuredOutputRequest(
                sampling_params=request.sampling_params) \
                    if request.sampling_params else None,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            last_func_call=last_func_call,
            is_last_step=is_last_step,
            this_func_call=this_func_call,
            agent_round=agent_round,
            trace_replay_token_ids=trace_replay_token_ids,
        )

    def append_output_token_ids(
        self,
        token_ids: Union[int, list[int]],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        if self.get_hash_new_full_blocks is not None:
            self.block_hashes.extend(self.get_hash_new_full_blocks())

    def pop_next_trace_token(self) -> int:
        if not self.trace_replay_enabled or self.trace_token_ids is None:
            raise RuntimeError("Trace replay is not enabled for this request")
        if self.trace_pos >= len(self.trace_token_ids):
            self.trace_finished = True
            raise IndexError("No more trace tokens available")

        token_id = self.trace_token_ids[self.trace_pos]
        self.trace_pos += 1
        if self.trace_pos >= len(self.trace_token_ids):
            self.trace_finished = True
        return token_id

    @property
    def is_output_corrupted(self) -> bool:
        return self.num_nans_in_logits > 0

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[FinishReason, None]:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id].length
        return num_tokens

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: Optional[float] = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> Optional[list[EngineCoreEvent]]:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(
            status: "RequestStatus") -> Union[FinishReason, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
