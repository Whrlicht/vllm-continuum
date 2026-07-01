# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.
"""
import copy
import os
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Generator  # noqa: UP035
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (ensure_kv_transfer_shutdown,
                                          get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput,
                             ModelRunnerOutput)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# ★ LICHT_PROBE=1: master switch for stall-investigation probes (EXEC-SPLIT
# here). Default off → zero overhead in production. See vllm/v1/engine/core.py.
_LICHT_PROBE = os.environ.get("LICHT_PROBE") == "1"


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
class KVConnectorModelRunnerMixin:

    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: "SchedulerOutput"):
        # Update KVConnector with the KVConnector metadata forward().
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(
                scheduler_output.kv_connector_metadata)

            # Background KV cache transfers happen here.
            # These transfers are designed to be async and the requests
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            kv_connector.start_load_kv(get_forward_context())

    @staticmethod
    def ensure_kv_transfer_shutdown() -> None:
        # has_kv_transfer_group can be None during interpreter shutdown.
        if has_kv_transfer_group and has_kv_transfer_group():
            ensure_kv_transfer_shutdown()

    @staticmethod
    def maybe_wait_for_kv_save() -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

    @staticmethod
    def get_finished_kv_transfers(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids)
        return None, None

    @staticmethod
    def kv_connector_no_forward(scheduler_output: "SchedulerOutput",
                                vllm_config: VllmConfig) -> ModelRunnerOutput:
        wait_for_save = False
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            # In direct block mode, producer start_load_kv enqueues bridge
            # metadata that must be published via wait_for_save even when
            # no model forward is executed in this step.
            if getattr(kv_connector, "is_producer", False) and getattr(
                    kv_connector, "direct_block_mode", False):
                wait_for_save = True

        # KV send/recv even if no work to do.
        with set_forward_context(
                None, vllm_config
        ), KVConnectorModelRunnerMixin._get_kv_connector_output(
            scheduler_output,
            wait_for_save=wait_for_save) as kv_connector_output:
            pass

        if (not kv_connector_output.finished_sending
                and not kv_connector_output.finished_recving):
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output: "SchedulerOutput"
    ) -> AbstractContextManager[Optional[KVConnectorOutput]]:
        return KVConnectorModelRunnerMixin._get_kv_connector_output(
            scheduler_output) if has_kv_transfer_group() else nullcontext()

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True
    ) -> Generator[KVConnectorOutput, None, None]:
        output = KVConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata)

        # Background KV cache transfers happen here.
        # These transfers are designed to be async and the requests
        # involved may be disjoint from the running requests.
        # Do this here to save a collective_rpc.
        # ★ EXEC-SPLIT 探针: 把 exec(execute_model)拆成 start_load_kv(connector
        # 加载/收 KV)/ forward(yield=model() dispatch)/ wait_for_save 三段。任一
        # >1s 告警。若 forward 段大且 GPU idle → 引擎线程连 model() dispatch 都跑
        # 不动 = GIL 被后台 arena 线程饿住(坐实 GIL 争用)。
        import time as _esp_t
        _esp0 = _esp_t.perf_counter()
        kv_connector.start_load_kv(get_forward_context())
        _esp1 = _esp_t.perf_counter()
        try:
            yield output
        finally:
            _esp2 = _esp_t.perf_counter()
            if wait_for_save:
                kv_connector.wait_for_save()
            try:
                _esp3 = _esp_t.perf_counter()
                _esp_load = (_esp1 - _esp0) * 1e3
                _esp_fwd = (_esp2 - _esp1) * 1e3
                _esp_save = (_esp3 - _esp2) * 1e3
                if _LICHT_PROBE and (_esp_load > 1000.0 or _esp_fwd > 1000.0
                                     or _esp_save > 1000.0):
                    logger.warning(
                        "EXEC-SPLIT start_load_kv=%.0f forward=%.0f "
                        "wait_save=%.0f ms — exec stall 在这段", _esp_load,
                        _esp_fwd, _esp_save)
            except Exception:  # pragma: no cover - probe must never break
                pass

            output.finished_sending, output.finished_recving = (
                kv_connector.get_finished(scheduler_output.finished_req_ids))

            pop_ts = getattr(kv_connector, "pop_delay_free_timestamps", None)
            if pop_ts is not None:
                req_ids = output.finished_sending or set()
                ts = pop_ts(req_ids)
                if ts:
                    output.delay_free_timestamps = ts

            kv_connector.clear_connector_metadata()
