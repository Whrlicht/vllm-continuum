# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class P2pConnector:
    """Dispatch P2P KV transfer to the platform-specific backend."""

    def __new__(cls, vllm_config: "VllmConfig", role: KVConnectorRole):
        device_type = current_platform.device_type

        if device_type == "cuda":
            from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (  # noqa: E501
                P2pNcclConnector)
            return P2pNcclConnector(vllm_config, role)

        if device_type in ("npu", "ascend"):
            from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (  # noqa: E501
                P2pNcclConnector) # 逻辑复用
            return P2pNcclConnector(vllm_config, role)

        raise ValueError(
            f"P2pConnector does not support device_type={device_type!r}")
