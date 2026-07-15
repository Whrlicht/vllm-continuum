# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

from vllm.config.kv_transfer import KVTransferConfig
from vllm.platforms import current_platform


def create_p2p_engine(local_rank: int,
                      config: KVTransferConfig,
                      hostname: str = "",
                      port_offset: int = 0,
                      library_path: Optional[str] = None):
    device_type = current_platform.device_type

    if device_type == "cuda":
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (  # noqa: E501
            P2pNcclEngine)
        return P2pNcclEngine(local_rank=local_rank,
                             config=config,
                             hostname=hostname,
                             port_offset=port_offset,
                             library_path=library_path)

    if device_type in ("npu", "ascend"):
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_hccl_engine import (  # noqa: E501
            P2pHcclEngine)
        return P2pHcclEngine(local_rank=local_rank,
                             config=config,
                             hostname=hostname,
                             port_offset=port_offset,
                             library_path=library_path)

    raise ValueError(f"P2P KV transfer does not support {device_type=}")
