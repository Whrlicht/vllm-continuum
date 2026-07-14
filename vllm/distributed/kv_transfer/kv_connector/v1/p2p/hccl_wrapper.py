# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes

import torch

from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    HCCLLibrary, aclrtStream_t, buffer_type, hcclComm_t, hcclDataTypeEnum,
    hcclDataType_t, hcclResult_t)


class P2pHcclLibrary(HCCLLibrary):
    """HCCL wrapper additions needed by the P2P KV connector."""

    def __init__(self, so_file: str | None = None):
        super().__init__(so_file)
        self._register_p2p_function(
            "HcclSend",
            [
                buffer_type,
                ctypes.c_size_t,
                hcclDataType_t,
                ctypes.c_int,
                hcclComm_t,
                aclrtStream_t,
            ],
        )
        self._register_p2p_function(
            "HcclRecv",
            [
                buffer_type,
                ctypes.c_size_t,
                hcclDataType_t,
                ctypes.c_int,
                hcclComm_t,
                aclrtStream_t,
            ],
        )

    def _register_p2p_function(self, name: str, argtypes: list[object]) -> None:
        if name in self._funcs:
            return
        fn = getattr(self.lib, name)
        fn.restype = hcclResult_t
        fn.argtypes = argtypes
        self._funcs[name] = fn

    def hcclSend(self, tensor: torch.Tensor, dst: int, comm: hcclComm_t,
                 stream) -> None:
        self.HCCL_CHECK(
            self._funcs["HcclSend"](
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                hcclDataTypeEnum.from_torch(tensor.dtype),
                dst,
                comm,
                aclrtStream_t(stream.npu_stream),
            ))

    def hcclRecv(self, tensor: torch.Tensor, src: int, comm: hcclComm_t,
                 stream) -> None:
        self.HCCL_CHECK(
            self._funcs["HcclRecv"](
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                hcclDataTypeEnum.from_torch(tensor.dtype),
                src,
                comm,
                aclrtStream_t(stream.npu_stream),
            ))
