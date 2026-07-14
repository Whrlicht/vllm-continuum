# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import logging
import os
import threading
import time
import typing
from typing import Any, Optional

import msgpack
import torch
import zmq

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.hccl_wrapper import (
    P2pHcclLibrary)
from vllm.utils import get_ip
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    hcclComm_t, hcclUniqueId)
from vllm_ascend.utils import current_stream

logger = logging.getLogger(__name__)
_LICHT_PROBE = os.environ.get("LICHT_PROBE") == "1"
_LICHT_KV_CHECK = os.environ.get("LICHT_KV_CHECK") == "1"


class P2pHcclEngine:
    """HCCL data-plane engine for P/D KV transfer on Ascend.

    This mirrors the NCCL engine's control-plane shape: workers exchange an
    HCCL root info through ZMQ, then create a 2-rank HCCL communicator for
    point-to-point tensor transfer.
    """

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 hostname: str = "",
                 port_offset: int = 0,
                 library_path: Optional[str] = None) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank
        self.device = torch.device(f"npu:{self.local_rank}")
        self.hccl = P2pHcclLibrary(library_path)

        if not hostname:
            hostname = get_ip()
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port
        self.zmq_address = f"{self._hostname}:{self._port}"
        self.http_address = (
            f"{self._hostname}:"
            f"{self.config.kv_connector_extra_config['http_port']}")

        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        self.proxy_address = "" if not proxy_ip or not proxy_port else (
            proxy_ip + ":" + proxy_port)

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.socks: dict[str, Any] = {}
        self.comms: dict[str, tuple[hcclComm_t, int]] = {}
        self._rpc_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.bridge_queue: dict[str, list[int]] = {}
        self.completed_recving_req_ids: set[str] = set()
        self.completed_release_req_ids: set[str] = set()
        self._delay_free_ts: dict[str, dict[str, float]] = {}
        self.recv_store: dict[str, torch.Tensor] = {}
        self.recv_store_cv = threading.Condition()
        self.send_store: dict[str, torch.Tensor] = {}
        self.send_store_cv = threading.Condition()
        self._layout_logged = False
        self._block_map = os.environ.get("LICHT_HCCL_BLOCK_MAP", "slot")

        self._listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True)
        self._listener_thread.start()

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address:
            self._ping_thread = threading.Thread(target=self.ping, daemon=True)
            self._ping_thread.start()

        logger.info(
            "P2pHcclEngine init, rank:%d, local_rank:%d, http_address:%s, "
            "zmq_address:%s, proxy_address:%s",
            self.rank, self.local_rank, self.http_address, self.zmq_address,
            self.proxy_address)

    @staticmethod
    def _unique_id_from_bytes(data: bytes) -> hcclUniqueId:
        unique_id = hcclUniqueId()
        if len(data) != ctypes.sizeof(unique_id):
            raise ValueError("Invalid HCCL root info size: "
                             f"{len(data)} != {ctypes.sizeof(unique_id)}")
        ctypes.memmove(ctypes.byref(unique_id), data, ctypes.sizeof(unique_id))
        return unique_id

    def create_connect(self, remote_address: typing.Optional[str] = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock

            with torch.npu.device(self.device):
                unique_id = self.hccl.hcclGetUniqueId()
                data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
                sock.send(msgpack.dumps(data))
                rank = 0
                comm = self.hccl.hcclCommInitRank(2, unique_id, rank)
                self.comms[remote_address] = (comm, rank)
                logger.info("HCCL comm init success, %s -> %s, rank=%s",
                            self.zmq_address, remote_address, rank)

        return self.socks[remote_address], self.comms[remote_address]

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.kv_caches = kv_caches
        if (_LICHT_PROBE or _LICHT_KV_CHECK) and not self._layout_logged:
            self._layout_logged = True
            for i, (layer_name, kv_cache) in enumerate(kv_caches.items()):
                if i >= 2:
                    break
                try:
                    block_dim = self._infer_block_dim(kv_cache)
                except Exception:
                    block_dim = -1
                logger.info(
                    "HCCL KV layout layer=%s shape=%s stride=%s dtype=%s "
                    "device=%s block_dim=%d block_map=%s", layer_name,
                    tuple(kv_cache.shape), tuple(kv_cache.stride()),
                    kv_cache.dtype, kv_cache.device, block_dim,
                    self._block_map)

    def stage_bridge_request(self, request_id: str,
                             context_block_ids: list[int]) -> None:
        with self.state_lock:
            self.bridge_queue[request_id] = context_block_ids
        self._delay_free_ts.setdefault(request_id, {})[
            "bridge_staged_ts"] = time.time()

    def was_bridge_staged(self, request_id: str) -> bool:
        ts = self._delay_free_ts.get(request_id)
        return ts is not None and "bridge_staged_ts" in ts

    def pop_bridge_request(self,
                           request_id: str,
                           remote_address: str,
                           timeout_s: Optional[float] = None
                           ) -> Optional[list[int]]:
        if timeout_s is None:
            timeout_s = 0.0

        deadline = time.time() + timeout_s
        while True:
            payload = self._rpc(remote_address, {
                "cmd": "BRIDGE_POP",
                "request_id": request_id,
            })
            if payload.get("ret") == 0:
                self._delay_free_ts.setdefault(request_id, {})[
                    "bridge_popped_ts"] = time.time()
                return [int(x) for x in payload.get("context_block_ids", [])]
            if timeout_s <= 0 or time.time() >= deadline:
                return None
            time.sleep(0.001)

    def set_arena_sink_handler(self, handler) -> None:
        self._arena_sink_handler = handler

    def send_arena_sink_request(self, request_id: str, job_id: str,
                                prompt_token_ids: list,
                                remote_address: str) -> bool:
        payload = self._rpc(remote_address, {
            "cmd": "ARENA_SINK",
            "request_id": request_id,
            "job_id": str(job_id),
            "token_ids": list(prompt_token_ids),
            "decode_zmq_address": self.zmq_address,
        })
        return payload.get("ret") == 0

    def _rpc(self, remote_address: str, payload: dict[str, Any]) -> dict[str,
                                                                         Any]:
        if remote_address not in self.socks:
            self.create_connect(remote_address)
        sock = self.socks[remote_address]
        with self._rpc_lock:
            sock.send(msgpack.dumps(payload))
            return msgpack.loads(sock.recv())

    def listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll(timeout=20))
            if self.router_socket not in socks:
                continue

            remote_address, message = self.router_socket.recv_multipart()
            data = msgpack.loads(message)
            if data["cmd"] == "NEW":
                unique_id = self._unique_id_from_bytes(
                    bytes(data["unique_id"]))
                with torch.npu.device(self.device):
                    rank = 1
                    comm = self.hccl.hcclCommInitRank(2, unique_id, rank)
                    self.comms[remote_address.decode()] = (comm, rank)
                    logger.info("HCCL comm init success, %s <- %s, rank=%s",
                                self.zmq_address, remote_address.decode(),
                                rank)
            elif data["cmd"] == "BRIDGE_POP":
                request_id = data["request_id"]
                with self.state_lock:
                    context_block_ids = self.bridge_queue.pop(request_id, None)
                if context_block_ids is None:
                    payload = {"ret": 1}
                    if _LICHT_PROBE:
                        logger.info(
                            "HCCL BRIDGE_POP miss req=%s queued=%s",
                            request_id, list(self.bridge_queue.keys())[:5])
                else:
                    payload = {
                        "ret": 0,
                        "context_block_ids": context_block_ids,
                    }
                    if _LICHT_PROBE:
                        logger.info("HCCL BRIDGE_POP hit req=%s blocks=%d",
                                    request_id, len(context_block_ids))
                self.router_socket.send_multipart(
                    [remote_address, msgpack.dumps(payload)])
            elif data["cmd"] == "BLOCK_MIGRATE":
                request_id = data["request_id"]
                context_block_ids = [int(x) for x in data["context_block_ids"]]
                logical_block_size = int(data.get("logical_block_size", 0))
                try:
                    layers = self._build_block_migration_meta(
                        context_block_ids, logical_block_size)
                    if _LICHT_PROBE:
                        logger.info(
                            "HCCL BLOCK_MIGRATE setup req=%s logical_blocks=%d "
                            "layers=%d logical_block_size=%d remote=%s",
                            request_id, len(context_block_ids), len(layers),
                            logical_block_size, remote_address.decode())
                    self.router_socket.send_multipart([
                        remote_address,
                        msgpack.dumps({
                            "ret": 0,
                            "layers": layers,
                        })
                    ])
                    sender = threading.Thread(
                        target=self._send_block_migration,
                        args=(request_id, context_block_ids,
                              remote_address.decode(), layers,
                              logical_block_size),
                        daemon=True)
                    sender.start()
                except Exception as e:
                    logger.exception("HCCL BLOCK_MIGRATE setup failed")
                    self.router_socket.send_multipart([
                        remote_address,
                        msgpack.dumps({
                            "ret": 1,
                            "error": str(e),
                        })
                    ])
            elif data["cmd"] == "ARENA_SINK":
                handler = getattr(self, "_arena_sink_handler", None)
                ok = False
                if handler is not None:
                    try:
                        ok = bool(handler(
                            data["request_id"], str(data.get("job_id", "")),
                            list(data.get("token_ids", [])),
                            data.get("decode_zmq_address", "")))
                    except Exception as e:
                        logger.warning("ARENA_SINK handler failed req=%s: %s",
                                       data.get("request_id"), e)
                self.router_socket.send_multipart([
                    remote_address,
                    msgpack.dumps({"ret": 0 if ok else 1})
                ])
            elif data["cmd"] == "RELEASE":
                request_id = data["request_id"]
                ts_entry = self._delay_free_ts.setdefault(request_id, {})
                ts_entry["release_received_ts"] = time.time()
                remote_ts = data.get("timestamps")
                if remote_ts and isinstance(remote_ts, dict):
                    ts_entry.update(remote_ts)
                with self.state_lock:
                    self.completed_release_req_ids.add(request_id)
                self.router_socket.send_multipart(
                    [remote_address, msgpack.dumps({"ret": 0})])
            elif data["cmd"] == "PUT":
                tensor_id = data["tensor_id"]
                dtype = getattr(torch, data["dtype"])
                tensor = torch.empty(data["shape"],
                                     dtype=dtype,
                                     device=self.device)
                self.router_socket.send_multipart([remote_address, b"0"])
                comm, rank = self.comms[remote_address.decode()]
                self.recv(comm, tensor, rank ^ 1)
                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.recv_store_cv.notify()
            elif data["cmd"] == "GET":
                tensor_id = data["tensor_id"]
                with self.send_store_cv:
                    tensor = self.send_store.get(tensor_id)
                    if tensor is None:
                        payload = {"ret": 1}
                    else:
                        payload = {
                            "ret": 0,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                        }
                self.router_socket.send_multipart(
                    [remote_address, msgpack.dumps(payload)])
                if payload["ret"] == 0:
                    comm, rank = self.comms[remote_address.decode()]
                    self.send(comm, tensor.to(self.device), rank ^ 1)
            else:
                self.router_socket.send_multipart([
                    remote_address,
                    msgpack.dumps({
                        "ret":
                        1,
                        "error":
                        f"unsupported_hccl_engine_cmd:{data.get('cmd')}",
                    })
                ])

    @staticmethod
    def _infer_block_dim(kv_cache: torch.Tensor) -> int:
        if kv_cache.dim() < 2:
            raise RuntimeError(
                f"Invalid KV cache shape for block migration: {kv_cache.shape}"
            )
        if kv_cache.shape[1] == 2:
            return 0
        if kv_cache.shape[0] == 2:
            return 1
        raise RuntimeError(
            "Unsupported KV cache layout for block migration, "
            f"shape={tuple(kv_cache.shape)}")

    def _build_block_migration_meta(
            self, context_block_ids: list[int],
            logical_block_size: int = 0) -> list[dict[str, Any]]:
        if not self.kv_caches:
            raise RuntimeError("KV caches are not registered")
        layers: list[dict[str, Any]] = []
        for layer_name, kv_cache in self.kv_caches.items():
            block_dim = self._infer_block_dim(kv_cache)
            physical_block_ids, physical_block_size = (
                self._expand_physical_block_ids(
                    kv_cache, block_dim, context_block_ids,
                    logical_block_size))
            num_blocks = len(physical_block_ids)
            shape = list(kv_cache.shape)
            shape[block_dim] = num_blocks
            layer_meta = {
                "name": layer_name,
                "shape": [int(x) for x in shape],
                "dtype": str(kv_cache.dtype).replace("torch.", ""),
                "block_dim": int(block_dim),
                "stride": [int(x) for x in kv_cache.stride()],
                "logical_block_size": int(logical_block_size),
                "physical_block_size": int(physical_block_size),
                "physical_num_blocks": int(num_blocks),
            }
            if _LICHT_KV_CHECK:
                tensor = self._gather_blocks(kv_cache, block_dim,
                                             physical_block_ids)
                layer_meta["signature"] = self._tensor_signature(tensor)
            layers.append(layer_meta)
        return layers

    @staticmethod
    def _infer_physical_block_size(kv_cache: torch.Tensor,
                                   block_dim: int) -> int:
        if kv_cache.shape[0] == 2:
            token_dim = 2 if block_dim == 1 else 1
        elif kv_cache.shape[1] == 2:
            token_dim = 2
        else:
            raise RuntimeError(
                "Unsupported KV cache layout for physical block inference, "
                f"shape={tuple(kv_cache.shape)}")
        if token_dim >= kv_cache.dim():
            raise RuntimeError(
                "Invalid KV cache token dimension for physical block "
                f"inference, shape={tuple(kv_cache.shape)}, "
                f"block_dim={block_dim}")
        return int(kv_cache.shape[token_dim])

    def _expand_physical_block_ids(self, kv_cache: torch.Tensor,
                                   block_dim: int,
                                   logical_block_ids: list[int],
                                   logical_block_size: int = 0
                                   ) -> tuple[list[int], int]:
        physical_block_size = self._infer_physical_block_size(kv_cache,
                                                              block_dim)
        if logical_block_size <= 0 or logical_block_size == physical_block_size:
            physical_block_ids = [int(x) for x in logical_block_ids]
            self._check_block_id_bounds(kv_cache, block_dim,
                                        physical_block_ids)
            return physical_block_ids, physical_block_size
        if logical_block_size % physical_block_size != 0:
            raise RuntimeError(
                "Cannot map logical KV blocks to physical KV blocks: "
                f"logical_block_size={logical_block_size}, "
                f"physical_block_size={physical_block_size}, "
                f"shape={tuple(kv_cache.shape)}")
        factor = logical_block_size // physical_block_size
        physical_block_ids: list[int] = []
        for block_id in logical_block_ids:
            logical_block_id = int(block_id)
            if self._block_map == "identity":
                physical_block_ids.append(logical_block_id)
                continue
            if self._block_map == "compact":
                base = (logical_block_id - 1) * factor
            else:
                # vLLM slot mapping is block_number * logical_block_size +
                # offset. Ascend stores the same slot range in smaller
                # physical KV blocks, so physical block ids scale by factor.
                base = logical_block_id * factor
            physical_block_ids.extend(base + offset for offset in range(factor))
        self._check_block_id_bounds(kv_cache, block_dim, physical_block_ids)
        return physical_block_ids, physical_block_size

    @staticmethod
    def _check_block_id_bounds(kv_cache: torch.Tensor, block_dim: int,
                               block_ids: list[int]) -> None:
        if not block_ids:
            return
        min_block = min(block_ids)
        max_block = max(block_ids)
        num_blocks = int(kv_cache.shape[block_dim])
        if min_block < 0 or max_block >= num_blocks:
            raise RuntimeError(
                "KV block id out of bounds: "
                f"ids=[{min_block}, {max_block}], "
                f"num_blocks={num_blocks}, shape={tuple(kv_cache.shape)}, "
                f"block_dim={block_dim}")

    @staticmethod
    def _gather_blocks(kv_cache: torch.Tensor, block_dim: int,
                       block_ids: list[int]) -> torch.Tensor:
        blocks = [
            kv_cache.select(block_dim, int(block_id))
            for block_id in block_ids
        ]
        return torch.stack(blocks, dim=block_dim).contiguous()

    @staticmethod
    def _scatter_blocks(kv_cache: torch.Tensor, block_dim: int,
                        block_ids: list[int],
                        recv_tensor: torch.Tensor) -> None:
        for src_idx, dst_block_id in enumerate(block_ids):
            kv_cache.select(block_dim, int(dst_block_id)).copy_(
                recv_tensor.select(block_dim, src_idx))

    @staticmethod
    def _zero_invalid_tail(kv_cache: torch.Tensor, block_dim: int,
                           block_ids: list[int], valid_tokens: int,
                           block_size: int = 0) -> None:
        if not block_ids:
            return
        if block_size <= 0:
            sample_block = kv_cache.select(block_dim, int(block_ids[-1]))
            if sample_block.dim() < 2:
                return
            block_size = int(sample_block.shape[1])
        if valid_tokens >= len(block_ids) * block_size:
            return
        valid_tokens = max(valid_tokens, 0)
        full_blocks = valid_tokens // block_size
        tail = valid_tokens % block_size

        first_clear_idx = full_blocks
        if tail > 0 and first_clear_idx < len(block_ids):
            trailing_block = kv_cache.select(block_dim,
                                             int(block_ids[first_clear_idx]))
            # KV cache block layout is [K/V row, block_size, ...] after
            # selecting the block for both supported layouts.
            if (trailing_block.dim() >= 2
                    and trailing_block.shape[1] >= block_size):
                trailing_block[:, tail:, ...].zero_()
            first_clear_idx += 1

        for block_id in block_ids[first_clear_idx:]:
            kv_cache.select(block_dim, int(block_id)).zero_()

    @staticmethod
    def _tensor_signature(tensor: torch.Tensor) -> dict[str, float | int]:
        flat = tensor.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {
                "numel": 0,
                "sum": 0.0,
                "abs_sum": 0.0,
                "max_abs": 0.0,
                "first": 0.0,
            }
        abs_flat = flat.abs()
        return {
            "numel": int(flat.numel()),
            "sum": float(flat.sum().item()),
            "abs_sum": float(abs_flat.sum().item()),
            "max_abs": float(abs_flat.max().item()),
            "first": float(flat[0].item()),
        }

    @staticmethod
    def _signature_close(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
        if int(lhs.get("numel", -1)) != int(rhs.get("numel", -2)):
            return False
        for key in ("sum", "abs_sum", "max_abs", "first"):
            a = float(lhs.get(key, 0.0))
            b = float(rhs.get(key, 0.0))
            tol = max(1e-2, 1e-5 * max(abs(a), abs(b), 1.0))
            if abs(a - b) > tol:
                return False
        return True

    @staticmethod
    def _format_signature(sig: dict[str, Any]) -> str:
        return (
            f"numel={int(sig.get('numel', -1))} "
            f"sum={float(sig.get('sum', 0.0)):.6g} "
            f"abs_sum={float(sig.get('abs_sum', 0.0)):.6g} "
            f"max_abs={float(sig.get('max_abs', 0.0)):.6g} "
            f"first={float(sig.get('first', 0.0)):.6g}")

    def _send_block_migration(self, request_id: str,
                              context_block_ids: list[int],
                              remote_address: str,
                              layers: list[dict[str, Any]],
                              logical_block_size: int = 0) -> None:
        try:
            comm, rank = self.comms[remote_address]
            layer_order = [layer["name"] for layer in layers]
            if _LICHT_PROBE:
                logger.info(
                    "HCCL block migration send start req=%s logical_blocks=%d "
                    "layers=%d logical_block_size=%d remote=%s", request_id,
                    len(context_block_ids), len(layer_order),
                    logical_block_size, remote_address)
            for layer_name in layer_order:
                kv_cache = self.kv_caches[layer_name]
                block_dim = self._infer_block_dim(kv_cache)
                physical_block_ids, physical_block_size = (
                    self._expand_physical_block_ids(
                        kv_cache, block_dim, context_block_ids,
                        logical_block_size))
                tensor = self._gather_blocks(kv_cache, block_dim,
                                             physical_block_ids)
                if _LICHT_PROBE or _LICHT_KV_CHECK:
                    logger.info(
                        "HCCL KV send layout req=%s layer=%s src_shape=%s "
                        "src_stride=%s gathered_shape=%s gathered_stride=%s "
                        "block_dim=%d logical_blocks=%s physical_blocks=%s "
                        "physical_block_size=%d", request_id, layer_name,
                        tuple(kv_cache.shape), tuple(kv_cache.stride()),
                        tuple(tensor.shape), tuple(tensor.stride()),
                        block_dim, context_block_ids[:8],
                        physical_block_ids[:12], physical_block_size)
                if _LICHT_KV_CHECK:
                    sig = self._tensor_signature(tensor)
                    logger.info(
                        "HCCL KV send checksum req=%s layer=%s %s",
                        request_id, layer_name,
                        self._format_signature(sig))
                self.send(comm, tensor, rank ^ 1)
            self._delay_free_ts.setdefault(request_id, {})[
                "migration_send_complete_ts"] = time.time()
            if _LICHT_PROBE:
                logger.info(
                    "HCCL block migration send done req=%s logical_blocks=%d "
                    "layers=%d remote=%s", request_id,
                    len(context_block_ids), len(layer_order), remote_address)
        except Exception:
            logger.exception("HCCL block migration send failed req=%s remote=%s",
                             request_id, remote_address)

    def launch_block_migration(self, request_id: str,
                               context_block_ids: list[int],
                               decoding_block_ids: list[int],
                               remote_address: str,
                               valid_external_tokens: int = 0,
                               logical_block_size: int = 0) -> bool:
        if not context_block_ids or not decoding_block_ids:
            with self.state_lock:
                self.completed_recving_req_ids.add(request_id)
            self._send_release_callback(request_id, remote_address)
            return True

        if len(context_block_ids) == len(decoding_block_ids) + 1:
            context_block_ids = context_block_ids[:len(decoding_block_ids)]

        num_pairs = min(len(context_block_ids), len(decoding_block_ids))
        if num_pairs != len(context_block_ids) or num_pairs != len(
                decoding_block_ids):
            logger.warning(
                "HCCL BLOCK_MIGRATE mismatched block_ids, req=%s, src=%d, "
                "dst=%d, copy=%d", request_id, len(context_block_ids),
                len(decoding_block_ids), num_pairs)
        context_block_ids = context_block_ids[:num_pairs]
        decoding_block_ids = decoding_block_ids[:num_pairs]

        try:
            t0 = time.time()
            if _LICHT_PROBE:
                logger.info(
                    "HCCL block migration recv start req=%s logical_src=%d "
                    "logical_dst=%d logical_block_size=%d remote=%s", request_id,
                    len(context_block_ids), len(decoding_block_ids),
                    logical_block_size, remote_address)
            _, (comm, rank) = self.create_connect(remote_address)
            payload = self._rpc(remote_address, {
                "cmd": "BLOCK_MIGRATE",
                "request_id": request_id,
                "context_block_ids": context_block_ids,
                "logical_block_size": int(logical_block_size),
            })
            if payload.get("ret") != 0:
                logger.warning("HCCL BLOCK_MIGRATE rejected req=%s: %s",
                               request_id, payload)
                return False

            first_error: Optional[str] = None
            copied_layers = 0
            for layer in payload.get("layers", []):
                layer_name = layer["name"]
                dtype = getattr(torch, layer["dtype"])
                recv_tensor = torch.empty(layer["shape"],
                                          dtype=dtype,
                                          device=self.device)
                self.recv(comm, recv_tensor, rank ^ 1)

                kv_cache = self.kv_caches.get(layer_name)
                if kv_cache is None:
                    if first_error is None:
                        first_error = (
                            f"local KV cache missing for layer {layer_name}")
                    continue

                remote_dtype = str(layer.get("dtype", ""))
                local_dtype = str(kv_cache.dtype).replace("torch.", "")
                if remote_dtype and remote_dtype != local_dtype:
                    if first_error is None:
                        first_error = (
                            "remote/local KV dtype mismatch for layer "
                            f"{layer_name}: remote={remote_dtype}, "
                            f"local={local_dtype}")
                    continue

                block_dim = self._infer_block_dim(kv_cache)
                physical_decoding_block_ids, physical_block_size = (
                    self._expand_physical_block_ids(
                        kv_cache, block_dim, decoding_block_ids,
                        logical_block_size))
                if int(layer.get("block_dim", block_dim)) != block_dim:
                    if first_error is None:
                        first_error = (
                            "remote/local KV block_dim mismatch for layer "
                            f"{layer_name}: remote={layer.get('block_dim')}, "
                            f"local={block_dim}")
                    continue
                expected_shape = list(kv_cache.shape)
                expected_shape[block_dim] = len(physical_decoding_block_ids)
                if list(recv_tensor.shape) != expected_shape:
                    if first_error is None:
                        first_error = (
                            "remote/local KV shape mismatch for layer "
                            f"{layer_name}: remote={list(recv_tensor.shape)}, "
                            f"expected={expected_shape}")
                    continue

                remote_sig = layer.get("signature")
                recv_sig: Optional[dict[str, Any]] = None
                if _LICHT_KV_CHECK and isinstance(remote_sig, dict):
                    recv_sig = self._tensor_signature(recv_tensor)
                    logger.info(
                        "HCCL KV recv checksum req=%s layer=%s %s",
                        request_id, layer_name,
                        self._format_signature(recv_sig))
                    if not self._signature_close(remote_sig, recv_sig):
                        if first_error is None:
                            first_error = (
                                "HCCL KV recv checksum mismatch for layer "
                                f"{layer_name}: remote="
                                f"{self._format_signature(remote_sig)}, "
                                f"recv={self._format_signature(recv_sig)}")
                        continue

                self._scatter_blocks(kv_cache, block_dim,
                                     physical_decoding_block_ids, recv_tensor)
                if _LICHT_PROBE or _LICHT_KV_CHECK:
                    logger.info(
                        "HCCL KV recv layout req=%s layer=%s dst_shape=%s "
                        "dst_stride=%s recv_shape=%s recv_stride=%s "
                        "block_dim=%d logical_dst=%s physical_dst=%s "
                        "physical_block_size=%d valid_external=%d",
                        request_id, layer_name,
                        tuple(kv_cache.shape), tuple(kv_cache.stride()),
                        tuple(recv_tensor.shape), tuple(recv_tensor.stride()),
                        block_dim, decoding_block_ids[:8],
                        physical_decoding_block_ids[:12],
                        physical_block_size, valid_external_tokens)
                if _LICHT_KV_CHECK:
                    written = self._gather_blocks(kv_cache, block_dim,
                                                  physical_decoding_block_ids)
                    written_sig = self._tensor_signature(written)
                    if recv_sig is None:
                        recv_sig = self._tensor_signature(recv_tensor)
                    logger.info(
                        "HCCL KV write checksum req=%s layer=%s %s",
                        request_id, layer_name,
                        self._format_signature(written_sig))
                    if not self._signature_close(recv_sig, written_sig):
                        if first_error is None:
                            first_error = (
                                "HCCL KV write checksum mismatch for layer "
                                f"{layer_name}: recv="
                                f"{self._format_signature(recv_sig)}, "
                                f"written="
                                f"{self._format_signature(written_sig)}")
                        continue
                self._zero_invalid_tail(kv_cache, block_dim,
                                        physical_decoding_block_ids,
                                        valid_external_tokens,
                                        physical_block_size)
                copied_layers += 1

            if first_error is not None:
                raise RuntimeError(first_error)
            if copied_layers == 0:
                raise RuntimeError("No HCCL KV layers were copied")

            current_stream().synchronize()
            self._delay_free_ts.setdefault(request_id, {})[
                "migration_complete_ts"] = time.time()
            with self.state_lock:
                self.completed_recving_req_ids.add(request_id)
            self._send_release_callback(request_id, remote_address)
            if _LICHT_PROBE:
                logger.info(
                    "HCCL block migration recv done req=%s logical_blocks=%d "
                    "layers=%d ms=%.1f remote=%s", request_id, num_pairs,
                    copied_layers, (time.time() - t0) * 1000.0,
                    remote_address)
            return True
        except Exception:
            logger.exception("HCCL block migration failed req=%s remote=%s",
                             request_id, remote_address)
            return False

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: typing.Optional[str] = None,
    ) -> bool:
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True

        self.create_connect(remote_address)
        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]
        payload = {
            "cmd": "PUT",
            "tensor_id": tensor_id,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        with self._rpc_lock:
            sock.send(msgpack.dumps(payload))
            ack = sock.recv()
        if ack != b"0":
            logger.warning("HCCL PUT rejected tensor_id=%s remote=%s",
                           tensor_id, remote_address)
            return False
        self.send(comm, tensor.to(self.device), rank ^ 1)
        return True

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        if remote_address is None:
            with self.recv_store_cv:
                return self.recv_store.pop(tensor_id, None)

        self.create_connect(remote_address)
        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]
        deadline = time.time() + float(
            self.config.get_from_extra_config("get_retry_timeout_s", 30.0))
        interval = max(
            float(
                self.config.get_from_extra_config("get_retry_interval_s",
                                                  0.002)), 1e-4)
        while True:
            with self._rpc_lock:
                sock.send(msgpack.dumps({
                    "cmd": "GET",
                    "tensor_id": tensor_id,
                }))
                payload = msgpack.loads(sock.recv())
            if payload.get("ret") == 0:
                tensor = torch.empty(payload["shape"],
                                     dtype=getattr(torch, payload["dtype"]),
                                     device=self.device)
                self.recv(comm, tensor, rank ^ 1)
                return tensor
            if time.time() >= deadline:
                logger.warning("HCCL GET timeout tensor_id=%s remote=%s",
                               tensor_id, remote_address)
                return None
            time.sleep(interval)

    def _send_release_callback(self, request_id: str,
                               remote_address: str,
                               extra_ts: Optional[dict[str, float]] = None
                               ) -> None:
        payload: dict[str, Any] = {
            "cmd": "RELEASE",
            "request_id": request_id,
        }
        if extra_ts:
            payload["timestamps"] = extra_ts
        resp = self._rpc(remote_address, payload)
        if resp.get("ret") != 0:
            logger.warning("HCCL RELEASE failed req=%s remote=%s resp=%s",
                           request_id, remote_address, resp)

    def wait_for_sent(self) -> None:
        return

    def get_finished(
            self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        with self.state_lock:
            finished_sending = set(self.completed_release_req_ids)
            finished_recving = set(self.completed_recving_req_ids)
            self.completed_release_req_ids.clear()
            self.completed_recving_req_ids.clear()
        return finished_sending or None, finished_recving or None

    def pop_delay_free_timestamps(
            self, req_ids: set[str]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for req_id in req_ids:
            ts = self._delay_free_ts.pop(req_id, None)
            if ts:
                result[req_id] = ts
        return result

    def send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this HCCL communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with torch.npu.stream(stream):
            self.hccl.hcclSend(tensor, dst, comm, stream)
        stream.synchronize()

    def recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this HCCL communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with torch.npu.stream(stream):
            self.hccl.hcclRecv(tensor, src, comm, stream)
        stream.synchronize()

    def ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address,
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)
