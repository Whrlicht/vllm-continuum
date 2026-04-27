# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import queue as queue_mod
import threading
import time
import typing
import ctypes
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import msgpack
import torch
import zmq

from vllm import _custom_ops as custom_ops
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary, cudaIpcMemHandle_t)
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary, buffer_type, cudaStream_t, ncclComm_t, ncclDataTypeEnum)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (  # noqa: E501
    TensorMemoryPool)
from vllm.utils import current_stream, get_ip

logger = logging.getLogger(__name__)

DEFAULT_MEM_POOL_SIZE_GB = 32
DEFAULT_REQUEST_COMPLETION_TIMEOUT_S = 120.0
DEFAULT_GET_RETRY_TIMEOUT_S = 30.0
DEFAULT_GET_RETRY_INTERVAL_S = 0.002
IPC_HANDLE_SIZE_BYTES = ctypes.sizeof(cudaIpcMemHandle_t)
DIRECT_BLOCK_SEND_TYPES = {"BLOCK_MIGRATE", "BLOCK_DIRECT", "DISTSERVE"}

# Module-level fast-release channel: listener thread → scheduler (same process).
# Populated by the producer-side listener on RELEASE; drained by the
# scheduler via poll_fast_release_queue().
_fast_release_queue: Optional[queue_mod.SimpleQueue] = None


def get_fast_release_queue() -> Optional[queue_mod.SimpleQueue]:
    return _fast_release_queue


@contextmanager
def set_p2p_nccl_context(num_channels: str):
    original_values: dict[str, Any] = {}
    env_vars = [
        'NCCL_MAX_NCHANNELS',
        'NCCL_MIN_NCHANNELS',
        'NCCL_CUMEM_ENABLE',
        'NCCL_BUFFSIZE',
        'NCCL_PROTO',  # LL,LL128,SIMPLE
        'NCCL_ALGO',  # RING,TREE
    ]

    for var in env_vars:
        original_values[var] = os.environ.get(var)

    logger.info("set_p2p_nccl_context, original_values: %s", original_values)

    try:
        os.environ['NCCL_MAX_NCHANNELS'] = num_channels
        os.environ['NCCL_MIN_NCHANNELS'] = num_channels
        os.environ['NCCL_CUMEM_ENABLE'] = '1'
        yield
    finally:
        for var in env_vars:
            if original_values[var] is not None:
                os.environ[var] = original_values[var]
            else:
                os.environ.pop(var, None)


@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor


class P2pNcclEngine:

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 hostname: str = "",
                 port_offset: int = 0,
                 library_path: Optional[str] = None) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.nccl = NCCLLibrary(library_path)

        if not hostname:
            hostname = get_ip()
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port

        # Each card corresponds to a ZMQ address.
        self.zmq_address = f"{self._hostname}:{self._port}"

        # The `http_port` must be consistent with the port of OpenAI.
        self.http_address = (
            f"{self._hostname}:"
            f"{self.config.kv_connector_extra_config['http_port']}")

        # If `proxy_ip` or `proxy_port` is `""`,
        # then the ping thread will not be enabled.
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        # The sending type includes tree mutually exclusive options:
        # PUT, GET, PUT_ASYNC, BLOCK_MIGRATE.
        self.send_type = str(
            self.config.get_from_extra_config("send_type",
                                              "PUT_ASYNC")).upper()
        self.direct_block_mode = self.send_type in DIRECT_BLOCK_SEND_TYPES

        if not self.direct_block_mode:
            mem_pool_size_gb = float(
                self.config.get_from_extra_config("mem_pool_size_gb",
                                                  DEFAULT_MEM_POOL_SIZE_GB))
            self.pool = TensorMemoryPool(max_block_size=int(mem_pool_size_gb *
                                                            1024**3))  # GB
        else:
            self.pool = None

        self.send_store: dict[str, torch.Tensor] = {}
        self.send_queue: deque[SendQueueItem] = deque()
        self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
        if self.send_type == "PUT_ASYNC":
            self._send_thread = threading.Thread(target=self.send_async,
                                                 daemon=True)
            self._send_thread.start()

        # tensor_id: torch.Tensor/(addr, dtype, shape)
        self.recv_store: dict[str, Any] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.socks: dict[str, Any] = {}  # remote_address: client socket
        self.comms: dict[str, Any] = {}  # remote_address: (ncclComm_t, rank)

        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)
        self.request_completion_timeout_s = float(
            self.config.get_from_extra_config(
                "request_completion_timeout_s",
                DEFAULT_REQUEST_COMPLETION_TIMEOUT_S,
            ))
        self.get_retry_timeout_s = float(
            self.config.get_from_extra_config(
                "get_retry_timeout_s",
                DEFAULT_GET_RETRY_TIMEOUT_S,
            ))
        self.get_retry_interval_s = max(
            float(
                self.config.get_from_extra_config(
                    "get_retry_interval_s",
                    DEFAULT_GET_RETRY_INTERVAL_S,
                )),
            1e-4,
        )
        self.state_lock = threading.Lock()
        self.expected_tensor_ids: dict[str, set[str]] = {}
        self.pending_sending_deadlines: dict[str, float] = {}
        self.pending_recving_deadlines: dict[str, float] = {}
        self.pending_release_deadlines: dict[str, float] = {}

        # Direct block migration runtime state.
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.cuda_rt: Optional[CudaRTLibrary] = None
        self.serialized_ipc_handles: Optional[dict[str, bytes]] = None
        self.serialized_ipc_meta: Optional[dict[str, dict[str, Any]]] = None
        self.local_ipc_epoch: int = 0
        self.remote_ipc_ptrs: dict[str, dict[str, ctypes.c_void_p]] = {}
        self.remote_kv_views: dict[str, dict[str, torch.Tensor]] = {}
        self.remote_ipc_epochs: dict[str, int] = {}
        self.bridge_queue: dict[str, list[int]] = {}
        self.pending_migrations: dict[str, tuple[torch.cuda.Event, str]] = {}
        self.completed_recving_req_ids: set[str] = set()
        self.completed_release_req_ids: set[str] = set()

        # Per-request delay-free timestamps collected on the worker side.
        # {req_id: {event_name: timestamp}}
        self._delay_free_ts: dict[str, dict[str, float]] = {}

        # Fast-release channel: only initialised on the producer side
        # so that the scheduler (same process, uniproc) can drain it.
        if self.config.is_kv_producer:
            global _fast_release_queue
            if _fast_release_queue is None:
                _fast_release_queue = queue_mod.SimpleQueue()

        self.nccl_num_channels = self.config.get_from_extra_config(
            "nccl_num_channels", "8")

        self._listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True)
        self._listener_thread.start()

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self.ping, daemon=True)
            self._ping_thread.start()

        # Background thread that polls cuda migration events and sends
        # RELEASE immediately, bypassing D-side forward step cadence.
        self._migration_poll_stop = threading.Event()
        self._migration_poll_thread: Optional[threading.Thread] = None
        if self.direct_block_mode:
            self._migration_poll_thread = threading.Thread(
                target=self._migration_poll_loop, daemon=True)
            self._migration_poll_thread.start()

        logger.info(
            "💯P2pNcclEngine init, rank:%d, local_rank:%d, http_address:%s, "
            "zmq_address:%s, proxy_address:%s, send_type:%s, buffer_size_"
            "threshold:%.2f, nccl_num_channels:%s, direct_block_mode:%s",
            self.rank, self.local_rank, self.http_address, self.zmq_address,
            self.proxy_address, self.send_type, self.buffer_size_threshold,
            self.nccl_num_channels, self.direct_block_mode)

    def create_connect(self, remote_address: typing.Optional[str] = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock

            if self.direct_block_mode:
                self.comms[remote_address] = (None, 0)
                return self.socks[remote_address], self.comms[remote_address]

            if remote_address in self.comms:
                logger.info("👋comm exists, remote_address:%s, comms:%s",
                            remote_address, self.comms)
                return sock, self.comms[remote_address]

            unique_id = self.nccl.ncclGetUniqueId()
            data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
            sock.send(msgpack.dumps(data))

            with torch.cuda.device(self.device):
                rank = 0
                with set_p2p_nccl_context(self.nccl_num_channels):
                    comm: ncclComm_t = self.nccl.ncclCommInitRank(
                        2, unique_id, rank)
                self.comms[remote_address] = (comm, rank)
                logger.info("🤝ncclCommInitRank Success, %s👉%s, MyRank:%s",
                            self.zmq_address, remote_address, rank)

        return self.socks[remote_address], self.comms[remote_address]

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.kv_caches = kv_caches
        # KV cache tensors may be re-initialized during engine lifetime.
        # Rebuild handles lazily on demand.
        self.serialized_ipc_handles = None
        self.serialized_ipc_meta = None
        self.local_ipc_epoch += 1

    def stage_bridge_request(self, request_id: str,
                             context_block_ids: list[int]) -> None:
        with self.state_lock:
            self.bridge_queue[request_id] = context_block_ids
        self._delay_free_ts.setdefault(request_id, {})[
            "bridge_staged_ts"] = time.time()

    def was_bridge_staged(self, request_id: str) -> bool:
        """Return True iff stage_bridge_request has ever been called for
        this request_id (i.e. bridge metadata was published to decode).

        Used by the connector's request_finished hook to distinguish:
          - normal completion (bridge staged, wait for decode RELEASE)
          - mid-prefill abort   (bridge never staged, free blocks now)

        Reads _delay_free_ts (single writer on the connector forward
        thread via stage_bridge_request).  We do not acquire state_lock
        because a miss here only degrades to "treat as abort", which is
        safe (free blocks locally; decode never saw this request anyway).
        """
        ts = self._delay_free_ts.get(request_id)
        return ts is not None and "bridge_staged_ts" in ts

    def pop_bridge_request(self,
                           request_id: str,
                           remote_address: str,
                           timeout_s: Optional[float] = None
                           ) -> Optional[list[int]]:
        if timeout_s is None:
            timeout_s = max(self.get_retry_timeout_s,
                            self.request_completion_timeout_s)

        # Fast path: single non-blocking probe, useful for per-step retries.
        if timeout_s <= 0:
            payload = self._rpc(
                remote_address,
                {
                    "cmd": "BRIDGE_POP",
                    "request_id": request_id,
                },
            )
            if payload.get("ret") == 0:
                self._delay_free_ts.setdefault(request_id, {})[
                    "bridge_popped_ts"] = time.time()
                return [int(x) for x in payload.get("context_block_ids", [])]
            return None

        deadline = time.time() + timeout_s
        while True:
            payload = self._rpc(
                remote_address,
                {
                    "cmd": "BRIDGE_POP",
                    "request_id": request_id,
                },
            )
            if payload.get("ret") == 0:
                self._delay_free_ts.setdefault(request_id, {})[
                    "bridge_popped_ts"] = time.time()
                return [int(x) for x in payload.get("context_block_ids", [])]

            if time.time() >= deadline:
                logger.warning(
                    "🔴[BLOCK]Bridge pop timeout from %s, request_id:%s, "
                    "rank:%d, timeout_s:%.3f", remote_address, request_id,
                    self.rank, timeout_s)
                return None
            time.sleep(self.get_retry_interval_s)

    def launch_block_migration(self, request_id: str,
                               context_block_ids: list[int],
                               decoding_block_ids: list[int],
                               remote_address: str) -> bool:
        if not self.direct_block_mode:
            raise RuntimeError("launch_block_migration only works in direct "
                               "block mode")

        if not context_block_ids or not decoding_block_ids:
            with self.state_lock:
                self.completed_recving_req_ids.add(request_id)
            self._send_release_callback(request_id, remote_address)
            return True

        # In disaggregated prefill/decode, decode side typically imports
        # prompt-1 tokens. This can lead to src blocks = dst blocks + 1 when
        # the last prompt token occupies a trailing partial block on prefill.
        # Trim that trailing source block to keep the migration one-to-one.
        if len(context_block_ids) == len(decoding_block_ids) + 1:
            context_block_ids = context_block_ids[:len(decoding_block_ids)]

        num_pairs = min(len(context_block_ids), len(decoding_block_ids))
        if num_pairs != len(context_block_ids) or num_pairs != len(
                decoding_block_ids):
            logger.warning(
                "🚧[BLOCK]Mismatched block_ids, req:%s, src:%d, dst:%d, "
                "copy:%d", request_id, len(context_block_ids),
                len(decoding_block_ids), num_pairs)

        src_ids = torch.tensor(context_block_ids[:num_pairs],
                               dtype=torch.int64,
                               device="cpu")
        dst_ids = torch.tensor(decoding_block_ids[:num_pairs],
                               dtype=torch.int64,
                               device="cpu")

        for attempt in range(2):
            try:
                self._ensure_remote_kv_views(
                    remote_address, force_refresh=(attempt > 0))
                remote_views = self.remote_kv_views.get(remote_address, {})
                if not remote_views:
                    raise RuntimeError("Remote KV views are not initialized "
                                       f"for {remote_address}")

                self._delay_free_ts.setdefault(request_id, {})[
                    "migration_launch_ts"] = time.time()
                event = torch.cuda.Event(blocking=False)
                with torch.cuda.stream(self.recv_stream):
                    for layer_name, dst_kv_cache in self.kv_caches.items():
                        src_kv_cache = remote_views.get(layer_name)
                        if src_kv_cache is None:
                            continue

                        block_dim = self._infer_block_dim(dst_kv_cache)
                        custom_ops.migrate_kv_cache_blocks(
                            dst_kv_cache, src_kv_cache, src_ids, dst_ids,
                            block_dim)

                    event.record(self.recv_stream)

                with self.state_lock:
                    self.pending_migrations[request_id] = (event,
                                                           remote_address)
                return True
            except Exception:
                if attempt == 0:
                    logger.warning(
                        "⚠️[BLOCK]Migration failed, refreshing IPC views and "
                        "retrying. req:%s remote:%s src:%d dst:%d",
                        request_id,
                        remote_address,
                        len(context_block_ids),
                        len(decoding_block_ids),
                        exc_info=True,
                    )
                    self._invalidate_remote_kv_views(remote_address)
                    continue

                logger.exception(
                    "🔴[BLOCK]Migration failed after IPC refresh. "
                    "req:%s remote:%s src:%d dst:%d",
                    request_id,
                    remote_address,
                    len(context_block_ids),
                    len(decoding_block_ids),
                )
                return False

        return False

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

    def _ensure_remote_kv_views(self,
                                remote_address: str,
                                force_refresh: bool = False) -> None:
        if force_refresh:
            self._invalidate_remote_kv_views(remote_address)
        elif remote_address in self.remote_kv_views:
            return

        payload = self._rpc(remote_address, {"cmd": "GET_IPC_METADATA"})
        if payload.get("ret") != 0:
            raise RuntimeError("Failed to fetch IPC metadata from "
                               f"{remote_address}: {payload}")

        handle_map: dict[str, bytes] = payload.get("handles", {})
        meta_map: dict[str, dict[str, Any]] = payload.get("meta", {})
        ipc_epoch = int(payload.get("ipc_epoch", 0))
        if not handle_map:
            raise RuntimeError("Peer returned empty IPC handle map from "
                               f"{remote_address}")
        if not meta_map:
            raise RuntimeError("Peer returned empty IPC metadata map from "
                               f"{remote_address}")

        if self.cuda_rt is None:
            self.cuda_rt = CudaRTLibrary()

        remote_ptrs: dict[str, ctypes.c_void_p] = {}
        remote_views: dict[str, torch.Tensor] = {}
        try:
            for layer_name, raw_handle in handle_map.items():
                local_kv_cache = self.kv_caches.get(layer_name)
                if local_kv_cache is None:
                    continue

                layer_meta = meta_map.get(layer_name)
                if layer_meta is None:
                    raise RuntimeError("Missing IPC metadata for layer "
                                       f"{layer_name} from {remote_address}")

                shape = [int(x) for x in layer_meta.get("shape", [])]
                stride = [int(x) for x in layer_meta.get("stride", [])]
                remote_dtype = str(layer_meta.get("dtype", ""))
                local_dtype = str(local_kv_cache.dtype).replace("torch.", "")
                if remote_dtype and remote_dtype != local_dtype:
                    raise RuntimeError(
                        "Remote/local KV dtype mismatch for layer "
                        f"{layer_name}: remote={remote_dtype}, local={local_dtype}"
                    )

                if not shape or len(shape) != len(stride):
                    raise RuntimeError(
                        "Invalid shape/stride metadata for layer "
                        f"{layer_name}: shape={shape}, stride={stride}")
                if len(shape) != local_kv_cache.dim():
                    raise RuntimeError(
                        "Remote/local KV dim mismatch for layer "
                        f"{layer_name}: remote_dim={len(shape)}, "
                        f"local_dim={local_kv_cache.dim()}")

                handle_bytes = raw_handle if isinstance(raw_handle,
                                                        (bytes,
                                                         bytearray)) else \
                    bytes(raw_handle)
                if len(handle_bytes) != IPC_HANDLE_SIZE_BYTES:
                    raise RuntimeError(
                        "Invalid IPC handle size for layer "
                        f"{layer_name}: {len(handle_bytes)}")

                handle = cudaIpcMemHandle_t()
                ctypes.memmove(ctypes.byref(handle), handle_bytes,
                               IPC_HANDLE_SIZE_BYTES)
                opened_ptr = self.cuda_rt.cudaIpcOpenMemHandle(handle)

                remote_ptrs[layer_name] = opened_ptr
                remote_views[layer_name] = \
                    custom_ops.get_cuda_view_from_ptr_shape_stride(
                        int(opened_ptr.value), shape, stride, local_kv_cache)
        except Exception:
            for ptr in remote_ptrs.values():
                try:
                    self.cuda_rt.cudaIpcCloseMemHandle(ptr)
                except Exception as exc:  # pragma: no cover - best effort
                    logger.debug("Failed to close IPC handle: %r", exc)
            raise

        if not remote_views:
            raise RuntimeError("No compatible remote KV views were created "
                               f"for {remote_address}")

        self.remote_ipc_ptrs[remote_address] = remote_ptrs
        self.remote_kv_views[remote_address] = remote_views
        self.remote_ipc_epochs[remote_address] = ipc_epoch

    def _invalidate_remote_kv_views(self, remote_address: str) -> None:
        ptrs_by_layer = self.remote_ipc_ptrs.pop(remote_address, None)
        self.remote_kv_views.pop(remote_address, None)
        self.remote_ipc_epochs.pop(remote_address, None)

        if ptrs_by_layer is None or self.cuda_rt is None:
            return

        for ptr in ptrs_by_layer.values():
            try:
                self.cuda_rt.cudaIpcCloseMemHandle(ptr)
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to close IPC handle: %r", exc)

    def _get_local_ipc_payload(self) -> dict[str, Any]:
        if self.serialized_ipc_handles is None:
            if not self.kv_caches:
                return {"ret": 1, "error": "kv_caches_not_registered"}
            if self.cuda_rt is None:
                self.cuda_rt = CudaRTLibrary()

            handles: dict[str, bytes] = {}
            meta: dict[str, dict[str, Any]] = {}
            for layer_name, kv_cache in self.kv_caches.items():
                dev_ptr = ctypes.c_void_p(int(kv_cache.data_ptr()))
                handle = self.cuda_rt.cudaIpcGetMemHandle(dev_ptr)
                handles[layer_name] = ctypes.string_at(ctypes.byref(handle),
                                                       IPC_HANDLE_SIZE_BYTES)
                meta[layer_name] = {
                    "shape": [int(x) for x in kv_cache.shape],
                    "stride": [int(x) for x in kv_cache.stride()],
                    "dtype": str(kv_cache.dtype).replace("torch.", ""),
                    "device": int(kv_cache.device.index or 0),
                    "block_dim": int(self._infer_block_dim(kv_cache)),
                }
            self.serialized_ipc_handles = handles
            self.serialized_ipc_meta = meta

        return {
            "ret": 0,
            "handles": self.serialized_ipc_handles,
            "meta": self.serialized_ipc_meta,
            "ipc_epoch": self.local_ipc_epoch,
        }

    def _rpc(self, remote_address: str, payload: dict[str, Any]) -> dict[str,
                                                                         Any]:
        if remote_address not in self.socks:
            self.create_connect(remote_address)
        sock = self.socks[remote_address]
        sock.send(msgpack.dumps(payload))
        return msgpack.loads(sock.recv())

    # ------------------------------------------------------------------
    # Background migration poll thread (Change 1)
    # ------------------------------------------------------------------

    def _migration_poll_loop(self) -> None:
        """Dedicated thread: poll cuda events, send RELEASE immediately."""
        poll_socks: dict[str, zmq.Socket] = {}
        poll_ctx = zmq.Context()

        while not self._migration_poll_stop.is_set():
            done: list[tuple[str, str]] = []
            with self.state_lock:
                for req_id, (event, addr) in list(
                        self.pending_migrations.items()):
                    if event.query():
                        done.append((req_id, addr))

            if not done:
                self._migration_poll_stop.wait(0.001)
                continue

            migration_complete_ts = time.time()
            for req_id, addr in done:
                decode_ts = self._delay_free_ts.pop(req_id, {})
                decode_ts["migration_complete_ts"] = migration_complete_ts
                decode_ts["release_callback_sent_ts"] = time.time()
                self._poll_thread_send_release(
                    poll_ctx, poll_socks, req_id, addr, decode_ts)

            with self.state_lock:
                for req_id, _ in done:
                    self.pending_migrations.pop(req_id, None)
                    self.completed_recving_req_ids.add(req_id)

        # Cleanup sockets on exit.
        for s in poll_socks.values():
            s.close(linger=0)
        poll_ctx.term()

    def _poll_thread_send_release(
        self,
        ctx: zmq.Context,
        socks: dict[str, zmq.Socket],
        request_id: str,
        remote_address: str,
        extra_ts: Optional[dict[str, float]],
    ) -> None:
        """Send RELEASE via the poll-thread's own ZMQ socket."""
        if remote_address not in socks:
            sock = ctx.socket(zmq.DEALER)
            identity = f"{self.zmq_address}#poll"
            sock.setsockopt_string(zmq.IDENTITY, identity)
            sock.connect(f"tcp://{remote_address}")
            socks[remote_address] = sock
        sock = socks[remote_address]
        rpc_payload: dict[str, Any] = {
            "cmd": "RELEASE",
            "request_id": request_id,
        }
        if extra_ts:
            rpc_payload["timestamps"] = extra_ts
        sock.send(msgpack.dumps(rpc_payload))
        resp = msgpack.loads(sock.recv())
        if resp.get("ret") != 0:
            logger.warning(
                "Poll-thread RELEASE failed, req:%s remote:%s resp:%s",
                request_id, remote_address, resp)

    def _send_release_callback(self, request_id: str,
                               remote_address: str,
                               extra_ts: Optional[dict[str, float]] = None,
                               ) -> None:
        rpc_payload: dict[str, Any] = {
            "cmd": "RELEASE",
            "request_id": request_id,
        }
        if extra_ts:
            rpc_payload["timestamps"] = extra_ts
        payload = self._rpc(remote_address, rpc_payload)
        if payload.get("ret") != 0:
            logger.warning(
                "🚧[BLOCK]Release callback failed, request_id:%s, "
                "remote:%s, payload:%s", request_id, remote_address, payload)

    def _poll_completed_migrations(self) -> None:
        # When the background poll thread is active, it handles cuda event
        # polling and RELEASE sending directly — skip here to avoid
        # double-processing.
        if self._migration_poll_thread is not None:
            return

        if not self.pending_migrations:
            return

        done: list[tuple[str, str]] = []
        with self.state_lock:
            pending_items = list(self.pending_migrations.items())

        for request_id, (event, remote_address) in pending_items:
            if event.query():
                done.append((request_id, remote_address))

        if not done:
            return

        migration_complete_ts = time.time()
        for request_id, remote_address in done:
            # Gather all decode-side timestamps for this request and
            # carry them inside the RELEASE RPC to the prefill side.
            decode_ts = self._delay_free_ts.pop(request_id, {})
            decode_ts["migration_complete_ts"] = migration_complete_ts
            decode_ts["release_callback_sent_ts"] = time.time()
            self._send_release_callback(request_id, remote_address,
                                        extra_ts=decode_ts)

        with self.state_lock:
            for request_id, _ in done:
                self.pending_migrations.pop(request_id, None)
                self.completed_recving_req_ids.add(request_id)

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

        item = SendQueueItem(tensor_id=tensor_id,
                             remote_address=remote_address,
                             tensor=tensor)
        self._mark_expected_tensor_id(tensor_id)

        if self.send_type == "PUT":
            return self.send_sync(item)

        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                self.send_queue.append(item)
                self.send_queue_cv.notify()
            return True

        # GET
        with self.send_store_cv:
            tensor_size = tensor.element_size() * tensor.numel()
            wait_deadline = time.time() + self.request_completion_timeout_s
            while (self.buffer_size + tensor_size
                   > self.buffer_size_threshold):
                now = time.time()
                remaining = wait_deadline - now
                if remaining <= 0:
                    logger.error(
                        "🔴[GET]Send buffer timeout, tensor_id:%s, "
                        "tensor_size:%d, buffer_size:%d, threshold:%d, "
                        "rank:%d",
                        tensor_id,
                        tensor_size,
                        self.buffer_size,
                        int(self.buffer_size_threshold),
                        self.rank,
                    )
                    return False
                self.send_store_cv.wait(timeout=min(remaining, 0.1))

            self.send_store[tensor_id] = tensor
            self.buffer_size += tensor_size
            logger.debug(
                "🔵[GET]Send to %s, tensor_id:%s, tensor_size:%d, "
                "shape:%s, rank:%d, buffer_size:%d(%.2f%%)", remote_address,
                tensor_id, tensor_size, tensor.shape, self.rank,
                self.buffer_size,
                self.buffer_size / self.buffer_size_threshold * 100)
        return True

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ) -> torch.Tensor:
        if self.send_type == "PUT" or self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.recv_store_cv:
                while tensor_id not in self.recv_store:
                    self.recv_store_cv.wait()
                tensor = self.recv_store[tensor_id]

            if tensor is not None:
                if isinstance(tensor, tuple):
                    addr, dtype, shape = tensor
                    tensor = self.pool.load_tensor(addr, dtype, shape,
                                                   self.device)
                else:
                    self.buffer_size -= (tensor.element_size() *
                                         tensor.numel())
            else:
                duration = time.time() - start_time
                logger.warning(
                    "🔴[PUT]Recv From %s, tensor_id:%s, duration:%.3fms, "
                    "rank:%d", remote_address, tensor_id, duration * 1000,
                    self.rank)
            return tensor

        # GET
        if remote_address is None:
            return None

        if remote_address not in self.socks:
            self.create_connect(remote_address)

        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]

        deadline = time.time() + self.get_retry_timeout_s
        data = {"cmd": "GET", "tensor_id": tensor_id}
        while True:
            sock.send(msgpack.dumps(data))

            message = sock.recv()
            payload = msgpack.loads(message)
            if payload["ret"] == 0:
                with torch.cuda.stream(self.recv_stream):
                    tensor = torch.empty(payload["shape"],
                                         dtype=getattr(torch,
                                                       payload["dtype"]),
                                         device=self.device)

                self.recv(comm, tensor, rank ^ 1, self.recv_stream)
                self.have_received_tensor_id(tensor_id)
                return tensor

            if time.time() >= deadline:
                logger.warning(
                    "🔴[GET]Recv timeout from %s, tensor_id:%s, rank:%d",
                    remote_address,
                    tensor_id,
                    self.rank,
                )
                return None
            time.sleep(self.get_retry_interval_s)

    def listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket not in socks:
                continue

            remote_address, message = self.router_socket.recv_multipart()
            data = msgpack.loads(message)
            if data["cmd"] == "NEW":
                unique_id = self.nccl.unique_id_from_bytes(
                    bytes(data["unique_id"]))
                with torch.cuda.device(self.device):
                    rank = 1
                    with set_p2p_nccl_context(self.nccl_num_channels):
                        comm: ncclComm_t = self.nccl.ncclCommInitRank(
                            2, unique_id, rank)
                    self.comms[remote_address.decode()] = (comm, rank)
                    logger.info("🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
                                self.zmq_address, remote_address.decode(),
                                rank)
            elif data["cmd"] == "GET_IPC_METADATA":
                payload = self._get_local_ipc_payload()
                self.router_socket.send_multipart(
                    [remote_address, msgpack.dumps(payload)])
            elif data["cmd"] == "BRIDGE_POP":
                request_id = data["request_id"]
                with self.state_lock:
                    context_block_ids = self.bridge_queue.pop(request_id, None)
                payload: dict[str, Any]
                if context_block_ids is None:
                    payload = {"ret": 1}
                else:
                    payload = {
                        "ret": 0,
                        "context_block_ids": context_block_ids,
                    }
                self.router_socket.send_multipart(
                    [remote_address, msgpack.dumps(payload)])
            elif data["cmd"] == "RELEASE":
                request_id = data["request_id"]
                release_received_ts = time.time()
                # Store timestamps carried by the decode-side RELEASE RPC
                # plus the local receive timestamp.
                ts_entry = self._delay_free_ts.setdefault(request_id, {})
                ts_entry["release_received_ts"] = release_received_ts
                remote_ts = data.get("timestamps")
                if remote_ts and isinstance(remote_ts, dict):
                    ts_entry.update(remote_ts)
                # Fast-release channel: push to module-level queue so that the
                # scheduler can drain it without waiting for the next forward
                # step's get_finished().  When the queue is active, do NOT
                # add to completed_release_req_ids — the fast path handles
                # block release exclusively, avoiding a race with the old
                # get_finished() path.
                if _fast_release_queue is not None:
                    _fast_release_queue.put_nowait(
                        (request_id, ts_entry.copy()))
                else:
                    with self.state_lock:
                        self.completed_release_req_ids.add(request_id)
                self.router_socket.send_multipart(
                    [remote_address, msgpack.dumps({"ret": 0})])
            elif data["cmd"] == "PUT":
                tensor_id = data["tensor_id"]
                try:
                    with torch.cuda.stream(self.recv_stream):
                        tensor = torch.empty(data["shape"],
                                             dtype=getattr(
                                                 torch, data["dtype"]),
                                             device=self.device)
                    self.router_socket.send_multipart([remote_address, b"0"])
                    comm, rank = self.comms[remote_address.decode()]
                    self.recv(comm, tensor, rank ^ 1, self.recv_stream)
                    tensor_size = tensor.element_size() * tensor.numel()
                    if (self.buffer_size + tensor_size
                            > self.buffer_size_threshold):
                        # Store Tensor in memory pool
                        addr = self.pool.store_tensor(tensor)
                        tensor = (addr, tensor.dtype, tensor.shape)
                        logger.warning(
                            "🔴[PUT]Recv Tensor, Out Of Threshold, "
                            "%s👈%s, data:%s, addr:%d", self.zmq_address,
                            remote_address.decode(), data, addr)
                    else:
                        self.buffer_size += tensor_size

                except torch.cuda.OutOfMemoryError:
                    self.router_socket.send_multipart([remote_address, b"1"])
                    tensor = None
                    logger.warning(
                        "🔴[PUT]Recv Tensor, Out Of Memory, %s👈%s, "
                        "data:%s", self.zmq_address, remote_address.decode(),
                        data)

                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.have_received_tensor_id(tensor_id)
                    self.recv_store_cv.notify()

            elif data["cmd"] == "GET":
                tensor_id = data["tensor_id"]
                with self.send_store_cv:
                    tensor = self.send_store.get(tensor_id)
                    if tensor is not None:
                        data = {
                            "ret": 0,
                            "shape": tensor.shape,
                            "dtype": str(tensor.dtype).replace("torch.", "")
                        }
                        self.have_sent_tensor_id(tensor_id)
                    else:
                        data = {"ret": 1}

                self.router_socket.send_multipart(
                    [remote_address, msgpack.dumps(data)])

                if data["ret"] == 0:
                    comm, rank = self.comms[remote_address.decode()]
                    self.send(comm, tensor.to(self.device), rank ^ 1,
                              self.send_stream)
            else:
                logger.warning(
                    "🚧Unexpected, Received message from %s, data:%s",
                    remote_address, data)

    def have_sent_tensor_id(self, tensor_id: str):
        request_id = self._request_id_from_tensor_id(tensor_id)
        with self.state_lock:
            if request_id not in self.send_request_id_to_tensor_ids:
                self.send_request_id_to_tensor_ids[request_id] = set()
            self.send_request_id_to_tensor_ids[request_id].add(tensor_id)

    def have_received_tensor_id(self, tensor_id: str):
        request_id = self._request_id_from_tensor_id(tensor_id)
        with self.state_lock:
            if request_id not in self.recv_request_id_to_tensor_ids:
                self.recv_request_id_to_tensor_ids[request_id] = set()
            self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)

    @staticmethod
    def _request_id_from_tensor_id(tensor_id: str) -> str:
        return tensor_id.split('#')[0]

    def _mark_expected_tensor_id(self, tensor_id: str) -> None:
        request_id = self._request_id_from_tensor_id(tensor_id)
        with self.state_lock:
            if request_id not in self.expected_tensor_ids:
                self.expected_tensor_ids[request_id] = set()
            self.expected_tensor_ids[request_id].add(tensor_id)

    def _request_is_transfer_complete(self,
                                      request_id: str,
                                      transfer_type: str) -> bool:
        if transfer_type == "send":
            expected = self.expected_tensor_ids.get(request_id, set())
            done = self.send_request_id_to_tensor_ids.get(request_id, set())
        else:
            expected = self.expected_tensor_ids.get(request_id, set())
            done = self.recv_request_id_to_tensor_ids.get(request_id, set())

        # No expected tensor means no async transfer to wait for.
        if not expected:
            return True
        return expected.issubset(done)

    def _cleanup_request_state(self,
                               request_id: str,
                               no_compile_layers: Optional[Iterable[str]]
                               ) -> None:
        with self.state_lock:
            tensor_ids = set()
            tensor_ids.update(self.expected_tensor_ids.pop(request_id, set()))
            tensor_ids.update(
                self.send_request_id_to_tensor_ids.pop(request_id, set()))
            tensor_ids.update(
                self.recv_request_id_to_tensor_ids.pop(request_id, set()))
            self.pending_sending_deadlines.pop(request_id, None)
            self.pending_recving_deadlines.pop(request_id, None)
            self.pending_release_deadlines.pop(request_id, None)
            self.bridge_queue.pop(request_id, None)
            self.pending_migrations.pop(request_id, None)
            self.completed_recving_req_ids.discard(request_id)
            self.completed_release_req_ids.discard(request_id)

        if no_compile_layers:
            for layer_name in no_compile_layers:
                tensor_ids.add(request_id + "#" + layer_name)

        request_prefix = request_id + "#"

        with self.send_store_cv:
            if self.send_type == "GET":
                if not tensor_ids:
                    tensor_ids.update(
                        tensor_id for tensor_id in self.send_store
                        if tensor_id.startswith(request_prefix))
                freed_size = 0
                for tensor_id in list(tensor_ids):
                    tensor = self.send_store.pop(tensor_id, None)
                    if tensor is None:
                        continue
                    freed_size += tensor.element_size() * tensor.numel()
                if freed_size > 0:
                    self.buffer_size = max(0, self.buffer_size - freed_size)
                self.send_store_cv.notify_all()

        with self.recv_store_cv:
            if not tensor_ids:
                tensor_ids.update(tensor_id for tensor_id in self.recv_store
                                  if tensor_id.startswith(request_prefix))
            for tensor_id in list(tensor_ids):
                tensor = self.recv_store.pop(tensor_id, None)
                if isinstance(tensor, tuple) and self.pool is not None:
                    addr, _, _ = tensor
                    self.pool.free(addr)
            self.recv_store_cv.notify_all()

        with self.send_queue_cv:
            if hasattr(self, "send_queue") and self.send_queue:
                self.send_queue = deque(
                    item for item in self.send_queue
                    if not item.tensor_id.startswith(request_prefix))
                self.send_queue_cv.notify_all()

    def send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                if not self.send_queue:
                    self.send_queue_cv.notify()
            self.send_sync(item)

    def wait_for_sent(self):
        if self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.send_queue_cv:
                while self.send_queue:
                    self.send_queue_cv.wait()
            duration = time.time() - start_time
            logger.debug(
                "🚧[PUT_ASYNC]It took %.3fms to wait for the send_queue"
                " to be empty, rank:%d", duration * 1000, self.rank)

    def send_sync(self, item: SendQueueItem) -> bool:
        if item.remote_address is None:
            return False
        if item.remote_address not in self.socks:
            self.create_connect(item.remote_address)

        tensor = item.tensor

        sock = self.socks[item.remote_address]
        comm, rank = self.comms[item.remote_address]
        data = {
            "cmd": "PUT",
            "tensor_id": item.tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", "")
        }
        sock.send(msgpack.dumps(data))

        response = sock.recv()
        if response != b"0":
            logger.error(
                "🔴Send Tensor, Peer Out Of Memory/Threshold, %s 👉 %s, "
                "MyRank:%s, data:%s, tensor:%s, size:%fGB, response:%s",
                self.zmq_address, item.remote_address, rank, data,
                tensor.shape,
                tensor.element_size() * tensor.numel() / 1024**3,
                response.decode())
            return False

        self.send(comm, tensor.to(self.device), rank ^ 1, self.send_stream)
        self.have_sent_tensor_id(item.tensor_id)

        return True

    def _get_finished_block_mode(
            self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        self._poll_completed_migrations()

        now = time.time()
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()

        with self.state_lock:
            if self.completed_recving_req_ids:
                finished_recving.update(self.completed_recving_req_ids)
                self.completed_recving_req_ids.clear()

            if self.config.is_kv_producer:
                # Producer can free local request state only after decode
                # sends RELEASE. This is required for bridge metadata/KV
                # correctness in direct block migration.
                if _fast_release_queue is not None:
                    # Fast-release path: RELEASE is consumed by scheduler via
                    # _fast_release_queue, so completed_release_req_ids is not
                    # populated. Track completion using release_received_ts
                    # written by the listener thread.
                    for request_id in finished_req_ids:
                        ts = self._delay_free_ts.get(request_id)
                        if ts is not None and "release_received_ts" in ts:
                            finished_sending.add(request_id)
                            self.pending_release_deadlines.pop(
                                request_id, None)
                            ts.setdefault("finished_sending_ts", now)
                        elif request_id not in self.pending_release_deadlines:
                            self.pending_release_deadlines[
                                request_id] = (
                                    now + self.request_completion_timeout_s)

                    for request_id, deadline in list(
                            self.pending_release_deadlines.items()):
                        ts = self._delay_free_ts.get(request_id)
                        if ts is not None and "release_received_ts" in ts:
                            finished_sending.add(request_id)
                            self.pending_release_deadlines.pop(
                                request_id, None)
                            ts.setdefault("finished_sending_ts", now)
                        elif now >= deadline:
                            logger.warning(
                                "⚠️RELEASE timeout (%.0fs), force-freeing "
                                "blocks for request_id:%s, rank:%d",
                                self.request_completion_timeout_s,
                                request_id, self.rank)
                            finished_sending.add(request_id)
                            self.pending_release_deadlines.pop(
                                request_id, None)
                            ts = self._delay_free_ts.setdefault(
                                request_id, {})
                            ts.setdefault("finished_sending_ts", now)
                            ts["release_timeout"] = True

                    pending_sending = set(
                        self.pending_release_deadlines.keys())
                else:
                    # Old path: producer can free blocks only after decode
                    # confirms migration via release callback.
                    for request_id in finished_req_ids:
                        if request_id in self.completed_release_req_ids:
                            finished_sending.add(request_id)
                            self.completed_release_req_ids.discard(request_id)
                            self.pending_release_deadlines.pop(
                                request_id, None)
                            self._delay_free_ts.setdefault(
                                request_id, {})["finished_sending_ts"] = now
                        elif request_id not in self.pending_release_deadlines:
                            self.pending_release_deadlines[
                                request_id] = (
                                    now + self.request_completion_timeout_s)

                    for request_id, deadline in list(
                            self.pending_release_deadlines.items()):
                        if request_id in self.completed_release_req_ids:
                            finished_sending.add(request_id)
                            self.completed_release_req_ids.discard(request_id)
                            self.pending_release_deadlines.pop(
                                request_id, None)
                            self._delay_free_ts.setdefault(
                                request_id, {})["finished_sending_ts"] = now
                        elif now >= deadline:
                            logger.warning(
                                "⚠️RELEASE timeout (%.0fs), force-freeing "
                                "blocks for request_id:%s, rank:%d",
                                self.request_completion_timeout_s,
                                request_id, self.rank)
                            finished_sending.add(request_id)
                            self.pending_release_deadlines.pop(
                                request_id, None)
                            self._delay_free_ts.setdefault(
                                request_id, {})["finished_sending_ts"] = now
                            self._delay_free_ts.setdefault(
                                request_id, {})["release_timeout"] = True

                    pending_sending = set(
                        self.pending_release_deadlines.keys())
            else:
                # Consumer does not wait for release callbacks and should not
                # report finished_sending from timeout paths.
                self.pending_release_deadlines.clear()
                self.completed_release_req_ids.clear()
                pending_sending = set()

        cleanup_req_ids = (set(finished_req_ids) - pending_sending
                           ) | finished_sending | finished_recving
        for request_id in cleanup_req_ids:
            self._cleanup_request_state(request_id, no_compile_layers)

        return finished_sending or None, finished_recving or None

    def get_finished(
            self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        if self.direct_block_mode:
            return self._get_finished_block_mode(finished_req_ids,
                                                 no_compile_layers)

        now = time.time()
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()

        with self.state_lock:
            for request_id in finished_req_ids:
                if (request_id in self.expected_tensor_ids
                        and request_id not in self.pending_sending_deadlines):
                    self.pending_sending_deadlines[
                        request_id] = now + self.request_completion_timeout_s

                if (request_id in self.recv_request_id_to_tensor_ids
                        and request_id not in self.pending_recving_deadlines):
                    self.pending_recving_deadlines[
                        request_id] = now + self.request_completion_timeout_s

            for request_id, deadline in list(
                    self.pending_sending_deadlines.items()):
                if self._request_is_transfer_complete(request_id, "send"):
                    finished_sending.add(request_id)
                    self.pending_sending_deadlines.pop(request_id, None)
                    self._delay_free_ts.setdefault(request_id, {})[
                        "send_complete_ts"] = now
                elif now >= deadline:
                    sent = len(
                        self.send_request_id_to_tensor_ids.get(request_id,
                                                              set()))
                    expected = len(self.expected_tensor_ids.get(request_id,
                                                                set()))
                    logger.warning(
                        "⚠️KV send completion timeout, request_id:%s, "
                        "sent:%d expected:%d, rank:%d",
                        request_id,
                        sent,
                        expected,
                        self.rank,
                    )
                    finished_sending.add(request_id)
                    self.pending_sending_deadlines.pop(request_id, None)

            for request_id, deadline in list(
                    self.pending_recving_deadlines.items()):
                if self._request_is_transfer_complete(request_id, "recv"):
                    finished_recving.add(request_id)
                    self.pending_recving_deadlines.pop(request_id, None)
                elif now >= deadline:
                    recved = len(
                        self.recv_request_id_to_tensor_ids.get(request_id,
                                                              set()))
                    expected = len(self.expected_tensor_ids.get(request_id,
                                                                set()))
                    logger.warning(
                        "⚠️KV recv completion timeout, request_id:%s, "
                        "recved:%d expected:%d, rank:%d",
                        request_id,
                        recved,
                        expected,
                        self.rank,
                    )
                    finished_recving.add(request_id)
                    self.pending_recving_deadlines.pop(request_id, None)

            pending_sending = set(self.pending_sending_deadlines.keys())

        # Keep producer-side KV tensors until migration/pull is acknowledged.
        cleanup_req_ids = (set(finished_req_ids) - pending_sending
                           ) | finished_sending | finished_recving
        for request_id in cleanup_req_ids:
            self._cleanup_request_state(request_id, no_compile_layers)

        return finished_sending or None, finished_recving or None

    def pop_delay_free_timestamps(
            self, req_ids: set[str]) -> dict[str, dict[str, float]]:
        """Return and clear delay-free timestamps for the given requests."""
        result: dict[str, dict[str, float]] = {}
        for req_id in req_ids:
            ts = self._delay_free_ts.pop(req_id, None)
            if ts:
                result[req_id] = ts
        return result

    def ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.debug("ping start, zmq_address:%s", self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)

    def send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            self.nccl.ncclSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), dst,
                               comm, cudaStream_t(stream.cuda_stream))
        stream.synchronize()

    def recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            self.nccl.ncclRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), src,
                               comm, cudaStream_t(stream.cuda_stream))
        stream.synchronize()

    def close(self) -> None:
        if self.cuda_rt is not None:
            for remote_address in list(self.remote_ipc_ptrs.keys()):
                self._invalidate_remote_kv_views(remote_address)
        self._listener_thread.join()
        if self.send_type == "PUT_ASYNC" and hasattr(self, "_send_thread"):
            self._send_thread.join()
        if self._ping_thread is not None:
            self._ping_thread.join()
