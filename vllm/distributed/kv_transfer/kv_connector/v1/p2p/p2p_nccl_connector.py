# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import regex as re
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Request Id
    request_id: str
    # Request block ids
    block_ids: torch.Tensor
    # Request num tokens
    num_tokens: int
    # Optional explicit remote endpoints.
    remote_prefill_address: Optional[str] = None
    remote_decode_address: Optional[str] = None

    @staticmethod
    def make_meta(request_id: str, token_ids: list[int], block_ids: list[int],
                  block_size: int,
                  remote_prefill_address: Optional[str] = None,
                  remote_decode_address: Optional[str] = None) -> "ReqMeta":
        block_ids_tensor = torch.tensor(block_ids)
        return ReqMeta(
            request_id=request_id,
            block_ids=block_ids_tensor,
            num_tokens=len(token_ids),
            remote_prefill_address=remote_prefill_address,
            remote_decode_address=remote_decode_address,
        )


@dataclass
class P2pNcclConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        remote_prefill_address: Optional[str] = None,
        remote_decode_address: Optional[str] = None,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(
                request_id,
                token_ids,
                block_ids,
                block_size,
                remote_prefill_address,
                remote_decode_address,
            ))


class P2pNcclConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer
        self.send_type = str(
            self.config.get_from_extra_config("send_type",
                                              "PUT_ASYNC")).upper()
        self.direct_block_mode = self.send_type in {
            "BLOCK_MIGRATE", "BLOCK_DIRECT", "DISTSERVE"
        }
        self.chunked_prefill: dict[str, Any] = {}
        self._pending_bridge_reqs: list[tuple[str, list[int]]] = []
        self._pending_failed_block_migrations: dict[
            str, tuple[list[int], list[int], str]
        ] = {}

        self._rank = get_world_group().rank \
            if role == KVConnectorRole.WORKER else 0
        self._local_rank = get_world_group().local_rank \
            if role == KVConnectorRole.WORKER else 0

        self.p2p_nccl_engine = P2pNcclEngine(
            local_rank=self._local_rank,
            config=self.config,
            hostname="",
            port_offset=self._rank,
        ) if role == KVConnectorRole.WORKER else None

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if self.p2p_nccl_engine is not None:
            self.p2p_nccl_engine.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """

        assert self.p2p_nccl_engine is not None

        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)

        if self.direct_block_mode:
            if self.is_producer:
                # Bridge publication must happen after prefill forward.
                self._pending_bridge_reqs.extend((
                    req.request_id,
                    [int(x) for x in req.block_ids.tolist()],
                ) for req in metadata.requests)
                return

            for req_meta in metadata.requests:
                remote_address = req_meta.remote_prefill_address
                if remote_address is None:
                    ip, port = self.parse_request_id(req_meta.request_id,
                                                     is_prefill=False)
                    remote_address = f"{ip}:{port + self._rank}"

                decoding_block_ids = [int(x) for x in req_meta.block_ids.tolist()]
                pending_migration = self._pending_failed_block_migrations.get(
                    req_meta.request_id)

                if pending_migration is not None:
                    context_block_ids, pending_decoding_block_ids, \
                        pending_remote_address = pending_migration
                    if pending_remote_address != remote_address:
                        logger.warning(
                            "⚠️[BLOCK]Remote address changed while retrying "
                            "migration for req:%s, old:%s, new:%s",
                            req_meta.request_id,
                            pending_remote_address,
                            remote_address,
                        )
                    remote_address = pending_remote_address
                    decoding_block_ids = pending_decoding_block_ids
                else:
                    # Pure decode-pull mode: fetch bridge metadata via a
                    # single non-blocking BRIDGE_POP probe.  If the
                    # producer has not staged yet, stay in
                    # WAITING_FOR_REMOTE_KVS and retry next forward step.
                    context_block_ids = \
                        self.p2p_nccl_engine.pop_bridge_request(
                            req_meta.request_id,
                            remote_address,
                            timeout_s=0.0,
                        )
                    if context_block_ids is None:
                        continue

                migrated = self.p2p_nccl_engine.launch_block_migration(
                    req_meta.request_id,
                    context_block_ids,
                    decoding_block_ids,
                    remote_address,
                )
                if migrated:
                    self._pending_failed_block_migrations.pop(
                        req_meta.request_id, None)
                else:
                    # Keep request in waiting-for-remote-kv and retry next step
                    # with the same bridge metadata.
                    self._pending_failed_block_migrations[req_meta.request_id] = (
                        context_block_ids,
                        decoding_block_ids,
                        remote_address,
                    )
            return

        # Legacy layer-wise GET/PUT path.
        # Only consumer/decode loads KV Cache.
        if self.is_producer:
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        def inject_kv_into_layer(
            layer: torch.Tensor,
            kv_cache: torch.Tensor,
            block_ids: torch.Tensor,
            request_id: str,
        ) -> None:
            """
            Inject KV cache data into a given attention layer tensor.

            This function updates `layer` in-place with values from `kv_cache`,
            handling different backend layouts:
              - MLA (Multi-Linear Attention) or FlashInfer: KV tensors are
                indexed along the first dimension.
              - FlashAttention: KV tensors are indexed along the second
                dimension.

            If the number of provided block IDs does not match the number of KV
            blocks, only the overlapping portion is updated, and a warning is
            logged.

            Args:
                layer (torch.Tensor): The attention layer KV tensor to update.
                kv_cache (torch.Tensor): The KV cache tensor to inject.
                block_ids (torch.Tensor): Indices of the blocks to update.
                request_id (str): Request identifier used for logging.

            Returns:
                None. The function modifies `layer` in-place.
            """
            if (isinstance(attn_metadata, MLACommonMetadata)
                    or layer.shape[1] == 2):  # MLA or FlashInfer
                num_block = kv_cache.shape[0]
                self.check_tensors_except_dim(layer, kv_cache, 0)
                if len(block_ids) == num_block:
                    layer[block_ids, ...] = kv_cache
                else:
                    layer[block_ids[:num_block], ...] = kv_cache
                    logger.warning(
                        "🚧kv_cache does not match, block_ids:%d, "
                        "num_block:%d, request_id:%s", len(block_ids),
                        num_block, request_id)

            elif layer.shape[0] == 2:  # FlashAttention
                num_block = kv_cache.shape[1]
                self.check_tensors_except_dim(layer, kv_cache, 1)
                if len(block_ids) == num_block:
                    layer[:, block_ids, ...] = kv_cache
                else:
                    layer[:, block_ids[:num_block], ...] = kv_cache
                    logger.warning(
                        "🚧kv_cache does not match, block_ids:%d, "
                        "num_block:%d, request_id:%s", len(block_ids),
                        num_block, request_id)

        # Load the KV for each request each layer
        for request in metadata.requests:
            ip, port = self.parse_request_id(request.request_id,
                                             is_prefill=False)
            remote_address = ip + ":" + str(port + self._rank)
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                # Only process layers that have kv_cache
                # attribute (attention layers) Skip non-attention
                # layers like FusedMoE
                kv_cache = getattr(layer, 'kv_cache', None)
                if kv_cache is None:
                    continue

                layer = kv_cache[forward_context.virtual_engine]

                kv_cache = self.p2p_nccl_engine.recv_tensor(
                    request.request_id + "#" + layer_name,
                    remote_address=remote_address)

                if kv_cache is None:
                    raise RuntimeError(
                        "Missing remote KV cache for request "
                        f"{request.request_id}, layer {layer_name} from "
                        f"{remote_address}")

                inject_kv_into_layer(layer, kv_cache, request.block_ids,
                                     request.request_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        if self.direct_block_mode:
            return

        # Only producer/prefill saves KV Cache
        if not self.is_producer:
            return

        assert self.p2p_nccl_engine is not None

        def extract_kv_from_layer(
            layer: torch.Tensor,
            block_ids: torch.Tensor,
        ) -> torch.Tensor:
            """
            Extract KV cache slices from a given attention layer tensor.

            This function handles multiple backend layouts:
              - MLA (Multi-Linear Attention) or FlashInfer: KV tensors are
                indexed along the first dimension.
              - FlashAttention: KV tensors are indexed along the second
                dimension.

            Args:
                layer (torch.Tensor): The KV cache from the attention layer.
                block_ids (torch.Tensor): Indices of blocks to extract.

            Returns:
                torch.Tensor: A tensor containing the extracted KV slices.
                Returns None if the layout is unsupported.
            """
            if (isinstance(attn_metadata, MLACommonMetadata)
                    or layer.shape[1] == 2):  # MLA or FlashInfer
                return layer[block_ids, ...]

            if layer.shape[0] == 2:  # FlashAttention
                return layer[:, block_ids, ...]

            return None

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, P2pNcclConnectorMetadata)
        for request in connector_metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, True)
            remote_address = ip + ":" + str(port + self._rank)

            kv_cache = extract_kv_from_layer(kv_layer, request.block_ids)
            ok = self.p2p_nccl_engine.send_tensor(request_id + "#" +
                                                  layer_name, kv_cache,
                                                  remote_address)
            if not ok:
                raise RuntimeError(
                    "Failed to stage KV cache for request "
                    f"{request_id}, layer {layer_name}")

    def wait_for_save(self):
        if self.is_producer:
            assert self.p2p_nccl_engine is not None
            if self.direct_block_mode:
                # Pure decode-pull mode: only stage bridge metadata
                # locally.  Decode will fetch via BRIDGE_POP RPC in its
                # own forward step.
                for request_id, context_block_ids in self._pending_bridge_reqs:
                    self.p2p_nccl_engine.stage_bridge_request(
                        request_id, context_block_ids)
                self._pending_bridge_reqs.clear()
                return
            self.p2p_nccl_engine.wait_for_sent()

    def get_finished(
            self, finished_req_ids: set[str],
            **kwargs) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        assert self.p2p_nccl_engine is not None

        no_compile_layers = (
            self._vllm_config.compilation_config.static_forward_context)
        return self.p2p_nccl_engine.get_finished(finished_req_ids,
                                                 no_compile_layers)

    def pop_delay_free_timestamps(
            self, req_ids: set[str]) -> dict[str, dict[str, float]]:
        if self.p2p_nccl_engine is not None:
            return self.p2p_nccl_engine.pop_delay_free_timestamps(req_ids)
        return {}

    @staticmethod
    def poll_fast_releases() -> list[tuple[str, dict[str, float]]]:
        """Drain the fast-release queue (scheduler-side, Change 3).

        Returns a list of (request_id, timestamps) for requests whose
        RELEASE has been received by the listener thread.
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
            get_fast_release_queue)
        q = get_fast_release_queue()
        if q is None:
            return []
        released: list[tuple[str, dict[str, float]]] = []
        while True:
            try:
                item = q.get_nowait()
                released.append(item)
            except Exception:
                break
        return released

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.is_producer:
            return 0, False

        num_external_tokens = (len(request.prompt_token_ids) - 1 -
                               num_computed_tokens)

        if num_external_tokens < 0:
            num_external_tokens = 0

        if self.direct_block_mode and num_external_tokens > 0:
            return num_external_tokens, True

        return num_external_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request, blocks.get_block_ids()[0])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        meta = P2pNcclConnectorMetadata()

        if not self.is_producer and self.direct_block_mode:
            for req_id, (request, local_block_ids) in \
                    self._requests_need_load.items():
                remote_prefill_address: Optional[str] = None
                remote_decode_address: Optional[str] = None

                kv_params = request.kv_transfer_params
                if isinstance(kv_params, dict):
                    remote_prefill_address = kv_params.get(
                        "prefill_zmq_address")
                    remote_decode_address = kv_params.get(
                        "decode_zmq_address")

                if remote_prefill_address is None:
                    try:
                        ip, port = self.parse_request_id(req_id,
                                                         is_prefill=False)
                        remote_prefill_address = f"{ip}:{port + self._rank}"
                    except Exception:
                        remote_prefill_address = None

                meta.add_request(
                    request_id=req_id,
                    token_ids=request.prompt_token_ids,
                    block_ids=local_block_ids,
                    block_size=self._block_size,
                    remote_prefill_address=remote_prefill_address,
                    remote_decode_address=remote_decode_address,
                )
            return meta

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[new_req.req_id]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                # the request's prompt is chunked prefill
                if num_tokens < len(new_req.prompt_token_ids):
                    # 'CachedRequestData' has no attribute 'prompt_token_ids'
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids[0], new_req.prompt_token_ids)
                    continue
                # the request's prompt is not chunked prefill
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size)
                continue
            if new_req.req_id in self._requests_need_load:
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size)
                self._requests_need_load.pop(new_req.req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = cached_reqs.resumed_from_preemption[i]

            if self.is_producer:
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[req_id]
                num_tokens = (num_scheduled_tokens + num_computed_tokens)
                if req_id not in self.chunked_prefill:
                    logger.warning(
                        "⚠️[PREFILL]Missing chunked_prefill state, req_id:%s, "
                        "resumed:%s", req_id, resumed_from_preemption)
                    continue

                if new_block_ids is None or not new_block_ids \
                        or new_block_ids[0] is None:
                    # No newly allocated block can be normal when the newly
                    # scheduled tokens still fit into existing blocks.
                    # For resumed requests we still need a full block list;
                    # keep waiting if scheduler cannot provide it.
                    if resumed_from_preemption:
                        logger.warning(
                            "⚠️[PREFILL]Resumed req has no new_block_ids, "
                            "req_id:%s, num_computed_tokens:%d", req_id,
                            num_computed_tokens)
                        continue
                    new_block_ids_0: list[int] = []
                else:
                    new_block_ids_0 = new_block_ids[0]

                block_ids = new_block_ids_0
                if not resumed_from_preemption:
                    block_ids = (self.chunked_prefill[req_id][0] + block_ids)
                prompt_token_ids = self.chunked_prefill[req_id][1]
                # the request's prompt is chunked prefill again
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids,
                                                    prompt_token_ids)
                    continue
                # the request's prompt is all prefilled finally
                meta.add_request(request_id=req_id,
                                 token_ids=prompt_token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size)
                self.chunked_prefill.pop(req_id, None)
                continue

            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                if new_block_ids is None or not new_block_ids \
                        or new_block_ids[0] is None:
                    logger.warning(
                        "⚠️[DECODE]No new_block_ids for resumed request, "
                        "req_id:%s, num_computed_tokens:%d", req_id,
                        num_computed_tokens)
                    continue

                request, _ = self._requests_need_load.pop(req_id)
                total_tokens = num_computed_tokens + 1
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                block_ids = new_block_ids[0]

                meta.add_request(request_id=req_id,
                                 token_ids=token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size)

        self._requests_need_load.clear()
        return meta

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        # Keep retrying direct-mode bridge pop until worker reports recving
        # complete for each request.
        if self.is_producer or not self.direct_block_mode:
            return

        for req_id in (connector_output.finished_recving or ()):
            self._requests_need_load.pop(req_id, None)
            self._pending_failed_block_migrations.pop(req_id, None)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """

        self.chunked_prefill.pop(request.request_id, None)
        self._requests_need_load.pop(request.request_id, None)
        self._pending_failed_block_migrations.pop(request.request_id, None)
        if self.is_producer and self.direct_block_mode:
            # Bug 4 fix: only treat this request as delay-free (waiting
            # for decode RELEASE) if its bridge metadata was actually
            # staged.  request_finished fires on two paths:
            #   - natural completion → prefill finished, bridge staged,
            #     decode will migrate KV and send RELEASE → delay-free ✓
            #   - external abort (finish_requests, e.g. client 300s
            #     timeout) → prefill may have been mid-chunk, bridge
            #     never staged → decode never sees this request, no
            #     RELEASE will ever come → must NOT be marked delay-free
            # Before this fix the second case leaked into delay-free
            # and locked KV blocks for request_completion_timeout_s
            # (600s), starving the admission control and triggering
            # engine-wide stalls.
            bridge_staged = (
                self.p2p_nccl_engine is not None
                and self.p2p_nccl_engine.was_bridge_staged(
                    request.request_id))
            if not bridge_staged:
                logger.debug(
                    "[REQUEST_FINISHED] req=%s aborted before bridge "
                    "staged; freeing blocks immediately",
                    request.request_id)
            return (bridge_staged and len(block_ids) > 0), None

        send_type = str(
            self.config.get_from_extra_config("send_type",
                                              "PUT_ASYNC")).upper()
        if self.is_producer and send_type in ("PUT_ASYNC", "GET"):
            return True, None
        return False, None

    # ==============================
    # Static methods
    # ==============================

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
        # Regular expression to match the string hostname and integer port
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            ip = match.group(1)
            port = int(match.group(2))

            return ip, port
        raise ValueError(
            f"Request id {request_id} does not contain hostname and port")

    @staticmethod
    def check_tensors_except_dim(tensor1, tensor2, dim):
        shape1 = tensor1.size()
        shape2 = tensor2.size()

        if len(shape1) != len(shape2) or not all(
                s1 == s2
                for i, (s1, s2) in enumerate(zip(shape1, shape2)) if i != dim):
            raise NotImplementedError(
                "Currently, only symmetric TP is supported. Asymmetric TP, PP,"
                "and others will be supported in future PRs.")
