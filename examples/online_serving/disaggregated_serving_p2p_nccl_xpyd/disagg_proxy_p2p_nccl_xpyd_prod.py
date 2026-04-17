# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import socket
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, Response, jsonify, make_response, request


DEFAULT_PING_SECONDS = 5
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class NodeInfo:
    zmq_address: str
    expire_at: float


class NodeRegistry:

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._instances: "OrderedDict[str, NodeInfo]" = OrderedDict()
        self._cv = threading.Condition()

    def upsert(self, http_address: str, zmq_address: str) -> bool:
        is_new = False
        with self._cv:
            if http_address not in self._instances:
                is_new = True
            self._instances[http_address] = NodeInfo(
                zmq_address=zmq_address,
                expire_at=time.time() + self._ttl_seconds,
            )
            self._instances.move_to_end(http_address)
            self._remove_expired_locked()
            self._cv.notify_all()
        return is_new

    def snapshot(self) -> list[tuple[str, NodeInfo]]:
        with self._cv:
            self._remove_expired_locked()
            return list(self._instances.items())

    def wait_until_ready(self, timeout_seconds: int = 0) -> bool:
        deadline = None if timeout_seconds <= 0 else (time.time() + timeout_seconds)
        with self._cv:
            while True:
                self._remove_expired_locked()
                if self._instances:
                    return True
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return False
                else:
                    remaining = None
                self._cv.wait(timeout=remaining)

    def _remove_expired_locked(self) -> None:
        now = time.time()
        while self._instances:
            first_key = next(iter(self._instances))
            value = self._instances[first_key]
            if value.expire_at > now:
                break
            print(
                "Remove stale node "
                f"[HTTP:{first_key}, ZMQ:{value.zmq_address}, expire_at:{value.expire_at}]"
            )
            self._instances.pop(first_key, None)


class DisaggProxy:

    def __init__(self, ping_seconds: int) -> None:
        self.prefill_nodes = NodeRegistry(ttl_seconds=ping_seconds)
        self.decode_nodes = NodeRegistry(ttl_seconds=ping_seconds)
        self._rr_counter = 0
        self._rr_lock = threading.Lock()

    def next_pair(self) -> tuple[str, str, str, str, int]:
        prefill_list = self.prefill_nodes.snapshot()
        decode_list = self.decode_nodes.snapshot()

        if not prefill_list:
            raise RuntimeError("No active prefill nodes")
        if not decode_list:
            raise RuntimeError("No active decode nodes")

        with self._rr_lock:
            token = self._rr_counter
            self._rr_counter += 1

        p_idx = token % len(prefill_list)
        d_idx = token % len(decode_list)
        p_http, p_info = prefill_list[p_idx]
        d_http, d_info = decode_list[d_idx]
        return p_http, p_info.zmq_address, d_http, d_info.zmq_address, token


def random_uuid() -> str:
    return uuid.uuid4().hex


async def forward_request(url: str, data: dict[str, Any], request_id: str):
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        "X-Request-Id": request_id,
    }
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status != 200:
                content = await response.text()
                raise RuntimeError(
                    f"Upstream request failed, status={response.status}, body={content}"
                )
            async for chunk_bytes in response.content.iter_chunked(1024):
                yield chunk_bytes


def listen_for_register(proxy: DisaggProxy, hostname: str, port: int) -> None:
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    print(f"Discovery listener started at tcp://{hostname}:{port}")
    while True:
        socks = dict(poller.poll())
        if router_socket not in socks:
            continue

        remote_address, message = router_socket.recv_multipart()
        try:
            data = msgpack.loads(message)
            role = data.get("type")
            http_address = data.get("http_address")
            zmq_address = data.get("zmq_address")
            if not role or not http_address or not zmq_address:
                raise ValueError("missing required register fields")

            if role == "P":
                is_new = proxy.prefill_nodes.upsert(http_address, zmq_address)
            elif role == "D":
                is_new = proxy.decode_nodes.upsert(http_address, zmq_address)
            else:
                raise ValueError(f"unknown role type={role}")

            if is_new:
                print(
                    "Add node "
                    f"[role={role}, HTTP:{http_address}, ZMQ:{zmq_address}, "
                    f"from={remote_address!r}]"
                )
        except Exception as exc:
            print(f"Invalid register packet from {remote_address!r}: {exc}")


def create_app(proxy: DisaggProxy) -> Quart:
    app = Quart(__name__)

    @app.get("/health")
    async def health() -> Response:
        return jsonify(
            {
                "prefill_active": len(proxy.prefill_nodes.snapshot()),
                "decode_active": len(proxy.decode_nodes.snapshot()),
            }
        )

    @app.route("/v1/completions", methods=["POST"])
    @app.route("/v1/chat/completions", methods=["POST"])
    async def handle_request():
        try:
            proxy_req_recv_ts = time.time()
            original_request_data = await request.get_json()
            if not isinstance(original_request_data, dict):
                return jsonify({"error": "Request body must be JSON object"}), 400

            prefill_request = dict(original_request_data)
            prefill_request["max_tokens"] = 1
            if "max_completion_tokens" in prefill_request:
                prefill_request["max_completion_tokens"] = 1

            p_http, p_zmq, d_http, d_zmq, token = proxy.next_pair()
            request_id = random_uuid()

            def _inject_proxy_xargs(payload: dict[str, Any]) -> None:
                xargs = dict(payload.get("vllm_xargs") or {})
                xargs["proxy_request_id"] = request_id
                payload["vllm_xargs"] = xargs

            _inject_proxy_xargs(prefill_request)

            kv_transfer_params: dict[str, Any] = dict(
                original_request_data.get("kv_transfer_params") or {})
            kv_transfer_params.update({
                "kv_transfer_mode": "BLOCK_MIGRATE",
                "prefill_zmq_address": p_zmq,
                "decode_zmq_address": d_zmq,
            })
            prefill_request["kv_transfer_params"] = kv_transfer_params
            decode_request = dict(original_request_data)
            _inject_proxy_xargs(decode_request)
            decode_request["kv_transfer_params"] = kv_transfer_params

            print(
                f"Request token={token}, route [HTTP:{p_http}, ZMQ:{p_zmq}] -> "
                f"[HTTP:{d_http}, ZMQ:{d_zmq}]"
            )

            prefill_forward_start_ts = time.time()
            async for _ in forward_request(
                f"http://{p_http}{request.path}", prefill_request, request_id
            ):
                continue
            prefill_forward_done_ts = time.time()

            decode_forward_start_ts = time.time()
            generator = forward_request(
                f"http://{d_http}{request.path}", decode_request, request_id
            )
            response = await make_response(generator)
            response.timeout = None
            response.headers["X-Disagg-Prefill-HTTP"] = p_http
            response.headers["X-Disagg-Decode-HTTP"] = d_http
            response.headers["X-Disagg-Request-Id"] = request_id
            response.headers["X-Disagg-Route-Token"] = str(token)
            response.headers["X-Disagg-Proxy-Req-Recv-Ts"] = (
                f"{proxy_req_recv_ts:.9f}")
            response.headers["X-Disagg-Proxy-Prefill-Start-Ts"] = (
                f"{prefill_forward_start_ts:.9f}")
            response.headers["X-Disagg-Proxy-Prefill-Done-Ts"] = (
                f"{prefill_forward_done_ts:.9f}")
            response.headers["X-Disagg-Proxy-Decode-Start-Ts"] = (
                f"{decode_forward_start_ts:.9f}")
            return response
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception as exc:
            return jsonify({"error": f"Internal proxy error: {exc}"}), 500

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production-ready disaggregated proxy for P2P NCCL XpYd"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--discovery-host", default="0.0.0.0")
    parser.add_argument("--discovery-port", type=int, default=30001)
    parser.add_argument("--api-port", type=int, default=10001)
    parser.add_argument("--ping-seconds", type=int, default=DEFAULT_PING_SECONDS)
    parser.add_argument("--wait-prefill-seconds", type=int, default=60)
    parser.add_argument("--wait-decode-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.discovery_port == 0 or args.api_port == 0:
        raise ValueError("Ports cannot be 0")

    if not args.discovery_host:
        args.discovery_host = socket.gethostname()

    proxy = DisaggProxy(ping_seconds=args.ping_seconds)

    listener_thread = threading.Thread(
        target=listen_for_register,
        args=(proxy, args.discovery_host, args.discovery_port),
        daemon=True,
    )
    listener_thread.start()

    app = create_app(proxy)

    prefill_ready = proxy.prefill_nodes.wait_until_ready(args.wait_prefill_seconds)
    decode_ready = proxy.decode_nodes.wait_until_ready(args.wait_decode_seconds)
    if not prefill_ready or not decode_ready:
        print(
            "Proxy started before all roles became ready. "
            "It will still accept traffic once nodes register."
        )

    app.run(host=args.host, port=args.api_port)


if __name__ == "__main__":
    main()
