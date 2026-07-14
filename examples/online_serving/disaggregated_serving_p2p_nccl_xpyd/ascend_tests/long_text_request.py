#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import urllib.error
import urllib.request


LONG_TEXT = (
    "In the past decade, many organizations have moved from simple "
    "rule-based automation to systems that combine data pipelines, machine "
    "learning models, human review, and continuous monitoring. This shift "
    "has changed how teams design software. A traditional application might "
    "treat user input as a fixed set of commands, but modern intelligent "
    "systems often need to interpret ambiguous requests, retrieve relevant "
    "context, make probabilistic decisions, and explain their results. This "
    "creates new engineering challenges. Teams must think about data quality, "
    "model behavior, latency, cost, privacy, observability, and failure modes "
    "at the same time. A reliable system usually does not depend on a model "
    "alone. It uses clear boundaries between components. Retrieval systems "
    "provide source material. Business logic enforces hard rules. Models "
    "handle language understanding, summarization, classification, or "
    "generation. Monitoring tools track unexpected outputs, slow requests, "
    "and changes in user behavior. Human operators review sensitive decisions "
    "and provide feedback when the system is uncertain. Over time, the "
    "feedback can improve prompts, evaluation datasets, routing rules, and "
    "product design. The most successful deployments tend to start with a "
    "narrow workflow. Instead of trying to automate an entire department, a "
    "team may begin with one repeatable task, such as drafting support "
    "replies, extracting fields from documents, or summarizing customer "
    "calls. This makes it easier to measure quality, compare model outputs "
    "with human decisions, and identify where automation is helpful or risky. "
    "As confidence grows, the system can expand to more complex workflows, "
    "but only if reliability and accountability remain visible."
)


def post_json(url: str, payload: dict, timeout_s: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {message}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:10234")
    parser.add_argument(
        "--model",
        default="/root/work/huggingface/models--meta-llama--Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{
            "role":
            "user",
            "content":
            "Summarize the following text in 3 concise bullet points:\n\n"
            + LONG_TEXT,
        }],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    response = post_json(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        payload,
        args.timeout_s,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
