#!/usr/bin/env python3
"""Concurrent text/vision latency benchmark for the MedGemma Modal endpoint."""

import argparse
import asyncio
import base64
import json
import math
import re
import statistics
import time
from difflib import SequenceMatcher
from pathlib import Path

import aiohttp

MODEL_ID = "google/medgemma-1.5-4b-it"
DEFAULT_URL = "https://mouadse--medgemma-1-5-4b-vllm-server.us-east.modal.direct"


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("ttft_ms", "tpot_ms", "total_ms"):
        values = [sample[key] for sample in samples]
        result[f"{key}_p50"] = percentile(values, 0.50)
        result[f"{key}_p90"] = percentile(values, 0.90)
        result[f"{key}_p99"] = percentile(values, 0.99)
    total_tokens = sum(sample["tokens"] for sample in samples)
    wall_s = (max(sample["ended_at"] for sample in samples) - min(sample["started_at"] for sample in samples))
    result["output_tok_s"] = total_tokens / wall_s
    result["requests"] = len(samples)
    result["completion_tokens"] = total_tokens
    return result


def normalize_answer(answer: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", answer.lower()))


def quality_similarity(answer: str, reference: str) -> float:
    return SequenceMatcher(None, normalize_answer(reference), normalize_answer(answer)).ratio()


def vision_data_url() -> str:
    image = base64.b64encode(Path(__file__).with_name("xray.jpg").read_bytes()).decode()
    return f"data:image/jpeg;base64,{image}"


def workload_messages(mode: str, request_index: int, image_url: str) -> list[dict]:
    system = (
        "You are a careful medical assistant. Give precise, clinically useful information, "
        "state uncertainty, and do not omit important observations."
    )
    if mode == "vision":
        content = [
            {
                "type": "text",
                "text": (
                    "Describe this chest image systematically, including modality, image quality, "
                    "major findings, and a concise impression. Continue through the response limit. "
                    f"Request identifier: {request_index}."
                ),
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        content = (
            "Explain the clinical evaluation of acute shortness of breath, including immediate "
            "assessment, important differential diagnoses, tests, and red flags. Continue through "
            f"the response limit. Request identifier: {request_index}."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


async def wait_until_ready(session: aiohttp.ClientSession, url: str, timeout_s: float) -> float:
    started = time.perf_counter()
    deadline = started + timeout_s
    while True:
        try:
            async with session.get(f"{url}/v1/models") as response:
                if response.status == 200:
                    return time.perf_counter() - started
                await response.read()
        except aiohttp.ClientError:
            pass
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"Endpoint was not ready after {timeout_s:.0f}s")
        await asyncio.sleep(2)


async def completion(session: aiohttp.ClientSession, url: str, messages: list[dict], max_tokens: int) -> str:
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
    }
    async with session.post(f"{url}/v1/chat/completions", json=payload) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"Quality request failed with HTTP {response.status}: {body[:1000]}")
    parsed = json.loads(body)
    answer = parsed["choices"][0]["message"]["content"]
    if not answer.strip() or not isinstance(parsed.get("usage", {}).get("completion_tokens"), int):
        raise RuntimeError("Quality response was empty or omitted numeric token usage")
    return answer


async def run_quality_checks(
    session: aiohttp.ClientSession,
    url: str,
    image_url: str,
    reference_path: Path | None,
) -> dict:
    cases = {
        "text": [
            {
                "role": "user",
                "content": (
                    "State the commonly cited normal resting heart-rate range for a healthy adult. "
                    "Include the exact range in beats per minute."
                ),
            }
        ],
        "vision": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify the imaging modality and body region shown. Be explicit.",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    answers = {
        name: await completion(session, url, messages, 256)
        for name, messages in cases.items()
    }
    normalized_text = normalize_answer(answers["text"])
    normalized_vision = normalize_answer(answers["vision"])
    facts_pass = bool(re.search(r"60\D+100", normalized_text)) and (
        ("x ray" in normalized_vision or "radiograph" in normalized_vision)
        and ("chest" in normalized_vision or "thorax" in normalized_vision)
    )
    similarities: dict[str, float] = {}
    if reference_path:
        references = json.loads(reference_path.read_text())
        similarities = {
            name: quality_similarity(answer, references[name])
            for name, answer in answers.items()
        }
    comparison_pass = not similarities or min(similarities.values()) >= 0.35
    return {
        "passed": facts_pass and comparison_pass,
        "facts_passed": facts_pass,
        "comparison_passed": comparison_pass,
        "similarity": similarities,
        "answers": answers,
    }


async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: bytes,
) -> dict[str, float]:
    started = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens = 0
    async with session.post(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    ) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(f"Load request failed with HTTP {response.status}: {body[:1000]}")
        async for raw_line in response.content:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
            if any((choice.get("delta") or {}).get("content") for choice in chunk.get("choices", [])):
                first_token_at = first_token_at or time.perf_counter()
    ended = time.perf_counter()
    if first_token_at is None or completion_tokens <= 0:
        raise RuntimeError("Streaming response had no content or numeric completion tokens")
    return {
        "tokens": completion_tokens,
        "ttft_ms": (first_token_at - started) * 1000,
        "tpot_ms": (ended - first_token_at) * 1000 / max(completion_tokens - 1, 1),
        "total_ms": (ended - started) * 1000,
        "started_at": started,
        "ended_at": ended,
    }


async def run_cell(
    session: aiohttp.ClientSession,
    url: str,
    mode: str,
    concurrency: int,
    requests: int,
    max_tokens: int,
    image_url: str,
) -> list[dict[str, float]]:
    warmup_count = max(2, concurrency)
    payloads = [
        json.dumps(
            {
                "model": MODEL_ID,
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": 0,
                "messages": workload_messages(mode, index, image_url),
            },
            separators=(",", ":"),
        ).encode()
        for index in range(warmup_count + requests)
    ]
    for payload in payloads[:warmup_count]:
        await stream_request(session, url, payload)

    semaphore = asyncio.Semaphore(concurrency)

    async def limited(payload: bytes) -> dict[str, float]:
        async with semaphore:
            return await stream_request(session, url, payload)

    return await asyncio.gather(*(limited(payload) for payload in payloads[warmup_count:]))


async def benchmark(args: argparse.Namespace) -> dict:
    url = args.url.rstrip("/")
    modes = args.modes.split(",")
    concurrencies = [int(value) for value in args.concurrency.split(",")]
    image_url = vision_data_url()
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(concurrencies) + 4)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        cold_start_s = await wait_until_ready(session, url, args.startup_timeout)
        reference_path = Path(args.quality_reference) if args.quality_reference else None
        quality = await run_quality_checks(session, url, image_url, reference_path)
        if not quality["passed"]:
            raise RuntimeError(f"Quality guardrail failed: {quality}")

        cells = {}
        for mode in modes:
            for concurrency in concurrencies:
                name = f"{mode}_c{concurrency}"
                print(f"Running {name}: {args.requests} measured requests", flush=True)
                samples = await run_cell(
                    session,
                    url,
                    mode,
                    concurrency,
                    args.requests,
                    args.max_tokens,
                    image_url,
                )
                cells[name] = summarize(samples)
                print(
                    f"  P99 TTFT {cells[name]['ttft_ms_p99']:.1f} ms; "
                    f"P99 total {cells[name]['total_ms_p99']:.1f} ms; "
                    f"{cells[name]['output_tok_s']:.1f} tok/s",
                    flush=True,
                )

    p99_values = [cell["ttft_ms_p99"] for cell in cells.values()]
    primary = math.exp(statistics.fmean(math.log(value) for value in p99_values))
    return {
        "config": {
            "url": url,
            "modes": modes,
            "concurrency": concurrencies,
            "requests_per_cell": args.requests,
            "max_tokens": args.max_tokens,
        },
        "quality": quality,
        "cold_start_s": cold_start_s,
        "cells": cells,
        "summary": {"primary_p99_ttft_geomean_ms": primary},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--modes", default="text,vision")
    parser.add_argument("--concurrency", default="1,4,16")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--quality-reference")
    parser.add_argument("--write-quality-reference")
    parser.add_argument("--output", default="benchmark-results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(benchmark(args))
    if args.write_quality_reference:
        Path(args.write_quality_reference).write_text(
            json.dumps(result["quality"]["answers"], indent=2) + "\n"
        )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"METRIC primary_p99_ttft_geomean_ms="
        f"{result['summary']['primary_p99_ttft_geomean_ms']:.3f}"
    )
    print(f"METRIC cold_start_s={result['cold_start_s']:.3f}")


if __name__ == "__main__":
    main()
