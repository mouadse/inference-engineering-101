#!/usr/bin/env python3
"""Benchmark toks/sec against an OpenAI-compatible chat endpoint (stdlib only).

Measures client-observed TTFT, TPOT, and end-to-end toks/sec over streaming
SSE responses. Warmup runs are excluded from stats; reports mean/median/min/max.

    uv run bench_toks.py --url https://<app>--serve.modal.run
    uv run bench_toks.py --url <url> --mode vision --runs 5 --max-tokens 128
"""

import argparse
import json
import statistics
import time
import urllib.request

DEFAULT_IMAGE = (
    "https://cdn.britannica.com/61/93061-050-99147DCE/"
    "Statue-of-Liberty-Island-New-York-Bay.jpg"
)


def build_messages(mode, prompt, image_url):
    if mode == "vision":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
    return [{"role": "user", "content": prompt}]


def stream_request(url, payload, timeout):
    """One streaming chat request. Returns dict of timing/token metrics."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t_start = time.perf_counter()
    t_first = None
    gen_tokens = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read(2000)!r}")
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            chunk = line[len("data: ") :]
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                if (choice.get("delta") or {}).get("content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    gen_tokens += 1
    t_end = time.perf_counter()
    total = t_end - t_start
    ttft = (t_first - t_start) if t_first else total
    decode_time = t_end - (t_first or t_start)
    # Prefer server-reported completion tokens; fall back to streamed count.
    comp = usage.get("completion_tokens", gen_tokens) or gen_tokens
    tpot = decode_time / comp if comp else 0.0
    return {
        "tokens": comp,
        "total_s": total,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "tps": comp / total if total > 0 else 0.0,
    }


def summarize(name, samples):
    def col(key):
        return [s[key] for s in samples]

    print(f"\n{name}: n={len(samples)} (warmups excluded)")
    print(f"  {'metric':<10} {'mean':>10} {'median':>10} {'min':>10} {'max':>10}")
    for key, unit in (("tps", "tok/s"), ("ttft_s", "s"), ("tpot_s", "s"), ("total_s", "s")):
        vals = col(key)
        print(
            f"  {key:<10} {statistics.mean(vals):>10.2f} "
            f"{statistics.median(vals):>10.2f} {min(vals):>10.2f} {max(vals):>10.2f}  [{unit}]"
        )
    agg_tps = sum(col("tokens")) / sum(col("total_s"))
    print(f"  aggregate throughput: {agg_tps:.2f} tok/s")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="Endpoint base URL (no trailing path)")
    p.add_argument("--model", default="google/medgemma-27b-it")
    p.add_argument("--mode", choices=["text", "vision"], default="vision")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompt", default="Describe this image in one sentence.")
    p.add_argument("--text-prompt",
                   default="Explain how a GPU executes a neural network inference workload in detail.")
    p.add_argument("--image-url", default=DEFAULT_IMAGE)
    p.add_argument("--timeout", type=int, default=600)
    a = p.parse_args()

    prompt = a.prompt if a.mode == "vision" else a.text_prompt
    payload = {
        "model": a.model,
        "stream": True,
        "max_tokens": a.max_tokens,
        "temperature": 0,
        "messages": build_messages(a.mode, prompt, a.image_url),
    }
    print(f"Endpoint: {a.url}/v1/chat/completions  mode={a.mode} "
          f"max_tokens={a.max_tokens} runs={a.runs} warmups={a.warmups}")

    for i in range(a.warmups):
        r = stream_request(a.url, payload, a.timeout)
        print(f"warmup {i + 1}/{a.warmups}: {r['tokens']} tok in "
              f"{r['total_s']:.2f}s ({r['tps']:.2f} tok/s, excluded)")

    samples = []
    for i in range(a.runs):
        r = stream_request(a.url, payload, a.timeout)
        samples.append(r)
        print(f"run {i + 1}/{a.runs}: {r['tokens']} tok in {r['total_s']:.2f}s, "
              f"TTFT {r['ttft_s']:.2f}s, TPOT {r['tpot_s'] * 1000:.1f}ms, {r['tps']:.2f} tok/s")
    summarize(a.mode, samples)


if __name__ == "__main__":
    main()
