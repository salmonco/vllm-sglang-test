"""vLLM vs SGLang 정량 벤치마크.

각 엔진의 OpenAI 호환 /v1/chat/completions 엔드포인트를 직접 때려서
TTFT(첫 토큰까지 시간), 종단 지연, decode 처리량(토큰/초), 동시성 처리량을 측정한다.

공정한 비교를 위해 한 번에 한 엔진만 단독으로 띄운 상태에서 돌리는 것을 권장:
    docker compose --profile vllm   up   # 측정 → 종료
    docker compose --profile sglang up   # 측정

사용 예:
    python benchmark.py                                  # 두 엔진 기본 주소로
    python benchmark.py --engines vllm                   # vLLM만
    python benchmark.py --concurrency 8 --requests 32    # 동시성 8, 총 32요청
"""

import argparse
import asyncio
import json
import time

import httpx

DEFAULT_URLS = {
    "vllm": "http://localhost:8000/v1",
    "sglang": "http://localhost:8001/v1",
}

PROMPT = "양자 컴퓨팅을 고등학생도 이해할 수 있게 5문장으로 설명해줘."


async def one_request(client, base_url, model, max_tokens):
    """스트리밍 요청 1건. (TTFT, 종단지연, 생성토큰수) 반환. 실패 시 None."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft = None
    completion_tokens = 0
    try:
        async with client.stream(
            "POST", f"{base_url}/chat/completions", json=body, timeout=120
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload.strip() == "[DONE]":
                    break
                chunk = json.loads(payload)
                choices = chunk.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                if chunk.get("usage"):
                    completion_tokens = chunk["usage"].get("completion_tokens", 0)
    except Exception as e:
        print(f"  요청 실패: {e}")
        return None

    latency = time.perf_counter() - start
    if ttft is None:
        ttft = latency
    return ttft, latency, completion_tokens


async def bench_engine(name, base_url, model, requests, concurrency, max_tokens):
    print(f"\n▶ {name} ({base_url}) — {requests}요청 / 동시성 {concurrency}")
    sem = asyncio.Semaphore(concurrency)
    results = []

    async with httpx.AsyncClient() as client:
        async def worker():
            async with sem:
                return await one_request(client, base_url, model, max_tokens)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*[worker() for _ in range(requests)])
        wall = time.perf_counter() - wall_start

    ok = [r for r in results if r]
    if not ok:
        print("  유효한 응답 없음 (엔진이 떠 있는지 확인)")
        return None

    ttfts = sorted(r[0] for r in ok)
    latencies = sorted(r[1] for r in ok)
    total_tokens = sum(r[2] for r in ok)
    # 요청별 decode 속도: 생성토큰 / (종단지연 - TTFT)
    decode_speeds = [
        r[2] / (r[1] - r[0]) for r in ok if r[2] > 0 and (r[1] - r[0]) > 0
    ]

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    stats = {
        "engine": name,
        "ok": len(ok),
        "ttft_p50": pct(ttfts, 0.50),
        "ttft_p95": pct(ttfts, 0.95),
        "latency_p50": pct(latencies, 0.50),
        "latency_p95": pct(latencies, 0.95),
        "decode_tps": sum(decode_speeds) / len(decode_speeds) if decode_speeds else 0,
        "throughput_tps": total_tokens / wall if wall > 0 else 0,
        "wall": wall,
    }
    return stats


def print_table(rows):
    rows = [r for r in rows if r]
    if not rows:
        return
    print("\n" + "=" * 78)
    header = (
        f"{'engine':<8} {'ok':>4} {'TTFT p50':>10} {'TTFT p95':>10} "
        f"{'lat p50':>9} {'decode t/s':>11} {'throughput':>11}"
    )
    print(header)
    print("-" * 78)
    for r in rows:
        print(
            f"{r['engine']:<8} {r['ok']:>4} "
            f"{r['ttft_p50'] * 1000:>9.0f}ms {r['ttft_p95'] * 1000:>9.0f}ms "
            f"{r['latency_p50']:>8.2f}s {r['decode_tps']:>11.1f} "
            f"{r['throughput_tps']:>10.1f}"
        )
    print("=" * 78)
    print("decode t/s = 요청당 평균 생성속도 · throughput = 전체 생성토큰/벽시계(동시성 반영)")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=list(DEFAULT_URLS), help="vllm sglang")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--requests", type=int, default=16, help="총 요청 수")
    ap.add_argument("--concurrency", type=int, default=4, help="동시 요청 수")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--url", action="append", default=[], help="name=URL 로 주소 오버라이드")
    args = ap.parse_args()

    urls = dict(DEFAULT_URLS)
    for ov in args.url:
        name, _, u = ov.partition("=")
        urls[name] = u

    rows = []
    for name in args.engines:
        if name not in urls:
            print(f"알 수 없는 엔진: {name}")
            continue
        rows.append(
            await bench_engine(
                name, urls[name], args.model,
                args.requests, args.concurrency, args.max_tokens,
            )
        )
    print_table(rows)


if __name__ == "__main__":
    asyncio.run(main())
