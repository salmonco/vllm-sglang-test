"""정성 비교용 — 고정 프롬프트 세트에 대한 엔진의 실제 응답을 받아 그대로 출력.

vLLM/SGLang에 같은 프롬프트를 던져 응답 품질·문체를 사람이 눈으로 비교하기 위함.
benchmark.py(정량)와 짝을 이룬다.

사용:
    python qualitative.py --url http://localhost:8000/v1 --engine vllm
"""

import argparse
import asyncio
import json

import httpx

PROMPTS = [
    ("설명-한국어", "양자 컴퓨팅을 고등학생도 이해할 수 있게 5문장으로 설명해줘."),
    ("코딩", "Write a Python function `nth_fib(n)` that returns the nth Fibonacci "
             "number using memoization. Include a one-line docstring."),
    ("추론", "기차가 45분 동안 60km를 달렸다. 평균 속력은 시속 몇 km인가? "
            "풀이 과정을 단계별로 보여줘."),
    ("창작-한국어", "고양이를 주제로 4행짜리 짧은 시를 써줘."),
]


async def ask(client, base_url, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    try:
        r = await client.post(f"{base_url}/chat/completions", json=body, timeout=180)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(요청 실패: {e})"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="OpenAI 호환 base URL (.../v1)")
    ap.add_argument("--engine", required=True, help="엔진 이름 (라벨용)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    print(f"\n{'#' * 70}\n# 정성 샘플 — {args.engine}  ({args.url})\n{'#' * 70}")
    async with httpx.AsyncClient() as client:
        for tag, prompt in PROMPTS:
            ans = await ask(client, args.url, args.model, prompt, args.max_tokens)
            print(f"\n[{tag}] {prompt}\n{'-' * 60}\n{ans}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
