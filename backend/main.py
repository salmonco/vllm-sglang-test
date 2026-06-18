from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-AWQ")

# 엔진별 OpenAI 호환 클라이언트. vLLM·SGLang 둘 다 동일 API라 base_url만 다름.
ENGINES = {
    "vllm": AsyncOpenAI(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key="not-needed",
    ),
    "sglang": AsyncOpenAI(
        base_url=os.getenv("SGLANG_BASE_URL", "http://localhost:8001/v1"),
        api_key="not-needed",
    ),
}


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []
    engine: str = "vllm"  # "vllm" | "sglang"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    client = ENGINES.get(req.engine)
    if client is None:
        raise HTTPException(status_code=400, detail=f"unknown engine: {req.engine}")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    async def generate():
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            max_tokens=1024,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/api/health")
async def health():
    """각 엔진에 모델 목록 조회를 시도해 가용 여부를 보고한다."""
    status = {}
    for name, client in ENGINES.items():
        try:
            await client.models.list()
            status[name] = "up"
        except Exception:
            status[name] = "down"
    return {"model": MODEL_NAME, "engines": status}
