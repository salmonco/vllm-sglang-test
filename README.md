# LLM Chatbot

**vLLM과 SGLang 추론 엔진을 직접 써보고 비교하는** 학습용 챗봇.
같은 모델(Qwen 2.5 7B AWQ)을 두 엔진에 올려 UI에서 나란히 정성 비교하고,
벤치마크 스크립트로 정량 비교한다. FastAPI 게이트웨이 + React SPA 구성.

## 기술스택

- **추론 엔진**: vLLM, SGLang (둘 다 OpenAI 호환 API → 백엔드 코드 공유)
- **모델**: Qwen 2.5 7B Instruct AWQ (4bit 양자화)
- **백엔드**: FastAPI + uvicorn (엔진 라우팅 게이트웨이)
- **프론트엔드**: React + TypeScript + Vite (CSR SPA, 엔진 토글)
- **GPU 대여**: vast.ai (16GB VRAM 이상)

## 프로젝트 구조

```
llm-chatbot/
├── backend/
│   ├── main.py              # FastAPI 게이트웨이 (엔진별 라우팅 + 스트리밍)
│   ├── benchmark.py         # vLLM vs SGLang 정량 벤치마크
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                # React + Vite SPA
│   └── src/
│       ├── App.tsx          # 챗봇 UI (엔진 토글)
│       └── App.css
├── docker-compose.yml       # vLLM(8000) + SGLang(8001) + FastAPI(8080)
└── .gitignore
```

## 엔진 구성과 포트

| 서비스 | 호스트 포트 | profiles |
|---|---|---|
| vLLM | `8000` | `both`, `vllm` |
| SGLang | `8001` | `both`, `sglang` |
| FastAPI 게이트웨이 | `8080` | 모든 profile |

백엔드는 요청의 `engine` 필드(`"vllm"` / `"sglang"`)에 따라 해당 엔진으로 라우팅한다.

## 실행 (vast.ai GPU 인스턴스)

Docker Compose **profile**로 무엇을 띄울지 고른다.

### A. 정성 비교 — 두 엔진 동시 실행

UI에서 엔진을 토글하며 같은 질문의 응답을 비교한다. 16GB GPU 한 장에 두 엔진을
같이 올리므로 GPU 메모리를 반씩 나눠 준다:

```bash
VLLM_GPU_FRAC=0.45 SGLANG_GPU_FRAC=0.45 docker compose --profile both up
```

- vLLM API: `http://<ip>:8000` · SGLang API: `http://<ip>:8001` · 게이트웨이: `http://<ip>:8080`
- 게이트웨이 상태 확인: `curl http://<ip>:8080/api/health` → 두 엔진 up/down 표시

### B. 정량 비교 — 한 엔진씩 단독 실행 (벤치마크 권장)

두 엔진이 GPU를 나눠 쓰면 처리량 수치가 왜곡되므로, 벤치마크는 한 번에
하나만 전체 메모리로 띄워 측정한다:

```bash
docker compose --profile vllm   up    # 측정 후 Ctrl-C
docker compose --profile sglang up    # 측정
```

벤치마크 실행 (엔진 컨테이너와 같은 호스트에서):

```bash
cd backend && pip install httpx
python benchmark.py --requests 32 --concurrency 8
# 단독 실행 중인 엔진만 측정하려면:  python benchmark.py --engines vllm
```

출력: 엔진별 **TTFT(p50/p95)**, **종단 지연**, **decode 토큰/초**, **동시성 throughput**.

### 프론트엔드 (로컬)

```bash
cd frontend
npm install
VITE_API_URL=http://<instance-ip>:8080 npm run dev
```

`http://localhost:5173`에서 우상단 토글로 vLLM ↔ SGLang을 바꿔가며 사용.

## 환경변수

| 변수 | 위치 | 기본값 | 설명 |
|---|---|---|---|
| `MODEL_NAME` | backend/엔진 | `Qwen/Qwen2.5-7B-Instruct-AWQ` | 모델 ID |
| `VLLM_BASE_URL` | backend | `http://localhost:8000/v1` | vLLM 서버 주소 |
| `SGLANG_BASE_URL` | backend | `http://localhost:8001/v1` | SGLang 서버 주소 |
| `VLLM_GPU_FRAC` | compose | `0.9` | vLLM GPU 메모리 점유율 (동시 실행 시 `0.45`) |
| `SGLANG_GPU_FRAC` | compose | `0.9` | SGLang GPU 메모리 점유율 (동시 실행 시 `0.45`) |
| `VITE_API_URL` | frontend | `http://localhost:8080` | FastAPI 서버 주소 |

## GPU 요구사항

- 최소 16GB VRAM (RTX 4060 Ti 16GB, RTX 3090, A100 등)
- vast.ai에서 `RTX 4060 Ti 16GB` 급 추천 (가성비)
- **두 엔진 동시 실행(`--profile both`)** 시 16GB는 다소 빡빡함. 여유 있게
  비교하려면 24GB(RTX 3090/4090) 이상 권장, 또는 `*_GPU_FRAC`를 낮춰 조정.
