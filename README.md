# LLM Chatbot

vLLM + Qwen 2.5 7B AWQ 기반 챗봇. FastAPI 백엔드와 React SPA 프론트엔드로 구성.

## 기술스택

- **추론 엔진**: vLLM (OpenAI-compatible API)
- **모델**: Qwen 2.5 7B Instruct AWQ (4bit 양자화)
- **백엔드**: FastAPI + uvicorn
- **프론트엔드**: React + TypeScript + Vite (CSR SPA)
- **GPU 대여**: vast.ai (16GB VRAM 이상)

## 프로젝트 구조

```
llm-chatbot/
├── backend/
│   ├── main.py              # FastAPI 서버 (스트리밍 응답)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                # React + Vite SPA
│   └── src/
│       ├── App.tsx          # 챗봇 UI
│       └── App.css
├── docker-compose.yml       # vLLM + FastAPI 컨테이너
└── .gitignore
```

## 실행

### 1. 백엔드 (vast.ai)

GPU 인스턴스에서 docker compose로 실행:

```bash
docker compose up
```

- vLLM이 HuggingFace에서 모델을 자동 다운로드
- vLLM API: `http://<instance-ip>:8000`
- FastAPI: `http://<instance-ip>:8080`

### 2. 프론트엔드 (로컬)

```bash
cd frontend
npm install
VITE_API_URL=http://<instance-ip>:8080 npm run dev
```

`http://localhost:5173`에서 챗봇 사용 가능.

## 환경변수

| 변수 | 위치 | 기본값 | 설명 |
|---|---|---|---|
| `VLLM_BASE_URL` | backend | `http://localhost:8000/v1` | vLLM 서버 주소 |
| `MODEL_NAME` | backend | `Qwen/Qwen2.5-7B-Instruct-AWQ` | 모델 ID |
| `VITE_API_URL` | frontend | `http://localhost:8080` | FastAPI 서버 주소 |

## GPU 요구사항

- 최소 16GB VRAM (RTX 4060 Ti 16GB, RTX 3090, A100 등)
- vast.ai에서 `RTX 4060 Ti 16GB` 급 추천 (가성비)
