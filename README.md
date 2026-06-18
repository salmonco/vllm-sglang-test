# LLM Chatbot

**vLLM과 SGLang 추론 엔진을 직접 써보고 비교하는** 학습용 챗봇.
같은 모델(Qwen 2.5 7B AWQ)을 두 엔진에 올려, UI 토글로 정성 비교 + 스크립트로 정량 비교.

## 목적

vLLM·SGLang 두 LLM 추론 엔진을 **실제로 돌려보며 차이를 체감**하는 것이 목표.
같은 모델·같은 OpenAI 호환 API 위에서 엔진만 바꿔 끼워, 기동 방식·성능·응답 품질이
어떻게 다른지 손으로 확인한다. 프로덕션 최적화가 아니라 학습·탐색이 우선.

## 요약

- **둘 다 OpenAI 호환 API** → 백엔드는 `base_url`만 바꿔 엔진 교체 (코드 공유).
- **vast.ai RTX 3090에서 실제 구동·측정 완료** (총 $0.24). 두 엔진 다 띄워 채팅·벤치 성공.
- **이번 측정상 vLLM이 raw throughput 우위**(243 vs 84 tok/s)지만, SGLang에 불리한
  조건(공유 GPU·cudagraph off·단발 요청)이라 **공정 비교는 아님** → [한계](#한계) 참고.
- **SGLang의 진짜 강점**(RadixAttention 접두사 재사용, 구조화 출력)은 멀티턴·few-shot·
  JSON 워크로드에서 드러나며 이번 벤치에선 미측정.
- **runtime 이미지로 pip 구동 시 함정이 많다** (nvcc/gcc/libcuda 부재) → [트러블슈팅](#트러블슈팅-vastai-pytorch-runtime-이미지--pip-구동-기준) 참고. 공식 이미지 쓰면 대부분 회피.

## 실측 결과 (RTX 3090 24GB, Qwen2.5-7B-AWQ)

vast.ai에서 직접 측정 (2026-06). **대여 비용**: $0.206/hr × 약 70분 = **총 $0.24**.

| 지표 | vLLM | SGLang |
|---|---|---|
| TTFT p50 | 45 ms | 122 ms |
| 지연 p50 | 1.70 s | 4.79 s |
| decode tok/s | 65.4 | 22.7 |
| throughput tok/s (동시성 4) | 243 | 83.6 |

정성(동일 프롬프트 4종): 코딩·추론은 양쪽 정답. 한국어 일관성은 이 샘플에서 SGLang이
약간 안정적(vLLM은 한 답변이 중국어로 새는 글리치) — 모델·샘플링 차이지 엔진 성질 아님.

## 한계

위 수치는 참고치이며 엔진의 우열 근거로 쓸 수 없다.

- **측정 조건이 다름**: vLLM 단독 vs SGLang은 GPU 공유 → throughput 직접 비교 불가.
- **둘 다 최적화 끈 상태**: CUDA graph 없이 구동 → 각자의 최대 성능이 아님.
- **워크로드가 SGLang에 불리**: 단발 요청뿐이라 SGLang 핵심인 RadixAttention(접두사
  재사용)이 작동 안 함. 멀티턴·공통 프롬프트·few-shot·JSON 시나리오는 미측정.
- **표본·범위 협소**: 4프롬프트 1회 · GPU 1장 · 7B · AWQ만. 멀티 GPU·MoE 등은 결론 불가.

제대로 비교하려면 각 엔진을 단독·공식 이미지·cudagraph 켜고 워크로드별로 측정해야 한다.

## 스택

- **추론 엔진**: vLLM, SGLang (둘 다 OpenAI 호환 API → 백엔드 코드 공유)
- **모델**: Qwen 2.5 7B Instruct AWQ (4bit)
- **백엔드**: FastAPI (엔진 라우팅 게이트웨이) · **프론트**: React + Vite SPA
- **GPU**: vast.ai, Ampere(sm_80) 이상 24GB 권장 (RTX 3090/4090)

## 구조

```
backend/
  main.py         # FastAPI 게이트웨이 (engine 필드로 vLLM/SGLang 라우팅 + 스트리밍)
  benchmark.py    # 정량 벤치 (TTFT·지연·tok/s·throughput)
  qualitative.py  # 정성 비교 (동일 프롬프트 응답 수집)
frontend/src/App.tsx   # 챗봇 UI (엔진 토글)
docker-compose.yml     # vLLM(8000) + SGLang(8001) + FastAPI(8080), profiles
```

## 실행

Docker Compose **profile**로 무엇을 띄울지 고른다.

```bash
# 정성 비교 — 두 엔진 동시 (16GB는 빠듯, 24GB 권장. 메모리 반씩 분할)
VLLM_GPU_FRAC=0.45 SGLANG_GPU_FRAC=0.45 docker compose --profile both up

# 정량 벤치 — 한 엔진씩 단독 (GPU 경합 없이 공정)
docker compose --profile vllm   up      # 또는 --profile sglang
cd backend && pip install httpx && python benchmark.py --requests 32 --concurrency 8

# 프론트엔드 (로컬)
cd frontend && npm install
VITE_API_URL=http://<instance-ip>:8080 npm run dev   # localhost:5173
```

게이트웨이 상태: `curl http://<ip>:8080/api/health` → 두 엔진 up/down.

## 환경변수

| 변수 | 위치 | 기본값 |
|---|---|---|
| `MODEL_NAME` | backend/엔진 | `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| `VLLM_BASE_URL` / `SGLANG_BASE_URL` | backend | `localhost:8000/v1` / `:8001/v1` |
| `VLLM_GPU_FRAC` / `SGLANG_GPU_FRAC` | compose | `0.9` (동시 실행 시 `0.45`) |
| `VITE_API_URL` | frontend | `http://localhost:8080` |

## 트러블슈팅 (vast.ai `pytorch-runtime` 이미지 + pip 구동 기준)

**공통 원인**: runtime 이미지엔 CUDA 툴킷(nvcc)·빌드툴(gcc)·일부 라이브러리가 없어 두
엔진 모두 런타임 JIT 컴파일에서 막힌다. 공식 이미지(`vllm/vllm-openai`,
`lmsysorg/sglang`)를 쓰면 대부분 회피된다.

| 증상 | 해결 |
|---|---|
| 인스턴스 `loading` 멈춤 / `No such container` | 불량 호스트 — 폐기 후 `reliability>0.98`·고대역폭 호스트로 재대여 |
| `scp` 만 `Permission denied` (ssh는 됨) | scp에도 `-i <key>` 지정 |
| `pip install vastai` PEP 668 거부 | 전용 venv에 설치 |
| 재기동 시 메모리 race로 엔진 죽음 | 잔류 `VLLM::EngineCore` 정리: `pkill -9 -f vllmvenv` |
| **vLLM** `torch.compile` 실패 | `--enforce-eager` |
| **vLLM** flashinfer 샘플러 nvcc 요구 | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| **vLLM** `Failed to find C compiler` | `apt install build-essential` + `export CC=gcc` |
| **SGLang** `libnuma.so.1` 누락 | `apt install libnuma1` |
| **SGLang** flashinfer nvcc/`CUDA_HOME` 요구 | `conda install -c nvidia cuda-toolkit` + `export CUDA_HOME=/opt/conda` |
| **SGLang** `ld: cannot find -lcuda` | `export LIBRARY_PATH=/opt/conda/lib/stubs:$LIBRARY_PATH` |
| **SGLang** 첫 요청 TTFT 수십 초 | 벤치 전 워밍업 2–3회 (커널 JIT) |
| 동시 실행 OOM | 메모리 합 ≲0.95로 분할 (`--gpu-memory-utilization` + `--mem-fraction-static`) |

검증된 기동 명령(runtime 이미지):

```bash
# vLLM
apt install -y build-essential
export CC=gcc VLLM_USE_FLASHINFER_SAMPLER=0
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq --max-model-len 4096 --port 8000 --enforce-eager

# SGLang
apt install -y libnuma1 build-essential && conda install -y -c nvidia cuda-toolkit
export CUDA_HOME=/opt/conda PATH=/opt/conda/bin:$PATH LIBRARY_PATH=/opt/conda/lib/stubs
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq --context-length 4096 --port 8001 --disable-cuda-graph
```

**원격 엔진을 로컬에서 쓰기**: `ssh -i <key> -p <port> -N -L 8080:localhost:8080 root@<host>`
로 터널을 열면 브라우저(localhost:5173) → 터널(8080) → 원격 엔진으로 연결된다.
