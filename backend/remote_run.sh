#!/usr/bin/env bash
# vast.ai 인스턴스에서 vLLM·SGLang을 순차로 띄워 (벤치마크 + 정성샘플) 수행.
# 두 엔진을 각자 venv로 격리 — 한쪽 실패해도 다른쪽 결과는 보존.
set -uo pipefail

MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
WORK=/workspace
OUT=$WORK/out
export HF_HOME=$WORK/hf
mkdir -p "$OUT" "$HF_HOME"
cd "$WORK"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

# 클라이언트(벤치/정성) 공용 venv — httpx만 필요
log "client venv 준비"
python -m venv "$WORK/clientvenv"
"$WORK/clientvenv"/bin/pip install -q --upgrade pip httpx >/dev/null 2>&1
CLIENT="$WORK/clientvenv/bin/python"

wait_health(){  # $1=port $2=server_pid $3=name
  local port=$1 pid=$2 name=$3 i
  for i in $(seq 1 150); do   # 최대 25분 (설치 후 weight 로드 포함)
    sleep 10
    if curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1; then
      log "$name 서버 READY (port $port, $((i*10))s)"; return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      log "$name 서버 프로세스 종료됨 — 로그 tail:"; tail -40 "$OUT/${name}_server.log"; return 1
    fi
    [ $((i % 6)) -eq 0 ] && log "$name 기동 대기 중... ($((i*10))s)"
  done
  log "$name READY 타임아웃"; tail -40 "$OUT/${name}_server.log"; return 1
}

measure(){  # $1=name $2=port
  local name=$1 port=$2
  log "$name 벤치마크 시작"
  "$CLIENT" benchmark.py --engines "$name" --url "$name=http://localhost:$port/v1" \
    --requests 16 --concurrency 4 --max-tokens 256 2>&1 | tee "$OUT/${name}_bench.txt"
  log "$name 정성 샘플 시작"
  "$CLIENT" qualitative.py --url "http://localhost:$port/v1" --engine "$name" 2>&1 \
    | tee "$OUT/${name}_qual.txt"
}

############################ vLLM ############################
run_vllm(){
  log "=== vLLM 설치 (venv) ==="
  python -m venv "$WORK/vllmvenv"
  "$WORK/vllmvenv"/bin/pip install -q --upgrade pip >/dev/null 2>&1
  if ! "$WORK/vllmvenv"/bin/pip install -q vllm 2>"$OUT/vllm_install.log"; then
    log "vLLM 설치 실패"; tail -20 "$OUT/vllm_install.log"; return 1
  fi
  log "=== vLLM 서버 기동 ==="
  "$WORK/vllmvenv"/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --quantization awq --max-model-len 4096 \
    --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.9 \
    > "$OUT/vllm_server.log" 2>&1 &
  local pid=$!
  if wait_health 8000 "$pid" vllm; then measure vllm 8000; fi
  log "vLLM 종료"; kill "$pid" 2>/dev/null; sleep 8; kill -9 "$pid" 2>/dev/null || true
  # GPU 메모리 정리 대기
  sleep 10
}

########################### SGLang ###########################
run_sglang(){
  log "=== SGLang 설치 (venv) ==="
  python -m venv "$WORK/sglvenv"
  "$WORK/sglvenv"/bin/pip install -q --upgrade pip >/dev/null 2>&1
  if ! "$WORK/sglvenv"/bin/pip install -q "sglang[all]" 2>"$OUT/sglang_install.log"; then
    log "sglang[all] 설치 경고 — flashinfer 보강 시도"; tail -10 "$OUT/sglang_install.log"
  fi
  # flashinfer 별도 보강 (CUDA 12.4 / torch2.5 휠)
  "$WORK/sglvenv"/bin/pip install -q flashinfer-python \
    -i https://flashinfer.ai/whl/cu124/torch2.5/ >>"$OUT/sglang_install.log" 2>&1 || true
  log "=== SGLang 서버 기동 ==="
  "$WORK/sglvenv"/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --quantization awq --context-length 4096 \
    --host 0.0.0.0 --port 8001 --mem-fraction-static 0.9 \
    > "$OUT/sglang_server.log" 2>&1 &
  local pid=$!
  if wait_health 8001 "$pid" sglang; then measure sglang 8001; fi
  log "SGLang 종료"; kill "$pid" 2>/dev/null; sleep 8; kill -9 "$pid" 2>/dev/null || true
}

run_vllm   || log "vLLM 단계 실패(계속 진행)"
run_sglang || log "SGLang 단계 실패(계속 진행)"

############################ 비교 요약 ############################
log "=== 비교 요약 ==="
{
  echo "================== 정량 비교 (벤치마크) =================="
  for e in vllm sglang; do
    echo "----- $e -----"; cat "$OUT/${e}_bench.txt" 2>/dev/null || echo "(결과 없음)"
  done
  echo
  echo "================== 정성 비교 (샘플 응답) =================="
  for e in vllm sglang; do
    cat "$OUT/${e}_qual.txt" 2>/dev/null || echo "($e 결과 없음)"
  done
} | tee "$OUT/SUMMARY.txt"

log "ALL DONE — 결과: $OUT/SUMMARY.txt"
echo "REMOTE_RUN_COMPLETE"
