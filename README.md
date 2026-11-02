# inference-gateway
A small admission-control gateway for any OpenAI-compatible LLM server (vLLM, llama.cpp server, etc.). It keeps the backend responsive under load: caps concurrency, queues a bounded number of requests, enforces deadlines, and rejects overflow with `429`.

**Phase 3 Update:** Added workload-aware scheduling. Requests are classified as short / medium / long, and the gateway can schedule across separate per-class queues using FIFO, strict priority, or weighted round-robin (WRR). This prevents long requests from starving while improving fairness and per-class latency under mixed workloads.

## Features
- Active-request limit plus a bounded shared waiting queue.
- Request classification by `max_tokens` or `X-Request-Class` header.
- Three scheduling policies:
  - `fifo` — oldest arrival first (baseline).
  - `priority` — shortest class first (can starve long requests).
  - `wrr` — weighted round-robin across classes (starvation-free).
- Queue deadline and upstream timeout (`429` / `504` / `502`).
- `/health` and `/metrics` endpoints with per-class counters.
- Per-request JSONL log for benchmarking.

## Config (env vars)
- `VLLM_URL` (default `http://localhost:8000`)
- `MAX_ACTIVE` (1)
- `MAX_QUEUE` (32)
- `QUEUE_DEADLINE` (60s)
- `UPSTREAM_TIMEOUT` (120s)
- `SCHED` (`fifo`, `priority`, or `wrr`, default `wrr`)
- `WEIGHTS` (default `4,2,1` for short,medium,long)
- `SHORT_MAX` (64), `MEDIUM_MAX` (256) — `max_tokens` thresholds
- `LOG_PATH` (default `results/phase3_requests.jsonl`)

## Run

### Install
```bash
# 1 :-
pip install -r requirements.txt

# 2 : Terminal 1: vLLM at port 8000
vllm serve Qwen/Qwen2.5-0.5B-Instruct --gpu-memory-utilization 0.65 --max-model-len 1024 --max-num-seqs 1


# 3 : Terminal 2: gateway at port 9000
# for FIFO baseline
SCHED=fifo uvicorn gateway3:app --port 9000

# for Priority (shortest class first)
SCHED=priority uvicorn gateway3:app --port 9000

# for Weighted round-robin (default)
SCHED=wrr WEIGHTS=4,2,1 uvicorn gateway3:app --port 9000


# 4 : Terminal 3: benchmark
# FIFO
python bench3.py --policy fifo --rate 1.5 --n 60 --seed 1

# Priority
python bench3.py --policy priority --rate 1.5 --n 60 --seed 1

# WRR
python bench3.py --policy wrr --rate 1.5 --n 60 --seed 1
``

## Results
See results/phase3_summary.csv for FIFO vs priority vs WRR under identical workloads.
