# inference-gateway

A small admission-control gateway for any OpenAI-compatible LLM server (vLLM, llama.cpp server, etc.). It keeps the backend responsive under load: it caps concurrent requests, queues a bounded number, enforces deadlines, and rejects the overflow fast with `429`.

**Phase 2 Update:** Added SLA-aware admission control. The gateway can now predict request completion times based on an EMA of upstream service time and reject requests early if they are predicted to miss their client-provided latency budgets.

## Features
- Active-request limit plus a bounded waiting queue.
- **SLA-aware admission:** Predicts if a request can finish within its `X-Request-Deadline-Ms` budget.
- **Baseline mode:** Original Phase 1 behavior (FIFO queue, no SLA prediction).
- Queue deadline and upstream timeout (`429` / `504` / `502` responses).
- `/health` and `/metrics` endpoints.
- Per-request JSONL log for benchmarking.

## Config (env vars)
- `UPSTREAM_URL` (default `http://localhost:8000`)
- `MAX_ACTIVE` (1)
- `MAX_QUEUE` (4)
- `QUEUE_DEADLINE` (5s)
- `UPSTREAM_TIMEOUT` (30s)
- `ADMISSION` (`sla` or `baseline`, default `sla`)
- `SAFETY` (default `1.0`; >1 = more cautious SLA predictions)
- `DEFAULT_DEADLINE_MS` (10000)

## Run

### Install required packages
```bash
pip install -r requirements.txt

## Terminal 1: vLLM from your baseline at port 8000
vllm serve {model_name} --gpu-memory-utilization 0.7 --max-model-len 1024 --max-num-seqs 1

## Terminal 2: the gateway at port 9000
SLA-aware admission mode:-
ADMISSION=sla uvicorn gateway_sla:app --port 9000

Baseline queue-full mode:-
ADMISSION=baseline uvicorn gateway_sla:app --port 9000

##Terminal 3: the benchmark

### Run SLA benchmark
python bench_sla.py --label sla --rate 4 --deadline-ms 1500 --n 40

### Run Baseline benchmark
python bench_sla.py --label baseline --rate 4 --deadline-ms 1500 --n 40

(Results are appended to results/sla_summary.csv)
