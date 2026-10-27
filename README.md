# inference-gateway

A small admission-control gateway for any OpenAI-compatible LLM server (vLLM, llama.cpp server, etc.). It keeps the backend responsive under load: it caps concurrent requests, queues a bounded number, enforces deadlines, and rejects the overflow fast with `429`.

## Features
- Active-request limit plus a bounded waiting queue
- Queue deadline and upstream timeout (`429` / `504` / `502` responses)
- `/health` and `/metrics` endpoints
- Per-request JSONL log (outcome, queue wait, service time) for benchmarking

## Config (env vars)
`UPSTREAM_URL` (default `http://localhost:8000`), `MAX_ACTIVE` (1), `MAX_QUEUE` (4), `QUEUE_DEADLINE` (5s), `UPSTREAM_TIMEOUT` (30s)

## Run
### install required packages
pip install -r requirements.txt

### terminal 1: vLLM from your baseline at port 8000
vllm serve Qwen/Qwen2.5-0.5B-Instruct --gpu-memory-utilization 0.65 --max-model-len 1024 --max-num-seqs 1

### terminal 2: the gateway at port 9000
uvicorn gateway:app --port 9000

### terminal 3: the test
python load_test.py 10

## Current limits (v0.1)
Non-streaming only, FIFO scheduling, single worker process, no retries. Tested on a 4 GB laptop GPU on a small model with non-production workload.
