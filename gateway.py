"""
Minimal SLA-aware gateway in front of vLLM.

Run (single process only, state lives in memory):
    uvicorn gateway:app --port 9000

Needs Python 3.11+ (uses asyncio.timeout).
"""
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

# ---------- Settings (change with environment variables) ----------
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
MAX_ACTIVE = int(os.getenv("MAX_ACTIVE", "8"))            # requests sent to vLLM at once
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "4"))              # requests allowed to wait
QUEUE_DEADLINE = float(os.getenv("QUEUE_DEADLINE", "1"))  # max seconds spent waiting
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "30"))  # max seconds vLLM may take
LOG_PATH = os.getenv("LOG_PATH", "results/requests.jsonl")

# ---------- State ----------
slots = asyncio.Semaphore(MAX_ACTIVE)  # the "active request" permits
waiting = 0                            # how many requests are queued right now
active = 0                             # how many are being served right now
stats = {
    "received": 0,
    "completed": 0,
    "rejected_queue_full": 0,
    "rejected_deadline": 0,
    "upstream_timeout": 0,
    "upstream_error": 0,
}

os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(base_url=VLLM_URL, timeout=UPSTREAM_TIMEOUT)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


def log_event(outcome: str, status: int, queue_wait: float, service_time: float):
    """One JSON line per request. This file becomes your benchmark data later."""
    row = {
        "ts": time.time(),
        "outcome": outcome,
        "status": status,
        "queue_wait_s": round(queue_wait, 4),
        "service_s": round(service_time, 4),
        "waiting_at_exit": waiting,
        "active_at_exit": active,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def reject(outcome: str, t0: float) -> JSONResponse:
    log_event(outcome, 429, time.perf_counter() - t0, 0.0)
    return JSONResponse(
        {"error": "server overloaded", "reason": outcome},
        status_code=429,
        headers={"Retry-After": "1"},
    )


@app.post("/v1/chat/completions")
async def chat(request: Request):
    global waiting, active
    stats["received"] += 1
    t0 = time.perf_counter()

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if body.get("stream"):
        return JSONResponse({"error": "streaming not supported yet"}, status_code=400)

    # 1) Bounded queue: if all slots are busy AND the waiting room is full, say no now.
    if slots.locked() and waiting >= MAX_QUEUE:
        stats["rejected_queue_full"] += 1
        return reject("queue_full", t0)

    # 2) Wait for a free slot, but only up to QUEUE_DEADLINE seconds.
    waiting += 1
    try:
        async with asyncio.timeout(QUEUE_DEADLINE):
            await slots.acquire()
    except TimeoutError:
        stats["rejected_deadline"] += 1
        return reject("queue_deadline", t0)
    finally:
        waiting -= 1

    # 3) We own a slot. Forward to vLLM, and ALWAYS give the slot back.
    queue_wait = time.perf_counter() - t0
    t1 = time.perf_counter()
    active += 1
    try:
        upstream = await request.app.state.client.post("/v1/chat/completions", json=body)
        service_time = time.perf_counter() - t1
        stats["completed"] += 1
        log_event("ok", upstream.status_code, queue_wait, service_time)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type="application/json",
        )
    except httpx.TimeoutException:
        stats["upstream_timeout"] += 1
        log_event("upstream_timeout", 504, queue_wait, time.perf_counter() - t1)
        return JSONResponse({"error": "vLLM took too long"}, status_code=504)
    except httpx.TransportError:
        stats["upstream_error"] += 1
        log_event("upstream_error", 502, queue_wait, time.perf_counter() - t1)
        return JSONResponse({"error": "cannot reach vLLM"}, status_code=502)
    finally:
        active -= 1
        slots.release()


@app.get("/health")
async def health(request: Request):
    try:
        r = await request.app.state.client.get("/health", timeout=2)
        ok = r.status_code == 200
    except httpx.HTTPError:
        ok = False
    return JSONResponse({"gateway": "up", "vllm": "up" if ok else "down"},
                        status_code=200 if ok else 503)


@app.get("/metrics")
async def metrics():
    return {
        **stats,
        "waiting_now": waiting,
        "active_now": active,
        "max_active": MAX_ACTIVE,
        "max_queue": MAX_QUEUE,
    }
