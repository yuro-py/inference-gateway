"""
SLA-aware gateway in front of vLLM (Phase 2).

Run (single process only, state lives in memory):
    ADMISSION=sla      uvicorn gateway:app --port 9000   # SLA-aware policy
    ADMISSION=baseline uvicorn gateway:app --port 9000   # Phase 1 behaviour

Clients send their total latency budget in a header:
    X-Request-Deadline-Ms: 3000

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
MAX_ACTIVE = int(os.getenv("MAX_ACTIVE", "1"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "4"))
QUEUE_DEADLINE = float(os.getenv("QUEUE_DEADLINE", "5"))       # hard cap on queue wait (s)
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "30"))
LOG_PATH = os.getenv("LOG_PATH", "results/requests.jsonl")

ADMISSION = os.getenv("ADMISSION", "sla")                      # "sla" or "baseline"
DEFAULT_DEADLINE_MS = float(os.getenv("DEFAULT_DEADLINE_MS", "10000"))  # if header missing
SAFETY = float(os.getenv("SAFETY", "1.0"))    # >1 = more cautious, <1 = more optimistic
EMA_ALPHA = 0.2                               # how fast the service-time estimate adapts

# ---------- State ----------
slots = asyncio.Semaphore(MAX_ACTIVE)
waiting = 0
active = 0
ema_service = None   # running average of vLLM service time in seconds (None until first sample)
stats = {
    "received": 0, "completed": 0,
    "rejected_queue_full": 0, "rejected_sla": 0, "rejected_deadline": 0,
    "upstream_timeout": 0, "upstream_error": 0,
    "deadline_met": 0, "deadline_missed": 0,
}

os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(base_url=VLLM_URL, timeout=UPSTREAM_TIMEOUT)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


def log_event(outcome, status, t0, queue_wait, service, deadline_ms, predicted_s=None):
    """One JSON line per request: the raw data for your benchmarks."""
    total = time.perf_counter() - t0
    met = None
    if outcome == "ok" and status == 200:
        met = total * 1000 <= deadline_ms
        stats["deadline_met" if met else "deadline_missed"] += 1
    row = {
        "ts": time.time(), "mode": ADMISSION, "outcome": outcome, "status": status,
        "deadline_ms": deadline_ms,
        "predicted_ms": None if predicted_s is None else round(predicted_s * 1000),
        "queue_wait_s": round(queue_wait, 4), "service_s": round(service, 4),
        "total_s": round(total, 4), "met_deadline": met,
        "waiting_at_exit": waiting, "active_at_exit": active,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def reject(outcome, t0, deadline_ms, predicted_s=None):
    log_event(outcome, 429, t0, time.perf_counter() - t0, 0.0, deadline_ms, predicted_s)
    remaining_ms = deadline_ms - (time.perf_counter() - t0) * 1000
    return JSONResponse(
        {"error": "server overloaded", "reason": outcome,
         "predicted_ms": None if predicted_s is None else round(predicted_s * 1000),
         "remaining_ms": round(remaining_ms)},
        status_code=429, headers={"Retry-After": "1"},
    )


@app.post("/v1/chat/completions")
async def chat(request: Request):
    global waiting, active, ema_service
    stats["received"] += 1
    t0 = time.perf_counter()

    # --- parse body and the client's latency budget ---
    try:
        raw = request.headers.get("X-Request-Deadline-Ms")
        deadline_ms = float(raw) if raw else DEFAULT_DEADLINE_MS
        if deadline_ms <= 0:
            raise ValueError
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON or X-Request-Deadline-Ms"}, status_code=400)
    if body.get("stream"):
        return JSONResponse({"error": "streaming not supported yet"}, status_code=400)

    def remaining() -> float:  # seconds of budget left
        return deadline_ms / 1000 - (time.perf_counter() - t0)

    # --- gate 1: bounded queue (same as Phase 1) ---
    if slots.locked() and waiting >= MAX_QUEUE:
        stats["rejected_queue_full"] += 1
        return reject("queue_full", t0, deadline_ms)

    # --- gate 2: SLA prediction ---
    predicted = None
    wait_limit = QUEUE_DEADLINE
    if ADMISSION == "sla" and ema_service is not None:
        ahead = waiting + active                        # requests that will finish before me
        predicted = (ahead / MAX_ACTIVE + 1) * ema_service   # their time + my own service
        if predicted * SAFETY > remaining():
            stats["rejected_sla"] += 1
            return reject("sla_unmeetable", t0, deadline_ms, predicted)
        # safety net: don't wait so long that there's no time left to be served
        wait_limit = max(0.0, min(QUEUE_DEADLINE, remaining() - ema_service))

    # --- wait for a slot (bounded by wait_limit) ---
    waiting += 1
    try:
        async with asyncio.timeout(wait_limit):
            await slots.acquire()
    except TimeoutError:
        stats["rejected_deadline"] += 1
        return reject("queue_deadline", t0, deadline_ms, predicted)
    finally:
        waiting -= 1

    # --- serve: forward to vLLM, always release the slot ---
    queue_wait = time.perf_counter() - t0
    t1 = time.perf_counter()
    active += 1
    try:
        upstream = await request.app.state.client.post("/v1/chat/completions", json=body)
        service = time.perf_counter() - t1
        if upstream.status_code == 200:   # learn only from successful requests
            ema_service = service if ema_service is None else (
                EMA_ALPHA * service + (1 - EMA_ALPHA) * ema_service)
        stats["completed"] += 1
        log_event("ok", upstream.status_code, t0, queue_wait, service, deadline_ms, predicted)
        return Response(content=upstream.content, status_code=upstream.status_code,
                        media_type="application/json")
    except httpx.TimeoutException:
        stats["upstream_timeout"] += 1
        log_event("upstream_timeout", 504, t0, queue_wait, time.perf_counter() - t1,
                  deadline_ms, predicted)
        return JSONResponse({"error": "vLLM took too long"}, status_code=504)
    except httpx.TransportError:
        stats["upstream_error"] += 1
        log_event("upstream_error", 502, t0, queue_wait, time.perf_counter() - t1,
                  deadline_ms, predicted)
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
        **stats, "mode": ADMISSION,
        "waiting_now": waiting, "active_now": active,
        "ema_service_ms": None if ema_service is None else round(ema_service * 1000),
        "max_active": MAX_ACTIVE, "max_queue": MAX_QUEUE,
    }
