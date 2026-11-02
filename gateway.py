import asyncio
import itertools
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

# ---------- Settings ----------
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
MAX_ACTIVE = int(os.getenv("MAX_ACTIVE", "1"))             # requests inside vLLM at once
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "32"))              # total waiting room (all classes)
QUEUE_DEADLINE = float(os.getenv("QUEUE_DEADLINE", "60"))  # max seconds waiting
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "120"))
LOG_PATH = os.getenv("LOG_PATH", "results/phase3_requests.jsonl")

POLICY = os.getenv("SCHED", "wrr")                         # fifo | priority | wrr
CLASSES = ["short", "medium", "long"]                      # order = priority order
WEIGHTS = dict(zip(CLASSES, map(int, os.getenv("WEIGHTS", "4,2,1").split(","))))
SHORT_MAX = int(os.getenv("SHORT_MAX", "64"))              # max_tokens <= this -> short
MEDIUM_MAX = int(os.getenv("MEDIUM_MAX", "256"))           # max_tokens <= this -> medium
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))
EMA_ALPHA = 0.2

# WRR cycle, interleaved. Weights 4,2,1 -> short medium long short medium short short
CYCLE = [c for r in range(max(WEIGHTS.values())) for c in CLASSES if r < WEIGHTS[c]]


class Scheduler:
    """Per-class waiting lines plus a rule for who gets the next free slot."""

    def __init__(self):
        self.queues = {c: deque() for c in CLASSES}   # items: (arrival_seq, future)
        self.seq = itertools.count()
        self.active = 0
        self.waiting = 0
        self.ptr = 0                                   # WRR position in CYCLE

    def _head(self, c):
        q = self.queues[c]
        while q and q[0][1].done():                    # drop requests that gave up
            q.popleft()
        return q[0] if q else None

    def _pick(self):
        heads = {c: self._head(c) for c in CLASSES}
        ready = [c for c in CLASSES if heads[c]]
        if not ready:
            return None
        if POLICY == "fifo":                           # oldest arrival wins
            c = min(ready, key=lambda k: heads[k][0])
        elif POLICY == "priority":                     # shortest class wins, always
            c = ready[0]
        else:                                          # wrr: next non-empty class in the cycle
            for _ in range(len(CYCLE)):
                c = CYCLE[self.ptr % len(CYCLE)]
                self.ptr += 1
                if heads[c]:
                    break
        return self.queues[c].popleft()

    def _dispatch(self):
        while self.active < MAX_ACTIVE:
            item = self._pick()
            if item is None:
                return
            self.active += 1
            self.waiting -= 1
            item[1].set_result(None)                   # wake the waiting request

    async def acquire(self, cls, wait_limit):
        fut = asyncio.get_running_loop().create_future()
        self.queues[cls].append((next(self.seq), fut))
        self.waiting += 1
        self._dispatch()
        try:
            async with asyncio.timeout(wait_limit):
                await fut
            return True
        except TimeoutError:
            if fut.done() and not fut.cancelled():
                return True                            # granted at the last instant
            fut.cancel()
            self.waiting -= 1
            return False

    def release(self):
        self.active -= 1
        self._dispatch()

    def waiting_by_class(self):
        return {c: sum(1 for _, f in self.queues[c] if not f.done()) for c in CLASSES}


sched = Scheduler()
stats = {"received": 0, "completed": 0, "rejected": 0, "upstream_timeout": 0, "upstream_error": 0}
by_class = {c: {"received": 0, "completed": 0, "rejected": 0, "ema_service_ms": None}
            for c in CLASSES}

os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(base_url=VLLM_URL, timeout=UPSTREAM_TIMEOUT)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


def classify(body, header):
    if header in CLASSES:
        return header
    n = body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS
    return "short" if n <= SHORT_MAX else "medium" if n <= MEDIUM_MAX else "long"


def log_event(cls, outcome, status, t0, queue_wait, service):
    row = {"ts": time.time(), "policy": POLICY, "cls": cls, "outcome": outcome,
           "status": status, "queue_wait_s": round(queue_wait, 4),
           "service_s": round(service, 4), "total_s": round(time.perf_counter() - t0, 4)}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def reject(outcome, cls, t0):
    stats["rejected"] += 1
    by_class[cls]["rejected"] += 1
    log_event(cls, outcome, 429, t0, time.perf_counter() - t0, 0.0)
    return JSONResponse({"error": "server overloaded", "reason": outcome, "class": cls},
                        status_code=429, headers={"Retry-After": "1"})


@app.post("/v1/chat/completions")
async def chat(request: Request):
    stats["received"] += 1
    t0 = time.perf_counter()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if body.get("stream"):
        return JSONResponse({"error": "streaming not supported yet"}, status_code=400)

    cls = classify(body, request.headers.get("X-Request-Class"))
    by_class[cls]["received"] += 1

    # bounded waiting room (shared by all classes, so policies are compared fairly)
    if sched.active >= MAX_ACTIVE and sched.waiting >= MAX_QUEUE:
        return reject("queue_full", cls, t0)

    if not await sched.acquire(cls, QUEUE_DEADLINE):
        return reject("queue_deadline", cls, t0)

    queue_wait = time.perf_counter() - t0
    t1 = time.perf_counter()
    try:
        upstream = await request.app.state.client.post("/v1/chat/completions", json=body)
        service = time.perf_counter() - t1
        if upstream.status_code == 200:
            prev = by_class[cls]["ema_service_ms"]
            ms = service * 1000
            by_class[cls]["ema_service_ms"] = round(ms if prev is None
                                                    else EMA_ALPHA * ms + (1 - EMA_ALPHA) * prev)
        stats["completed"] += 1
        by_class[cls]["completed"] += 1
        log_event(cls, "ok", upstream.status_code, t0, queue_wait, service)
        return Response(content=upstream.content, status_code=upstream.status_code,
                        media_type="application/json", headers={"X-Request-Class": cls})
    except httpx.TimeoutException:
        stats["upstream_timeout"] += 1
        log_event(cls, "upstream_timeout", 504, t0, queue_wait, time.perf_counter() - t1)
        return JSONResponse({"error": "vLLM took too long"}, status_code=504)
    except httpx.TransportError:
        stats["upstream_error"] += 1
        log_event(cls, "upstream_error", 502, t0, queue_wait, time.perf_counter() - t1)
        return JSONResponse({"error": "cannot reach vLLM"}, status_code=502)
    finally:
        sched.release()


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
    return {**stats, "policy": POLICY, "active_now": sched.active, "waiting_now": sched.waiting,
            "waiting_by_class": sched.waiting_by_class(), "by_class": by_class,
            "max_active": MAX_ACTIVE, "max_queue": MAX_QUEUE}
