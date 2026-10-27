"""
Fire N requests at the same moment and summarise what happened.

    python load_test.py            # 10 clients
    python load_test.py 20         # 20 clients
"""
import asyncio
import statistics
import sys
import time
from collections import Counter

import httpx

URL = "http://localhost:9000/v1/chat/completions"
PAYLOAD = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64,
}


async def one(client: httpx.AsyncClient):
    t = time.perf_counter()
    try:
        r = await client.post(URL, json=PAYLOAD)
        return r.status_code, time.perf_counter() - t
    except httpx.HTTPError:
        return "client_error", time.perf_counter() - t


async def main(n: int):
    async with httpx.AsyncClient(timeout=60) as client:
        results = await asyncio.gather(*[one(client) for _ in range(n)])

    codes = Counter(code for code, _ in results)
    print(f"Sent {n} simultaneous requests -> status counts: {dict(codes)}")

    ok = sorted(t for code, t in results if code == 200)
    if ok:
        p95 = ok[min(len(ok) - 1, int(0.95 * len(ok)))]
        print(f"200s: median {statistics.median(ok):.2f}s, p95 {p95:.2f}s, max {ok[-1]:.2f}s")
    fast_429 = [t for code, t in results if code == 429]
    if fast_429:
        print(f"429s: slowest rejection took {max(fast_429):.2f}s (should be near-instant for queue_full)")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
