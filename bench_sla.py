import argparse
import asyncio
import csv
import os
import random
import time

import httpx

URL = "http://localhost:9000/v1/chat/completions"
PAYLOAD = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Explain what a queue is in two sentences."}],
    "max_tokens": 64,
    "temperature": 0,
}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


async def one(client, deadline_ms):
    t = time.perf_counter()
    try:
        r = await client.post(URL, json=PAYLOAD,
                              headers={"X-Request-Deadline-Ms": str(deadline_ms)})
        return r.status_code, (time.perf_counter() - t) * 1000
    except httpx.HTTPError:
        return 0, (time.perf_counter() - t) * 1000


async def main(a):
    rng = random.Random(a.seed)
    async with httpx.AsyncClient(timeout=60) as client:
        await one(client, 60000)  # warm-up so the first sample isn't a cold start
        tasks = []
        for _ in range(a.n):
            tasks.append(asyncio.create_task(one(client, a.deadline_ms)))
            await asyncio.sleep(rng.expovariate(a.rate))  # Poisson arrivals
        results = await asyncio.gather(*tasks)

    sent = len(results)
    accepted = [(c, ms) for c, ms in results if c == 200]
    rejected = sum(1 for c, _ in results if c == 429)
    met = [ms for _, ms in accepted if ms <= a.deadline_ms]
    row = {
        "label": a.label, "rate_rps": a.rate, "deadline_ms": a.deadline_ms, "n": sent,
        "accept_rate": round(len(accepted) / sent, 3),
        "reject_rate": round(rejected / sent, 3),
        "deadline_success_of_all": round(len(met) / sent, 3),  # goodput
        "deadline_success_of_accepted": round(len(met) / len(accepted), 3) if accepted else 0,
        "p50_ms": round(pct([ms for _, ms in accepted], 0.5)),
        "p95_ms": round(pct([ms for _, ms in accepted], 0.95)),
    }
    print(row)

    os.makedirs("results", exist_ok=True)
    path = "results/sla_summary.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--rate", type=float, default=2.0, help="requests per second")
    p.add_argument("--deadline-ms", type=int, default=3000)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=1)
    asyncio.run(main(p.parse_args()))
