import argparse
import asyncio
import csv
import os
import random
import statistics
import time

import httpx

# class -> (max_tokens, prompt). min_tokens == max_tokens makes cost deterministic.
SPEC = {
    "short": (32, "Say hello in one friendly sentence."),
    "medium": (160, "Write a detailed paragraph about how a library organizes its books."),
    "long": (600, "Write a long, detailed essay about the history of computing."),
}
MIX = [("short", 0.6), ("medium", 0.3), ("long", 0.1)]
CLASSES = list(SPEC)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def jain(xs):
    """Jain's fairness index: 1.0 = perfectly even, 1/n = one class gets everything."""
    xs = [x for x in xs if x is not None]
    denom = len(xs) * sum(x * x for x in xs)
    return (sum(xs) ** 2) / denom if xs and denom else None


def rnd(x, d=0):
    return "" if x is None else round(x, d)


def make_schedule(n, rate, seed):
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)  # Poisson arrivals
        out.append((t, rng.choices([c for c, _ in MIX], [w for _, w in MIX])[0]))
    return out


async def send(client, url, model, cls):
    tokens, prompt = SPEC[cls]
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens, "min_tokens": tokens, "temperature": 0}
    t = time.perf_counter()
    try:
        r = await client.post(url, json=body)
        return r.status_code, (time.perf_counter() - t) * 1000
    except httpx.HTTPError:
        return 0, (time.perf_counter() - t) * 1000


async def main(a):
    url = f"{a.url}/v1/chat/completions"
    schedule = make_schedule(a.n, a.rate, a.seed)

    async with httpx.AsyncClient(timeout=300) as client:
        await send(client, url, a.model, "short")  # warm-up

        # Calibration on an idle system: the "no waiting" time of each class.
        base = {}
        for c in CLASSES:
            xs = [ms for code, ms in [await send(client, url, a.model, c) for _ in range(a.calib)]
                  if code == 200]
            if not xs:
                raise SystemExit(f"calibration failed for class {c}; is the gateway/vLLM up?")
            base[c] = statistics.median(xs)
        capacity = 1 / sum(w * base[c] / 1000 for c, w in MIX)
        print("idle service time (ms):", {c: round(v) for c, v in base.items()})
        print(f"capacity ~ {capacity:.2f} req/s; offered {a.rate} req/s "
              f"(load {a.rate / capacity:.2f}x)")

        start = time.perf_counter()

        async def fire(at, cls):
            await asyncio.sleep(max(0.0, at - (time.perf_counter() - start)))
            code, ms = await send(client, url, a.model, cls)
            return cls, code, ms

        results = await asyncio.gather(*[fire(at, c) for at, c in schedule])
        wall = time.perf_counter() - start

    rows, slow, succ = [], {}, {}
    total_ok = total_tokens = 0
    for c in CLASSES:
        rs = [(code, ms) for cl, code, ms in results if cl == c]
        ok = [ms for code, ms in rs if code == 200]
        rej = sum(1 for code, _ in rs if code == 429)
        slow[c] = statistics.mean(ok) / base[c] if ok else None  # 1.0 = no waiting at all
        succ[c] = len(ok) / len(rs) if rs else None
        total_ok += len(ok)
        total_tokens += len(ok) * SPEC[c][0]
        rows.append({"policy": a.policy, "rate": a.rate, "seed": a.seed, "n": a.n, "class": c,
                     "sent": len(rs), "ok": len(ok), "rejected": rej,
                     "p50_ms": rnd(pct(ok, 0.5)), "p95_ms": rnd(pct(ok, 0.95)),
                     "mean_slowdown": rnd(slow[c], 2)})
    rows.append({"policy": a.policy, "rate": a.rate, "seed": a.seed, "n": a.n, "class": "ALL",
                 "sent": a.n, "ok": total_ok, "rejected": sum(r["rejected"] for r in rows),
                 "throughput_rps": rnd(total_ok / wall, 3), "throughput_tps": rnd(total_tokens / wall, 1),
                 "jain_slowdown": rnd(jain(list(slow.values())), 3),
                 "jain_success": rnd(jain(list(succ.values())), 3)})

    print(f"\npolicy={a.policy}")
    print(f"{'class':8}{'sent':>6}{'ok':>5}{'rej':>5}{'p50 ms':>9}{'p95 ms':>9}{'slowdown':>10}")
    for r in rows[:-1]:
        print(f"{r['class']:8}{r['sent']:>6}{r['ok']:>5}{r['rejected']:>5}"
              f"{r['p50_ms']:>9}{r['p95_ms']:>9}{r['mean_slowdown']:>10}")
    al = rows[-1]
    print(f"throughput {al['throughput_rps']} req/s, {al['throughput_tps']} tok/s | "
          f"Jain(slowdown) {al['jain_slowdown']} | Jain(success) {al['jain_success']}")

    os.makedirs("results", exist_ok=True)
    path = "results/phase3_summary.csv"
    fields = ["policy", "rate", "seed", "n", "class", "sent", "ok", "rejected", "p50_ms", "p95_ms",
              "mean_slowdown", "throughput_rps", "throughput_tps", "jain_slowdown", "jain_success"]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="tag: fifo | priority | wrr")
    p.add_argument("--url", default="http://localhost:9000")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--rate", type=float, default=1.0, help="requests per second (offered load)")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--calib", type=int, default=2, help="idle calibration runs per class")
    asyncio.run(main(p.parse_args()))
