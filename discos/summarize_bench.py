#!/usr/bin/env python3
"""
summarize_bench.py — run_bench_all.sh 결과를 표로 요약

사용:
  python3 summarize_bench.py ~/bench_0823_0634
  python3 summarize_bench.py ~/bench_0823_0634 --detail    # step별 상세도
"""

import csv
import os
import re
import sys
from collections import defaultdict

RE_GET = re.compile(
    r"Retrieved (\d+) out of (\d+) required tokens.*?"
    r"cost ([\d.]+) ms, throughput: ([\d.]+) GB/s"
)
RE_PUT = re.compile(
    r"Stored (\d+) out of total (\d+) tokens.*?"
    r"cost ([\d.]+) ms.*?put_time: ([\d.]+) ms"
)

BACKEND_LABEL = {
    "L_GDSF":  "GDS(false) 로컬NVMe",
    "L_GDST":  "GDS(true)  로컬NVMe",
    "D_DFUSE": "DAOS dfuse",
    "D_GDR":   "DAOS native GDR",
}
# KV 크기 (Qwen2.5-0.5B: 12KB/token)
#KV_MB = {256: 3, 512: 6, 1024: 12}
KV_MB = {256: 40, 512: 80, 1024: 160, 2048: 320}

def parse_log(path):
    gets, puts = [], []
    if not os.path.exists(path):
        return gets, puts
    with open(path, errors="ignore") as f:
        for line in f:
            m = RE_GET.search(line)
            if m:
                gets.append((int(m.group(1)), float(m.group(3)), float(m.group(4))))
                continue
            m = RE_PUT.search(line)
            if m:
                puts.append((int(m.group(1)), float(m.group(3)), float(m.group(4))))
    return gets, puts


def parse_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "step": int(r["step"]),
                    "ptok": int(r["prompt_tokens"]),
                    "cached": int(r["cached_tokens"]),
                })
            except (KeyError, ValueError):
                pass
    return rows


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <bench_dir> [--detail]")
        return 1
    d = os.path.expanduser(sys.argv[1])
    detail = "--detail" in sys.argv

    # (backend, chunk) -> {run1:..., run2:...}
    data = defaultdict(dict)
    for fn in sorted(os.listdir(d)):
        m = re.match(r"(.+)_c(\d+)_(run[12])\.log$", fn)
        if not m:
            continue
        backend, chunk, run = m.group(1), int(m.group(2)), m.group(3)
        gets, puts = parse_log(os.path.join(d, fn))
        rows = parse_csv(os.path.join(d, fn[:-4] + ".csv"))
        data[(backend, chunk)][run] = {"gets": gets, "puts": puts, "rows": rows}

    if not data:
        print(f"{d} 에 결과 없음")
        return 1

    # ── 표 1: run2 기준 (cross-process 재사용 성능) ──
    print("\n" + "=" * 92)
    print("run2 (cross-process 재사용) — 이전 프로세스가 저장한 KV를 읽음")
    print("=" * 92)
    print(f"{'backend':<22}{'chunk':>6}{'KV':>6}"
          f"{'get avg':>10}{'get min':>9}{'get max':>9}"
          f"{'GB/s':>7}{'hit':>5}{'put(run1)':>11}")
    print("-" * 92)

    for (backend, chunk) in sorted(data.keys(), key=lambda k: (k[0], k[1])):
        r2 = data[(backend, chunk)].get("run2")
        if not r2:
            continue
        costs = [g[1] for g in r2["gets"]]
        gbps = [g[2] for g in r2["gets"]]
        r1 = data[(backend, chunk)].get("run1", {})
        putms = [p[2] for p in r1.get("puts", [])]
        label = BACKEND_LABEL.get(backend, backend)
        kv = KV_MB.get(chunk, "?")
        print(f"{label:<22}{chunk:>6}{str(kv)+'MB':>6}"
              f"{avg(costs):>10.2f}{min(costs) if costs else 0:>9.2f}"
              f"{max(costs) if costs else 0:>9.2f}"
              f"{avg(gbps):>7.2f}{len(costs):>5}{avg(putms):>11.2f}")

    # ── 표 2: persistence (run1 vs run2 step별 cached) ──
    print("\n" + "=" * 92)
    print("persistence — run1 대비 run2가 앞서 재사용하는가")
    print("=" * 92)
    print(f"{'backend':<22}{'chunk':>6}   {'step별 cached (run1 → run2)':<44}{'Retrieved':>12}")
    print("-" * 92)

    for (backend, chunk) in sorted(data.keys(), key=lambda k: (k[0], k[1])):
        e = data[(backend, chunk)]
        r1, r2 = e.get("run1"), e.get("run2")
        if not (r1 and r2):
            continue
        c1 = [r["cached"] for r in r1["rows"]]
        c2 = [r["cached"] for r in r2["rows"]]
        pairs = " ".join(
            f"{a}→{b}" + ("*" if b > a else "")
            for a, b in zip(c1, c2)
        )
        label = BACKEND_LABEL.get(backend, backend)
        ret = f"{len(r1['gets'])}→{len(r2['gets'])}"
        print(f"{label:<22}{chunk:>6}   {pairs:<44}{ret:>12}")

    print("\n  * = run2가 더 많이 재사용 (= 이전 프로세스 KV를 읽음)")
    print("  step1이 0→N 이면 새 프로세스가 첫 요청부터 hit — persistence 성립")

    # ── 상세 ──
    if detail:
        for (backend, chunk) in sorted(data.keys(), key=lambda k: (k[0], k[1])):
            e = data[(backend, chunk)]
            print("\n" + "=" * 60)
            print(f"{BACKEND_LABEL.get(backend, backend)} / chunk {chunk}")
            print("=" * 60)
            for run in ("run1", "run2"):
                if run not in e:
                    continue
                r = e[run]
                print(f"\n  [{run}]")
                print(f"    {'step':>5}{'ptok':>7}{'cached':>8}{'reuse%':>8}")
                for row in r["rows"]:
                    pct = row["cached"] / row["ptok"] * 100 if row["ptok"] else 0
                    print(f"    {row['step']:>5}{row['ptok']:>7}"
                          f"{row['cached']:>8}{pct:>7.1f}")
                if r["gets"]:
                    print(f"    get  : " + ", ".join(f"{g[1]:.2f}ms" for g in r["gets"]))
                if r["puts"]:
                    print(f"    put  : " + ", ".join(f"{p[3]:.2f}ms" for p in r["puts"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
