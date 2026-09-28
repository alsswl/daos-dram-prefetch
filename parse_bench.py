#!/usr/bin/env python3
"""
LMCache 벤치마크 로그 파서.
kv_measure 실행 로그(DEBUG 레벨) + CSV에서 3개 메트릭을 뽑는다:
  1) get 지연     - "Retrieved X out of Y ... cost C ms, throughput T GB/s"
  2) I/O 패턴     - "Disk read/write size: N bytes, Bandwidth: B MB/s" + read/write 순서
  3) 재사용률     - CSV의 cached_tokens / prompt_tokens (step별)
사용:
  python parse_bench.py <label> <run.log> [run.csv]
예:
  python parse_bench.py "LocalDisk-diffproc" ~/L_run2_dbg.log ~/L_run2.csv
"""
import sys, re, os

def parse_log(path):
    retr, disk = [], []   # retrieved(get 지연), disk io
    scan = []             # "Read N cache entries" (GDS persistence 신호)
    if not os.path.exists(path):
        return retr, disk, scan
    ts_re = re.compile(r'(\d{2}:\d{2}:\d{2})[.,](\d+)')
    for line in open(path, errors='ignore'):
        # get 지연
        m = re.search(r'Retrieved (\d+) out of (\d+) required tokens.*?cost ([\d.]+) ms, throughput: ([\d.]+) GB/s', line)
        if m:
            t = ts_re.search(line)
            retr.append(dict(ts=t.group(0) if t else '', got=int(m.group(1)), req=int(m.group(2)),
                             cost_ms=float(m.group(3)), gbps=float(m.group(4))))
            continue
        # disk io
        m = re.search(r'Disk (read|write) size: (\d+) bytes, Bandwidth: ([\d.]+) MB/s', line)
        if m:
            t = ts_re.search(line)
            disk.append(dict(ts=t.group(0) if t else '', op=m.group(1),
                             bytes=int(m.group(2)), mbps=float(m.group(3))))
            continue
        # gds persistence scan
        m = re.search(r'Read (\d+) cache entries from persistent storage', line)
        if m:
            scan.append(int(m.group(1)))
    return retr, disk, scan

def parse_csv(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    import csv
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append(dict(step=int(r['step']), ptok=int(r['prompt_tokens']),
                                 cached=int(r['cached_tokens']), reuse=float(r['reuse_ratio'])))
            except (KeyError, ValueError):
                pass
    return rows

def summarize(label, logp, csvp=None):
    retr, disk, scan = parse_log(logp)
    rows = parse_csv(csvp)
    print(f"\n{'='*70}\n[{label}]\n{'='*70}")

    # 1) persistence 신호
    if scan:
        print(f"  persistence scan : Read {scan} cache entries (GDS startup)")
    else:
        print(f"  persistence scan : (없음 - LocalDisk이거나 scan 안 함)")

    # 2) I/O 패턴 + read/write 순서
    reads  = [d for d in disk if d['op']=='read']
    writes = [d for d in disk if d['op']=='write']
    print(f"\n  -- I/O 패턴 --")
    print(f"  disk write : {len(writes)}회", end='')
    if writes: print(f", 평균 {sum(w['mbps'] for w in writes)/len(writes):.1f} MB/s", end='')
    print()
    print(f"  disk read  : {len(reads)}회", end='')
    if reads: print(f", 평균 {sum(r['mbps'] for r in reads)/len(reads):.1f} MB/s", end='')
    print()
    # 순서 판별: 첫 write vs 첫 read 타임스탬프
    if writes and reads:
        fw, fr = writes[0]['ts'], reads[0]['ts']
        order = "WRITE 먼저 → 자기 것 읽음 (세션 내 spill, cross-proc 아님)" if fw < fr \
                else "READ 먼저 → 남의 것 읽음 (cross-proc persistence 가능성)"
        print(f"  순서       : 첫write={fw} 첫read={fr} → {order}")
    elif reads and not writes:
        print(f"  순서       : write 없이 read만 → 순수 로드 (cross-proc persistence)")

    # 3) get 지연
    print(f"\n  -- get 지연 (Retrieved) --")
    hits = [r for r in retr if r['got']>0]
    if hits:
        costs = [r['cost_ms'] for r in hits]
        gbps  = [r['gbps'] for r in hits]
        print(f"  hit 횟수   : {len(hits)}")
        print(f"  cost(ms)   : min {min(costs):.3f} / avg {sum(costs)/len(costs):.3f} / max {max(costs):.3f}")
        print(f"  throughput : avg {sum(gbps)/len(gbps):.3f} GB/s")
    else:
        print(f"  hit 없음 (retrieve 0 out of N)")

    # 4) 재사용률
    if rows:
        print(f"\n  -- 재사용률 (step별) --")
        print(f"  {'step':>4} {'ptok':>5} {'cached':>6} {'reuse%':>7}")
        for r in rows:
            print(f"  {r['step']:>4} {r['ptok']:>5} {r['cached']:>6} {r['reuse']*100:>6.1f}")
        c = [r for r in rows if r['cached']>0]
        if c:
            print(f"  첫 cached step: {c[0]['step']}  (이 전은 청크 미완성)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python parse_bench.py <label> <run.log> [run.csv]")
        sys.exit(1)
    summarize(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else None)
