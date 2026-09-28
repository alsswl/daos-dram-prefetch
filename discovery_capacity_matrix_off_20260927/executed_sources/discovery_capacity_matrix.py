#!/usr/bin/env python3
"""Six replay cases: DRAM 8/4/2 GiB x staging 8/4 GiB, DRAM prefetch ON or OFF."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import compare_e2e as common
from discovery_fixed_replay import run_case

ROOT = common.ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--requests', type=Path, default=ROOT/'discovery_fixed_replay_20260926/requests.json')
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--dram-prefetch', choices=('on', 'off'), default='on')
    a = p.parse_args()
    records = json.loads(a.requests.read_text())
    if len(records) != 256 or [r['index'] for r in records] != list(range(256)):
        p.error('Expected the unchanged 256-request replay manifest')
    for r in records:
        assert hashlib.sha256(r['prompt'].encode()).hexdigest() == r['prompt_sha256']
    a.model, a.max_model_len, a.capacity_failure_probe = 'Qwen/Qwen3-14B', 32768, True
    folder = a.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    common.dump(folder/'requests.json', records)
    enabled = a.dram_prefetch == 'on'
    cases = [dict(name=f'd{d}_s{s}',cpu_gib=d,staging_gib=s,concurrency=16,prefetch=enabled)
             for d in (8,4,2) for s in (8,4)]
    sources = ['discovery_capacity_matrix.py','report_capacity_matrix.py','discovery_fixed_replay.py',
        'analyze_discovery_staging.py','staging_mixed_pressure.py','compare_e2e.py','run_vllm.sh',
        'lmcache_config_daosgds_async_dram.yaml','libdaosgdr.so','libdaosgdr.c']
    sources += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'lmcache_daos').glob('*.py'))]
    hashes = {}
    for name in sources:
        dst = folder/'executed_sources'/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    common.dump(folder/'plan.json',dict(cases=cases,model=a.model,chunk_tokens=128,
        max_model_len=a.max_model_len,requests_per_case=256,concurrency=16,
        source_requests=str(a.requests.resolve()),
        requests_sha256=hashlib.sha256((folder/'requests.json').read_bytes()).hexdigest(),
        source_sha256=hashes, generation=dict(temperature=0,seed=0,max_tokens=2048,
            enable_thinking=False,stop=['\nObservation:']),
        notes=[f'All six cases DRAM prefetch {a.dram_prefetch.upper()}; DAOS prefetch retained. ON uses capacity policy, no soft watermark.',
            'Fresh process, empty DRAM, new DAOS namespace per case; no shared deletion or server-cache flush.',
            'Same 256 recorded inputs; wave size16, no Python execution; response-paced waves.',
            'One trial per combination, sequential listed order. EOS enabled; output length may differ.',
            'Existing DAOS serializer and async DRAM mirror budgets retained; physical GPU limit enforced.',
            'Diagnostic subclass records per-read allocation failure without changing parallel GET/prefix behavior.',
            'Main recompute ratio = verified capacity-caused lost prefix tokens / all prompt tokens.',
            'Cold misses and successful prefix retrieval are not counted as capacity-caused recomputation.']))
    if a.dry_run:
        print(folder)
        return
    try:
        for spec in cases:
            case = folder/spec['name']; case.mkdir()
            a.cpu_gb, a.staging_gib = spec['cpu_gib'], spec['staging_gib']
            common.dump(folder/'status.json',dict(status='running',case=spec['name']))
            run_case(a,case,records,16,enabled)
        common.dump(folder/'status.json',dict(status='completed'))
    except BaseException as exc:
        common.dump(folder/'status.json',dict(status='failed',error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
