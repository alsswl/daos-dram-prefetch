#!/usr/bin/env python3
"""Matched rolling C8/C16 OFF/ON diagnostics; preserve original execution policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import compare_e2e as common
from discovery_rolling_replay import run_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    a.model, a.max_model_len = 'Qwen/Qwen3-14B', 32768
    a.cpu_gb, a.staging_gib = 8, 8
    folder = a.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source = common.ROOT/'discovery_capacity_matrix_20260927/requests.json'
    records = json.loads(source.read_text())
    assert len(records) == 256
    assert all(r['index'] == i and hashlib.sha256(r['prompt'].encode()).hexdigest() == r['prompt_sha256']
               for i, r in enumerate(records))
    common.dump(folder/'requests.json', records)
    cases = [dict(name=f'r{r}_c{c}_{"on" if enabled else "off"}', repeat=r,
                  concurrency=c, prefetch=enabled)
             for r, settings in [(1, [(8, False), (8, True), (16, True), (16, False)]),
                                 (2, [(16, False), (16, True), (8, True), (8, False)])]
             for c, enabled in settings]
    names = ['diagnose_prefetch_timing.py', 'discovery_rolling_replay.py', 'discovery_fixed_replay.py',
             'staging_mixed_pressure.py', 'compare_e2e.py', 'run_vllm.sh', 'libdaosgdr.so',
             'lmcache_config_daosgds_async_dram.yaml', 'tests/test_prefetch_timing.py']
    names += [str(p.relative_to(common.ROOT)) for p in sorted((common.ROOT/'lmcache_daos').glob('*.py'))]
    hashes = {}
    for name in names:
        dest = folder/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(common.ROOT/name, dest)
        hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    common.dump(folder/'plan.json', dict(cases=cases, model=a.model, cpu_gib=8, staging_gib=8,
        chunk_tokens=128, arrival_mode='rolling', requests_per_case=256, source_sha256=hashes,
        notes=['Two repetitions per condition, reversed four-case order.',
               'Fresh process/DRAM and unique DAOS namespace per case; no storage cache flush.',
               'Host timestamps only: copy interval includes enqueue and stream synchronization.',
               'No new CUDA synchronization; no change to allocation, copying or scheduling policy.',
               'EOS retained; output lengths and cache residency may vary.',
               'Diagnostic timing overhead exists; not a pure DMA or counterfactual critical-path measurement.']))
    if a.dry_run:
        common.dump(folder/'status.json', dict(status='dry_run')); return
    os.environ['DAOS_GDS_PREFETCH_TIMING'] = '1'
    try:
        cmd = [str(common.ROOT/'run_vllm.sh'), str(common.ROOT/'venv/bin/python3'),
               str(common.ROOT/'tests/object_gpu_roundtrip.py'), '--size-mib', '20']
        result = subprocess.run(cmd, cwd=common.ROOT, env=dict(os.environ, DAOSGDS_TRANSPORT='object'),
                                capture_output=True, text=True, timeout=120)
        (folder/'storage_preflight.log').write_text(result.stdout + result.stderr)
        if result.returncode: raise RuntimeError('DAOS preflight failed; see storage_preflight.log')
        for spec in cases:
            case = folder/spec['name']; case.mkdir()
            common.dump(folder/'status.json', dict(status='running', case=spec['name']))
            run_case(a, case, records, spec['concurrency'], spec['prefetch'])
        common.dump(folder/'status.json', dict(status='completed'))
    except BaseException as exc:
        common.dump(folder/'status.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
