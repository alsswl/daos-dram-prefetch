#!/usr/bin/env python3
"""Repeated fixed-input DRAM-prefetch comparison, DRAM8/staging8/concurrency8."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import compare_e2e as common
from discovery_fixed_replay import run_case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--requests',type=Path,default=common.ROOT/'discovery_capacity_matrix_20260927/requests.json')
    p.add_argument('--port',type=int,default=8017)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    records=json.loads(a.requests.read_text())
    assert len(records)==256 and [r['index'] for r in records]==list(range(256))
    assert all(hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256'] for r in records)
    a.model,a.max_model_len='Qwen/Qwen3-14B',32768
    a.cpu_gb,a.staging_gib,a.capacity_failure_probe=8,8,True
    folder=a.output.resolve(); folder.mkdir(parents=True,exist_ok=False)
    common.dump(folder/'requests.json',records)
    cases=[]
    for repeat,modes in enumerate(((False,True),(True,False),(False,True)),1):
        for enabled in modes:
            cases.append(dict(name=f'r{repeat}_{"on" if enabled else "off"}',repeat=repeat,
                              prefetch=enabled,cpu_gib=8,staging_gib=8,concurrency=8))
    names=['repeat_prefetch_c8.py','report_prefetch_c8.py','discovery_fixed_replay.py',
           'report_capacity_matrix.py','analyze_discovery_staging.py','staging_mixed_pressure.py',
           'compare_e2e.py','run_vllm.sh','lmcache_config_daosgds_async_dram.yaml','libdaosgdr.so','libdaosgdr.c']
    names += [str(p.relative_to(common.ROOT)) for p in sorted((common.ROOT/'lmcache_daos').glob('*.py'))]
    hashes={}
    for name in names:
        dest=folder/'executed_sources'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(common.ROOT/name,dest)
        hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    common.dump(folder/'plan.json',dict(cases=cases,model=a.model,chunk_tokens=128,
        max_model_len=a.max_model_len,requests_per_case=256,concurrency=8,source_sha256=hashes,
        source_requests=str(a.requests.resolve()),
        notes=['Fresh process, empty DRAM, new DAOS namespace per run; no server/OS cache flush.',
               'DRAM prefetch only toggle; DAOS prefetch, asynchronous mirror/promotion retained.',
               'ON capacity policy without soft watermark; physical limits and DAOS serializer retained.',
               '256 unchanged prompts, response-paced waves of8; EOS enabled; no Python tool execution.',
               'Three process-level repetitions each; OFF/ON, ON/OFF, OFF/ON order.',
               'vLLM max-num-seqs remains16; client concurrency is8.']))
    if a.dry_run: print(folder); return
    try:
        for spec in cases:
            case=folder/spec['name']; case.mkdir()
            common.dump(folder/'status.json',dict(status='running',case=spec['name']))
            run_case(a,case,records,8,spec['prefetch'])
        common.dump(folder/'status.json',dict(status='completed'))
    except BaseException as exc:
        common.dump(folder/'status.json',dict(status='failed',error=repr(exc)))
        raise


if __name__=='__main__': main()
