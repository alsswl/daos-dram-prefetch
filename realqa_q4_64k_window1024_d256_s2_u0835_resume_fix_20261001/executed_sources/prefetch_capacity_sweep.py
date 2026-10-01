#!/usr/bin/env python3
"""36 cases: concurrency8/16 x DRAM8/4/2 x staging8/4 x off/wait/cancel.

Each independently cold process runs the same cold256 then warm256. No cache
deletion, artificial copy delays or soft staging watermark in this runner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import compare_e2e as common
from cold_warm_prefetch import run_case

MODES = {
    'off': dict(prefetch=False, cancel_queued=False, early_ready=False),
    'wait': dict(prefetch=True, cancel_queued=False, early_ready=False),
    'cancel': dict(prefetch=True, cancel_queued=True, early_ready=True),
}


def cases_for():
    result=[]
    for concurrency in (8,16):
        for cpu in (8,4,2):
            for staging in (8,4):
                modes=list(MODES)
                # Rotate arm order for successive cells, fixed before results.
                rotation=(len(result)//3)%3
                modes=modes[rotation:]+modes[:rotation]
                for mode in modes:
                    result.append(dict(name=f'c{concurrency}_d{cpu}_s{staging}_{mode}',
                        concurrency=concurrency,cpu_gib=cpu,staging_gib=staging,mode=mode,
                        **MODES[mode]))
    return result


def pool_query():
    result=subprocess.run([str(common.ROOT/'run_vllm.sh'),'daos','-j','pool','query','discospool'],
        cwd=common.ROOT,env=dict(os.environ,DAOSGDS_TRANSPORT='object'),capture_output=True,text=True,timeout=30)
    if result.returncode: raise RuntimeError(result.stderr)
    query=json.loads(result.stdout)
    if query['status'] or query['error']: raise RuntimeError(str(query))
    return query


def check_space(query):
    pool=query['response']
    assert pool['uuid']=='f973c142-2353-41da-b154-5079ba6969f2'
    nvme=next(t for t in pool['tier_stats'] if t['media_type']=='nvme')
    if pool['disabled_targets'] or pool['rebuild']['status'] or pool['state']!='Ready':
        raise RuntimeError('Pool unhealthy, do not run comparison')
    # Conservative guard, NOT a DAOS guarantee. Previously NOSPACE near 6GB min.
    if nvme['free']<220_000_000_000 or nvme['min']<10_000_000_000:
        raise RuntimeError(f"Insufficient safe headroom: free={nvme['free']} min={nvme['min']}")
    return nvme


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=8017)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); a.model='Qwen/Qwen3-14B'; a.max_model_len=32768
    root=a.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    records=json.loads((common.ROOT/'discovery_capacity_matrix_20260927/requests.json').read_text())
    assert len(records)==256 and all(r['index']==i and hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256'] for i,r in enumerate(records))
    common.dump(root/'requests.json',records)
    names=['prefetch_capacity_sweep.py','report_prefetch_capacity_sweep.py','cold_warm_prefetch.py',
           'discovery_fixed_replay.py','discovery_rolling_replay.py','compare_e2e.py',
           'staging_mixed_pressure.py','report_prefetch_timing.py','report_cold_warm_prefetch.py',
           'report_capacity_matrix.py','analyze_discovery_staging.py',
           'run_vllm.sh','libdaosgdr.so','lmcache_config_daosgds_async_dram.yaml']
    names += [str(x.relative_to(common.ROOT)) for x in sorted((common.ROOT/'lmcache_daos').glob('*.py'))]
    hashes={}
    for name in names:
        dst=root/'executed_sources'/name; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(common.ROOT/name,dst); hashes[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
    cases=cases_for()
    common.dump(root/'plan.json',dict(cases=cases,requests_per_phase=256,requests_total=18432,
        model=a.model,chunk_tokens=128,repeats=1,source_sha256=hashes,
        notes=['Recorded DiscoveryBench replay; not live Python/full agentic.',
            'Fresh process, empty DRAM and UUID namespace per case. Same live cache cold->warm.',
            'DRAM prefetch off / on-wait / on-cancel. DAOS prefetch always enabled.',
            'Cancel also changes early readiness/serializer lifetime; no early_wait-only control in this sweep.',
            'Physical staging limits only. No artificial queue waits. Store/read promotion unchanged.',
            'Rolling same input order, not identical wall-clock arrivals or generated lengths.',
            'Pool/OS caches not flushed. No deletions by this runner. One trial per cell.']))
    if a.dry_run:
        common.dump(root/'status.json',dict(status='dry_run')); print(root); return
    os.environ['DAOS_GDS_PREFETCH_TIMING']='1'
    completed=[]
    try:
        space=pool_query(); common.dump(root/'pool_before.json',space); check_space(space)
        cmd=[str(common.ROOT/'run_vllm.sh'),str(common.ROOT/'venv/bin/python3'),
             str(common.ROOT/'tests/object_gpu_roundtrip.py'),'--size-mib','20']
        result=subprocess.run(cmd,cwd=common.ROOT,env=dict(os.environ,DAOSGDS_TRANSPORT='object'),
                              capture_output=True,text=True,timeout=120)
        (root/'storage_preflight.log').write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError('DAOS roundtrip preflight failed')
        for i,spec in enumerate(cases,1):
            assert all(hashlib.sha256((common.ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
            case=root/spec['name']; case.mkdir()
            space=pool_query(); common.dump(case/'pool_before.json',space); nvme=check_space(space)
            common.dump(root/'status.json',dict(status='running',case=spec['name'],index=i,
                                               total=len(cases),completed=completed))
            print(f"CASE {i}/36 {spec['name']} free={nvme['free']/1e9:.1f}GB min={nvme['min']/1e9:.1f}GB",flush=True)
            for key in ('cpu_gib','staging_gib','cancel_queued','early_ready'): setattr(a,key,spec[key])
            run_case(a,case,records,spec['concurrency'],spec['prefetch'])
            common.dump(case/'pool_after.json',pool_query())
            completed.append(spec['name'])
        common.dump(root/'status.json',dict(status='completed',completed=completed))
        subprocess.run([str(common.ROOT/'venv/bin/python3'),str(common.ROOT/'report_prefetch_capacity_sweep.py'),str(root)],check=True)
    except BaseException as exc:
        common.dump(root/'status.json',dict(status='failed',error=repr(exc),completed=completed)); raise


if __name__=='__main__': main()
