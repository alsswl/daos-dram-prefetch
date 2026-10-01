#!/usr/bin/env python3
"""Manifest-scoped KV cleanup; list first, explicit --execute second.

Only the allowlisted, completed local replay experiments are eligible. Never
punch a container/object or use a broad minji-* prefix. Logs are not deleted.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

import yaml

ROOT = Path(__file__).resolve().parent
POOL = 'f973c142-2353-41da-b154-5079ba6969f2'
CONTAINER = 'a2d875e3-195b-4598-98b4-b381ef49867a'
OID = '281543696187392.1000'
EXPERIMENTS = {
    'prefetch_cold_warm_d8_s8_20260927',
    'prefetch_d8_s8_c16_rolling_repeat3_20260927',
    'prefetch_timing_d8_s8_20260927',
    'queued_prefetch_experiment_20260927',
    'discovery_async_dram_20260926',
    'discovery_async_dram_prefetch_20260926',
    'discovery_staging_long_20260926_v2',
    'discovery_staging_pilot_20260926_v1',
    'discovery_staging_pilot_20260926_v2',
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pool_query():
    return json.loads(subprocess.check_output(['daos','-j','pool','query',POOL], text=True))


def list_keys():
    data=subprocess.check_output([str(ROOT/'list_experiment_dkeys')])
    offset=0; keys=[]
    while offset<len(data):
        assert len(data)-offset>=4
        length=struct.unpack_from('=I',data,offset)[0]; offset+=4
        assert 0<length<=1024*1024 and offset+length<=len(data)
        key=data[offset:offset+length]; offset+=length
        assert b'\0' not in key
        keys.append(key.decode('utf-8'))
    assert len(keys)==len(set(keys)), 'Unstable/duplicate enumeration'
    return sorted(keys)


def eligible():
    rows=json.loads((ROOT/'namespace_review_20260927_current/inventory.json').read_text())['namespaces']
    selected=[]
    for row in rows:
        if not row['experiments'] or not set(row['experiments'])<=EXPERIMENTS: continue
        assert row['pool']=='discospool' and row['container']=='kvcache'
        assert not row['has_default_reference'] and row['execution_logs_found']
        assert row['run_statuses'] and all(s['status'] in ('completed','complete') for s in row['run_statuses'])
        ns=row['namespace']
        assert ns.endswith(':') and len(ns)>35
        assert ns not in ('minji-v2:','minji-async-dram:','minji-gpu-store:')
        for name in row['sources']:
            path=ROOT/name
            assert path.resolve().is_relative_to(ROOT) and path.parts[-1].endswith(('.yaml','.yml'))
            cfg=yaml.safe_load(path.read_text()); ec=cfg['extra_config']
            assert ec['daosgds.pool']=='discospool' and ec['daosgds.container']=='kvcache'
            assert ec['daosgds.transport']=='object' and ec['daosgds.object_namespace']==ns
        selected.append(row)
    assert len(selected)==30, f'Unexpected scope: {len(selected)} namespaces'
    return sorted(selected,key=lambda r:r['namespace'])


def classify(keys, rows):
    namespaces={r['namespace'] for r in rows}
    targets=[]; counts=Counter()
    for key in keys:
        ns,sep,suffix=key.partition(':')
        ns+=':'
        if sep and ns in namespaces and suffix.startswith('Qwen/Qwen3-14B@'):
            targets.append(key); counts[ns]+=1
    return targets,dict(counts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(); folder=a.output.resolve()
    assert folder.is_relative_to(ROOT)
    rows=eligible()
    if not a.execute:
        folder.mkdir(parents=True,exist_ok=False)
        keys=list_keys(); targets,counts=classify(keys,rows)
        dump(folder/'before_keys.json',keys); dump(folder/'target_keys.json',targets)
        dump(folder/'pool_before.json',pool_query())
        manifest=dict(pool=POOL,container=CONTAINER,oid=OID,namespaces=rows,
            target_count=len(targets),total_count=len(keys),preserved_count=len(keys)-len(targets),
            per_namespace=counts,target_sha256=digest(folder/'target_keys.json'),
            binding_sha256=digest(ROOT/'lmcache_daos/object_binding.py'),
            library_sha256=digest(ROOT/'libdaosgdr.so'),
            source_sha256=digest(ROOT/'libdaosgdr.c'))
        dump(folder/'manifest.json',manifest)
        print(json.dumps({k:v for k,v in manifest.items() if k!='namespaces'},indent=2))
        return
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['pool']==POOL and manifest['container']==CONTAINER and manifest['oid']==OID
    assert manifest['namespaces']==rows
    for name,key in [('libdaosgdr.so','library_sha256'),('libdaosgdr.c','source_sha256'),
                     ('lmcache_daos/object_binding.py','binding_sha256')]:
        assert digest(ROOT/name)==manifest[key]
    assert digest(folder/'target_keys.json')==manifest['target_sha256']
    before=json.loads((folder/'before_keys.json').read_text())
    targets=json.loads((folder/'target_keys.json').read_text())
    assert list_keys()==before, 'Keys changed since read-only planning; abort'
    assert classify(before,rows)[0]==targets and len(targets)==manifest['target_count']
    # Never automatically resume a partly executed deletion using stale counts.
    with (folder/'deletion.jsonl').open('x') as journal:
        from lmcache_daos.object_binding import DaosObjectStore
        store=DaosObjectStore(POOL,CONTAINER,library_path=str(ROOT/'libdaosgdr.so'))
        try:
            for i,key in enumerate(targets,1):
                store.remove(key)
                journal.write(json.dumps(dict(index=i,key=key))+'\n')
                if i%500==0:
                    journal.flush(); print(f'deleted {i}/{len(targets)}',flush=True)
        finally:
            store.close()
    after=list_keys(); dump(folder/'after_keys.json',after)
    assert set(after)==set(before)-set(targets), 'Preserved key set changed; inspect before/after'
    assert not classify(after,rows)[0]
    dump(folder/'pool_after.json',pool_query())
    result=dict(deleted=len(targets),remaining_targets=0,preserved=len(after),
                preserved_set_unchanged=True,pool=POOL,container=CONTAINER,oid=OID,
                namespaces=len(rows),payload_backup=False,logs_preserved=True,completed_ns=time.time_ns())
    dump(folder/'result.json',result); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
