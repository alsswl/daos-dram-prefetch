#!/usr/bin/env python3
"""Audited cleanup for the completed Qwen4B diagnostic, exact namespace only.

The legacy cleanup's Qwen3-14B key filter cannot match offline model paths.
Keep its original no-op manifest intact and produce a separate manifest.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import yaml
import cleanup_experiment_cache as cleaner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args = p.parse_args()
    case = args.case.resolve()
    root = cleaner.ROOT
    read = lambda path: json.loads(path.read_text())
    assert case.parent.parent == root
    plan = read(case.parent/'plan.json')
    assert plan['model'] == 'Qwen/Qwen3-4B-Instruct-2507'
    assert case.name in {s['name'] for s in plan['cases']}
    assert read(case/'status.json')['status'] == 'completed'
    assert read(case/'payload_policy_check.json')['passed'] is True
    ec = yaml.safe_load((case/'config.yaml').read_text())['extra_config']
    ns = ec['daosgds.object_namespace']
    assert re.fullmatch(r'minji-cold-warm-[0-9a-f]{32}:',ns)
    assert ec['daosgds.pool']=='discospool' and ec['daosgds.container']=='kvcache'
    assert ec['daosgds.transport']=='object'
    assert Path(ec['daosgds.object_library']).resolve()==root/'libdaosgdr.so'
    model_names = set(re.findall(r"LMCacheMetadata\(model_name='([^']+)'",(case/'server.log').read_text()))
    assert len(model_names)==1
    model_name = model_names.pop()
    assert re.fullmatch(r'/home/hf/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/[0-9a-f]{40}',model_name)
    rows=[dict(namespace=ns,model_key_prefix=model_name+'@',
               sources=[str((case/'config.yaml').relative_to(root))],
               pool='discospool',container='kvcache',experiments=[case.parent.name])]
    cleaner.eligible = lambda: rows
    def classify(keys,selected):
        assert selected==rows
        targets=[]
        for key in keys:
            if key.startswith(ns):
                assert key[len(ns):].startswith(model_name+'@'), 'Unexpected model inside namespace; stop'
                targets.append(key)
        return targets,dict(Counter({ns:len(targets)}))
    cleaner.classify = classify
    folder=case/'namespace_cleanup_q4'
    sys.argv=['cleanup','--output',str(folder)]+(['--execute'] if args.execute else [])
    cleaner.main()


if __name__=='__main__':
    main()
