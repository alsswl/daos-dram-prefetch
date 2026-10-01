"""Read-only inventory by default; exact prior CXS namespace only.

--execute is a separate operation requiring approval for the listed keys.
"""
from collections import Counter
import json
import yaml
import cleanup_experiment_cache as cleaner

root=cleaner.ROOT/'realqa_mixed_nodram2g_20261002_w2048'
plan=json.loads((root/'plan.json').read_text());case=root/plan['cases'][0]['name']
ns=plan['experiment_namespace']
assert json.loads((root/'status.json').read_text())['status']=='completed'
assert json.loads((case/'no_dram_check.json').read_text())['passed']
cfg=yaml.safe_load((case/'config.yaml').read_text())['extra_config']
assert cfg['daosgds.object_namespace']==ns and ns.startswith('minji-mixed-')
assert cfg['daosgds.pool']=='discospool' and cfg['daosgds.container']=='kvcache'
rows=[dict(namespace=ns,sources=[str((case/'config.yaml').relative_to(cleaner.ROOT))],
           pool='discospool',container='kvcache',experiments=[root.name])]


def classify(keys,selected):
    assert selected==rows
    targets=[k for k in keys if k.startswith(ns)]
    assert all(k[len(ns):].startswith('Qwen/Qwen3-4B-Instruct-2507@') for k in targets)
    return targets,dict(Counter({ns:len(targets)}))


cleaner.eligible=lambda:rows
cleaner.classify=classify
if __name__=='__main__':cleaner.main()
