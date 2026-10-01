"""Manifest-scoped deletion of only the completed split experiment cache."""
import sys
import json
from collections import Counter
import yaml
import cleanup_experiment_cache as cleaner

root=cleaner.ROOT
experiment=root/'realqa_q4_64k_split_r1536_s512_w500_20261001'
case=experiment/'c8_s10_split_r1536_s512_w500'
assert json.loads((case/'status.json').read_text())['status']=='completed'
ec=yaml.safe_load((case/'config.yaml').read_text())['extra_config']
ns=ec['daosgds.object_namespace']
assert ns.startswith('minji-windowed-') and ns.endswith(':')
assert ec['daosgds.pool']=='discospool' and ec['daosgds.container']=='kvcache'
rows=[dict(namespace=ns,sources=[str((case/'config.yaml').relative_to(root))],
           pool='discospool',container='kvcache',experiments=[experiment.name])]
cleaner.eligible=lambda:rows
def classify(keys,selected):
    assert selected==rows
    targets=[k for k in keys if k.startswith(ns)]
    assert all(k[len(ns):].startswith('Qwen/Qwen3-4B-Instruct-2507@') for k in targets)
    return targets,dict(Counter({ns:len(targets)}))
cleaner.classify=classify
execute='--execute' in sys.argv
sys.argv=['cleanup','--output',str(experiment/'mixed_rerun_cleanup')]+(['--execute'] if execute else [])
cleaner.main()
