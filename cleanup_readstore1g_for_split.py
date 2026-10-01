"""Exact completed experiment namespace only; prepare manifest before execute."""
import sys
import json
from collections import Counter
import yaml
import cleanup_experiment_cache as cleaner

root = cleaner.ROOT
experiment = root/'realqa_q4_64k_read500_store500_d256_s1_20261001'
case = experiment/'c8_s10_window500'
assert json.loads((case/'status.json').read_text())['status']=='completed'
ec = yaml.safe_load((case/'config.yaml').read_text())['extra_config']
ns = 'minji-windowed-7f376e2c70b144e78ae04c8f2aba3c4f:'
assert ec['daosgds.object_namespace']==ns
assert ec['daosgds.pool']=='discospool' and ec['daosgds.container']=='kvcache'
rows = [dict(namespace=ns, sources=[str((case/'config.yaml').relative_to(root))],
             pool='discospool',container='kvcache',experiments=[experiment.name])]
cleaner.eligible=lambda:rows
def classify(keys, selected):
    assert selected==rows
    targets=[k for k in keys if k.startswith(ns)]
    assert all(k[len(ns):].startswith('Qwen/Qwen3-4B-Instruct-2507@') for k in targets)
    return targets, dict(Counter({ns:len(targets)}))
cleaner.classify=classify
execute='--execute' in sys.argv
sys.argv=['cleanup','--output',str(experiment/'split_rerun_cleanup')]+(['--execute'] if execute else [])
cleaner.main()
