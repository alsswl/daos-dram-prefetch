"""User-authorized old cache cleanup for the target-spread cold/warm run.

Exact two old experiment namespaces; preserve all logs and other keys.
"""
from collections import Counter
import json
import cleanup_experiment_cache as cleaner

NAMESPACES=(
    'minji-windowed-b75ac64b95c24300a94a771fef30132e:',
    'minji-cold-warm-d8a5d478733f47a5bbd6172391669895:',
)
rows=[dict(namespace=ns,pool='discospool',container='kvcache',
           authorization='User explicitly requested deleting old caches to run target-count cold/warm CXS',
           prior_inventory='realqa_placement_20261002/previous_cache_cleanup_plan/after_keys.json') for ns in NAMESPACES]


def classify(keys,selected):
    assert selected==rows
    targets=[k for k in keys if k.partition(':')[0]+':' in NAMESPACES]
    approved=json.loads((cleaner.ROOT/'cxs_target_coldwarm_readonly_20261002/target_keys.json').read_text())
    assert not targets or (targets==approved and len(targets)==36132)
    return targets,dict(Counter(k.split(':',1)[0]+':' for k in targets))


cleaner.eligible=lambda:rows
cleaner.classify=classify
if __name__=='__main__':cleaner.main()
