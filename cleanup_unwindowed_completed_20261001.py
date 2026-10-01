"""Audited cleanup of the completed s10 unwindowed v2 private namespace."""
import json
import yaml
import cleanup_experiment_cache as c
from cleanup_window1024_20261001 import classify


def eligible():
    root=c.ROOT/'realqa_q4_64k_unwindowed_d256_s10_20261001_v2'
    case=root/'c8_s10_window0'
    read=lambda p:json.loads(p.read_text())
    assert read(root/'status.json')['status']=='completed'
    assert read(case/'status.json')['status']=='completed'
    assert read(case/'window_policy_check.json')['passed']
    calls=read(case/'calls.json')
    assert len(calls)==480 and all(x['status']=='success' for x in calls)
    ec=yaml.safe_load((case/'config.yaml').read_text())['extra_config']
    ns='minji-windowed-56d2edbbe832411db929a432ab66e75c:'
    assert ec['daosgds.object_namespace']==ns
    assert ec['daosgds.pool']=='discospool' and ec['daosgds.container']=='kvcache'
    assert ec['daosgds.transport']=='object'
    assert ec['daosgds.object_library']==str(c.ROOT/'libdaosgdr.so')
    return [dict(namespace=ns,pool='discospool',container='kvcache',
        sources=[str((case/'config.yaml').relative_to(c.ROOT))],experiments=[root.name])]


if __name__=='__main__':
    c.eligible=eligible
    c.classify=classify
    c.main()
