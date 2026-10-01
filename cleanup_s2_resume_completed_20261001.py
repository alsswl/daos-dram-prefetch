"""Exact namespace cleanup of completed s2 split experiment; retain evidence."""
import json
import yaml
import cleanup_experiment_cache as c
from cleanup_window1024_20261001 import classify


def eligible():
    root=c.ROOT/'realqa_q4_64k_window1024_d256_s2_u0835_resume_fix_20261001'
    case=root/'c8_s10_window1024'
    read=lambda p:json.loads(p.read_text())
    assert read(root/'status.json')['status']=='completed'
    assert read(case/'status.json')['status']=='completed'
    assert read(root/'summary.json')['by_turn'][0]['requests']==480
    assert read(case/'window_policy_check.json')['passed']
    ec=yaml.safe_load((case/'config.yaml').read_text())['extra_config']
    ns='minji-windowed-cca64a2443b4418ea1049c870b2aa07d:'
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
