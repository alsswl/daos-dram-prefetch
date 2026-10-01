"""Exact cleanup of the interrupted s10 run; never label it completed."""
import json
import yaml
import cleanup_experiment_cache as c
from cleanup_window1024_20261001 import classify


def eligible():
    root=c.ROOT/'realqa_q4_64k_unwindowed_d256_s10_20261001'
    case=root/'c8_s10_window0'
    state=json.loads((root/'status.json').read_text())
    assert state['status']=='interrupted' and state['completed']==26
    ec=yaml.safe_load((case/'config.yaml').read_text())['extra_config']
    ns='minji-windowed-fe758b3072ce4cfea7ea29c4ab7025a9:'
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
