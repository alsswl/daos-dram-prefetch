"""Exact cleanup of failed s2/u0835 run; preserve all failure evidence."""
import json
import yaml
import cleanup_experiment_cache as c
from cleanup_window1024_20261001 import classify


def eligible():
    root=c.ROOT/'realqa_q4_64k_window1024_d256_s2_u0835_20261001'
    case=root/'c8_s10_window1024'
    read=lambda p:json.loads(p.read_text())
    assert read(root/'status.json')['status']=='failed'
    assert read(case/'partial_failure/failure_summary.json')['successful_calls']==406
    assert 'Num retrieved tokens: 62720, num expected tokens: 62848' in (case/'server.log').read_text()
    ec=yaml.safe_load((case/'config.yaml').read_text())['extra_config']
    ns='minji-windowed-603113062d5c49688d49b64f1e21bc48:'
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
