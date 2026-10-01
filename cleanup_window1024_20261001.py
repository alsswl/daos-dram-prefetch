"""Audited exact-namespace cleanup after the completed 64K windowed run."""
import json
import yaml
import cleanup_experiment_cache as c


def eligible():
    root=c.ROOT/'realqa_q4_64k_window1024_d256_s8_20261001'
    case=root/'c8_s10_window1024'
    read=lambda p:json.loads(p.read_text())
    assert read(root/'status.json')['status']=='completed'
    assert read(case/'status.json')['status']=='completed'
    assert read(case/'window_policy_check.json')['passed']
    assert read(root/'summary.json')['by_turn'][0]['requests']==480
    ec=yaml.safe_load((case/'config.yaml').read_text())['extra_config']
    ns='minji-windowed-ffbfc515e6a643d2bc6218bbd9767b2a:'
    assert ec['daosgds.object_namespace']==ns
    assert ec['daosgds.pool']=='discospool' and ec['daosgds.container']=='kvcache'
    assert ec['daosgds.transport']=='object'
    assert ec['daosgds.object_library']==str(c.ROOT/'libdaosgdr.so')
    return [dict(namespace=ns,pool='discospool',container='kvcache',
        sources=[str((case/'config.yaml').relative_to(c.ROOT))],experiments=[root.name])]


def classify(keys,rows):
    assert len(rows)==1
    ns=rows[0]['namespace']
    prefix='/home/hf/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554@'
    targets=[k for k in keys if k.startswith(ns)]
    assert all(k[len(ns):].startswith(prefix) for k in targets)
    return targets,{ns:len(targets)}


if __name__=='__main__':
    c.eligible=eligible
    c.classify=classify
    c.main()
