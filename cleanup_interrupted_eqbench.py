#!/usr/bin/env python3
"""Manifest-scoped cleanup for the audited interrupted EQ matrix case only.

Requires the experiment's vLLM PID to be gone. Preserves every unrelated key.
No model termination is performed by this script.
"""
from pathlib import Path
import json
import re
import yaml
import cleanup_experiment_cache as cleanup


def eligible():
    root=cleanup.ROOT/'eqbench_matrix_20260928_v2'
    case=root/'c8_d8_s4_wait'
    status=json.loads((root/'status.json').read_text())
    assert status['status']=='failed' and status['failure_kind']=='runner_disappeared'
    assert status['current']==case.name and case.name not in status['completed']
    assert any(s['name']==case.name for s in json.loads((root/'plan.json').read_text())['cases'])
    assert json.loads((case/'status.json').read_text())['status']=='failed'
    pid=json.loads((case/'server/pid.json').read_text())['pid']
    assert not Path(f'/proc/{pid}').exists(), 'Do not delete while the case server exists'
    config=case/'server/config.yaml';ec=yaml.safe_load(config.read_text())['extra_config']
    namespace=ec['daosgds.object_namespace']
    assert re.fullmatch(r'minji-eqmatrix-[0-9a-f]{32}:',namespace)
    assert namespace==json.loads((case/'identity.json').read_text())['namespace']
    assert ec['daosgds.pool']=='discospool' and ec['daosgds.container']=='kvcache'
    assert ec['daosgds.transport']=='object' and ec['daosgds.object_library']==str(cleanup.ROOT/'libdaosgdr.so')
    return [dict(namespace=namespace,case=str(case),config_sha256=cleanup.digest(config),
                 status_sha256=cleanup.digest(case/'status.json'),interrupted=True)]


if __name__=='__main__':
    cleanup.eligible=eligible
    cleanup.main()
