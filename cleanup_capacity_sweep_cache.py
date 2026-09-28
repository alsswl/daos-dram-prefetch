#!/usr/bin/env python3
"""Delete only the explicitly approved, completed 20260927 36-case sweep KV.

Reuse the manifest/check/punch-one-key/verify-preserved-set cleanup workflow.
This does not authorize any other experiment or delete any local result file.
"""
import json
import re

import yaml

import cleanup_experiment_cache as cleanup


def eligible():
    experiment = cleanup.ROOT / 'prefetch_capacity_sweep_20260927'
    plan = json.loads((experiment / 'plan.json').read_text())
    status = json.loads((experiment / 'status.json').read_text())
    names = [case['name'] for case in plan['cases']]
    expected = {f'c{c}_d{d}_s{s}_{m}' for c in (8, 16)
                for d in (8, 4, 2) for s in (8, 4)
                for m in ('off', 'wait', 'cancel')}
    assert len(names) == 36 and set(names) == expected
    assert status['status'] == 'completed' and set(status['completed']) == expected
    rows = []
    for name in sorted(names):
        folder = experiment / name
        case_status = json.loads((folder / 'status.json').read_text())
        assert case_status['status'] == 'completed'
        config = folder / 'config.yaml'
        ec = yaml.safe_load(config.read_text())['extra_config']
        assert ec['daosgds.pool'] == 'discospool'
        assert ec['daosgds.container'] == 'kvcache'
        assert ec['daosgds.transport'] == 'object'
        assert ec['daosgds.object_library'] == str(cleanup.ROOT / 'libdaosgdr.so')
        ns = ec['daosgds.object_namespace']
        assert re.fullmatch(r'minji-cold-warm-[0-9a-f]{32}:', ns)
        rows.append(dict(namespace=ns, case=name,
                         config=str(config.relative_to(cleanup.ROOT)),
                         config_sha256=cleanup.digest(config),
                         status_sha256=cleanup.digest(folder / 'status.json')))
    namespaces = {r['namespace'] for r in rows}
    assert len(namespaces) == 36
    # Protect the most recent worker experiment explicitly as well.
    for config in (cleanup.ROOT / 'prefetch_workers_20260928').glob('*/config.yaml'):
        ns = yaml.safe_load(config.read_text())['extra_config']['daosgds.object_namespace']
        assert ns not in namespaces
    return sorted(rows, key=lambda r: r['namespace'])


if __name__ == '__main__':
    cleanup.eligible = eligible
    cleanup.main()
