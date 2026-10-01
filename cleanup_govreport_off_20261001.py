"""Remove only the completed 20260928 GovReport OFF namespace, with audit."""
import json
import sys
import yaml
import cleanup_experiment_cache as cleanup


def eligible():
    case = cleanup.ROOT / 'govreport_prefetch_20260928_v2/off'
    assert json.loads((case / 'status.json').read_text())['status'] == 'completed'
    assert all(json.loads((case.parent / 'validation.json').read_text()).values())
    ec = yaml.safe_load((case / 'config.yaml').read_text())['extra_config']
    ns = 'minji-cold-warm-01bf0aea4d064979b4b969703fda2188:'
    assert ec['daosgds.object_namespace'] == ns
    assert ec['daosgds.pool'] == 'discospool'
    assert ec['daosgds.container'] == 'kvcache'
    assert ec['daosgds.transport'] == 'object'
    assert ec['daosgds.object_library'] == str(cleanup.ROOT / 'libdaosgdr.so')
    assert "LMCacheMetadata(model_name='Qwen/Qwen3-14B'" in (case / 'server.log').read_text()
    return [dict(namespace=ns, pool='discospool', container='kvcache',
                 sources=[str((case / 'config.yaml').relative_to(cleanup.ROOT))],
                 experiments=['govreport_prefetch_20260928_v2'])]


if __name__ == '__main__':
    cleanup.eligible = eligible
    cleanup.main()
