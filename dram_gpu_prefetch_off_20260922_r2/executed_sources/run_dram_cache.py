#!/usr/bin/env python3
"""Opt-in CPU cache retention alongside the existing async DAOS GPU backend.

No installed/package/source config edits. A fresh effective YAML is written for
each launch. Default is OFF. Stop the process and use run_vllm.sh to roll back.
This reuses existing CPU MemoryObjs; it is NOT a new dual GPU DMA implementation.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import yaml

ROOT = Path(__file__).resolve().parent


def build_profile(base, dram='off', cpu_gb=None, gpu_prefetch='off', prefetch_gb=None):
    if not isinstance(base, dict):
        raise ValueError('LMCache YAML must be a mapping')
    if dram not in ('on', 'off'):
        raise ValueError('dram must be on or off')
    if gpu_prefetch not in ('on', 'off'):
        raise ValueError('gpu-prefetch must be on or off')
    if gpu_prefetch == 'on' and dram != 'on':
        raise ValueError('gpu-prefetch on requires dram on')
    if prefetch_gb is not None and (not math.isfinite(prefetch_gb) or prefetch_gb <= 0):
        raise ValueError('prefetch-gb must be finite and positive')
    if cpu_gb is not None and (not math.isfinite(cpu_gb) or cpu_gb <= 0):
        raise ValueError('cpu-gb must be finite and positive')
    cfg = copy.deepcopy(base)
    # Allow a generated ON YAML to be switched back to the original backend.
    ec = cfg.get('extra_config', {})
    if ec.get('storage_plugin.daosgds.module_path') == 'lmcache_daos.dram_prefetch_backend':
        ec['storage_plugin.daosgds.module_path'] = 'lmcache_daos.gds_backend'
        ec['storage_plugin.daosgds.class_name'] = 'DaosGdsBackend'
    ec.pop('daosgds.cpu_prefetch_gpu_gb', None)
    cfg['local_cpu'] = dram == 'on'
    if cpu_gb is not None:
        cfg['max_local_cpu_size'] = cpu_gb
    if dram == 'on':
        ec = cfg.get('extra_config', {})
        if 'daosgds' not in (cfg.get('storage_plugins') or []):
            raise ValueError('ON profile requires the daosgds storage plugin')
        if ec.get('storage_plugin.daosgds.module_path') != 'lmcache_daos.gds_backend':
            raise ValueError('ON profile is validated only with lmcache_daos.gds_backend')
        if ec.get('storage_plugin.daosgds.class_name') != 'DaosGdsBackend':
            raise ValueError('ON profile requires DaosGdsBackend')
        if ec.get('daosgds.store', True) is not True:
            raise ValueError('ON profile must keep DAOS storage enabled')
        if cfg.get('enable_async_loading') is not True or cfg.get('use_layerwise', False):
            raise ValueError('ON profile currently supports async, non-layerwise loading only')
        if cfg.get('store_location') is not None:
            raise ValueError('store_location must be unset so both tiers receive stores')
    if gpu_prefetch == 'on':
        capacity = float(ec.get('daosgds.gpu_buffer_gb', 6))
        limit = capacity / 2 if prefetch_gb is None else prefetch_gb
        if not math.isfinite(limit) or not 0 < limit <= capacity:
            raise ValueError('prefetch-gb must be positive and <= GPU staging capacity')
        ec['storage_plugin.daosgds.module_path'] = 'lmcache_daos.dram_prefetch_backend'
        ec['storage_plugin.daosgds.class_name'] = 'DaosDramPrefetchBackend'
        ec['daosgds.cpu_prefetch_gpu_gb'] = limit
    return cfg


def profile_environment(env, config_path, cfg):
    env = dict(env)
    env['LMCACHE_CONFIG_FILE'] = str(config_path)
    # LMCache applies environment overrides after parsing YAML. Ensure stale
    # shell settings cannot silently invert the requested ON/OFF condition.
    env['LMCACHE_LOCAL_CPU'] = 'True' if cfg['local_cpu'] else 'False'
    env.pop('LMCACHE_MAX_LOCAL_CPU_SIZE', None)
    if 'max_local_cpu_size' in cfg:
        env['LMCACHE_MAX_LOCAL_CPU_SIZE'] = str(cfg['max_local_cpu_size'])
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dram', choices=['off', 'on'], default='off')
    p.add_argument('--gpu-prefetch', choices=['off', 'on'], default='off')
    p.add_argument('--prefetch-gb', type=float, help='Shared-pool CPU prefetch watermark; default = half GPU pool')
    p.add_argument('--cpu-gb', type=float, help='CPU pool GiB in either mode; omitted = base/default')
    p.add_argument('--config', type=Path, default=Path(os.environ.get(
        'LMCACHE_CONFIG_FILE', str(ROOT/'lmcache_config_daosgds_unified.yaml'))))
    p.add_argument('--state-dir', type=Path, help='New directory for effective config and manifest')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command and not a.dry_run:
        p.error('provide a command after -- (use venv/bin/python3, not copied vllm shebang)')
    source = a.config.resolve()
    try:
        cfg = build_profile(yaml.safe_load(source.read_text()), a.dram, a.cpu_gb,
                            a.gpu_prefetch, a.prefetch_gb)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        p.error(str(exc))
    if a.state_dir:
        state = a.state_dir.resolve()
        state.mkdir(parents=True, exist_ok=False)
    else:
        state = Path(tempfile.mkdtemp(prefix='dram_profile_', dir=ROOT))
    effective = state/'lmcache_effective.yaml'
    effective.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = profile_environment(os.environ, effective, cfg)
    manifest = dict(dram=a.dram, gpu_prefetch=a.gpu_prefetch,
                    source_config=str(source), effective_config=str(effective),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    effective_sha256=hashlib.sha256(effective.read_bytes()).hexdigest(),
                    command=[str(ROOT/'run_vllm.sh'), *command],
                    transport=env.get('DAOSGDS_TRANSPORT', 'dfs'),
                    local_cpu=cfg['local_cpu'], max_local_cpu_size=cfg.get('max_local_cpu_size'),
                    notes=['CPU cache retention uses native LocalCPUBackend references.',
                           'DAOS put remains asynchronous; CPU hit alone does not prove DAOS commit.',
                           'New process required to change modes; DRAM cache is process-local.',
                           'Existing source config and run_vllm.sh are unchanged.'])
    (state/'launch.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'DRAM cache {a.dram.upper()}, GPU prefetch {a.gpu_prefetch.upper()}: {effective}', flush=True)
    if not a.dry_run:
        launcher = str(ROOT/'run_vllm.sh')
        os.execvpe(launcher, [launcher, *command], env)


if __name__ == '__main__':
    main()
