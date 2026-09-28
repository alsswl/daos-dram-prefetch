#!/usr/bin/env python3
"""Paired DRAM retention comparison from fresh KV-cache namespaces/processes.

Default: every measured prompt is a cold KV miss, without application warmup.
Optional reuse passes are explicitly labelled warm and never pooled with cold.
No shared cache deletion, OS cache dropping, or DAOS server restart.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
import uuid

import yaml

import compare_e2e as common
from dram_cache_bench import sample, server
from run_dram_cache import build_profile

ROOT = common.ROOT
CONDITIONS = {
    'daos_only': ('off', 'off'),
    'dram': ('on', 'off'),
    'dram_prefetch': ('on', 'on'),
}


def schedule(repeats):
    names = list(CONDITIONS)
    for repeat in range(1, repeats + 1):
        for concurrency in ([1, 4] if repeat % 2 else [4, 1]):
            offset = (repeat - 1 + (1 if concurrency == 4 else 0)) % 3
            for condition in names[offset:] + names[:offset]:
                yield repeat, concurrency, condition


def validate_cold(result):
    if any(row['cached_tokens'] != 0 for row in result['rows']):
        raise RuntimeError('Cold sample has cache hits')
    if any(result[field] for field in ('daos_prefetch_calls', 'cpu_gpu_prefetch_calls',
                                      'cpu_gpu_fallback_calls')):
        raise RuntimeError('Cold sample unexpectedly used a cache retrieval path')


def aggregate(cases):
    grouped = {}
    for case in cases:
        for result in case['samples']:
            key = (case['condition'], case['concurrency'], result['tag'])
            grouped.setdefault(key, []).append(result)
    out = []
    for (condition, concurrency, phase), batches in sorted(grouped.items()):
        rows = [row for batch in batches for row in batch['rows']]
        out.append(dict(condition=condition, concurrency=concurrency, phase=phase,
            fresh_processes=len(batches), requests=len(rows),
            mean_ttft_ms=statistics.mean(row['ttft_ms'] for row in rows),
            mean_e2e_ms=statistics.mean(row['e2e_ms'] for row in rows),
            p95_ttft_ms=common.percentile([row['ttft_ms'] for row in rows], .95),
            per_process_mean_ttft_ms=[batch['ttft_mean_ms'] for batch in batches],
            first_submitted_request_mean_ttft_ms=statistics.mean(
                batch['rows'][0]['ttft_ms'] for batch in batches),
            cached_tokens=sorted(set(row['cached_tokens'] for row in rows))))
    return out


def readiness(client, prompts, a):
    """Record every unmeasured readiness request; do not hide warmup traffic."""
    rows = []
    deadline = time.monotonic() + a.drain_timeout
    for prompt in prompts:
        while True:
            row = common.request(client, prompt, a, max_tokens=1)
            rows.append(row)
            if row['cached_tokens'] >= common.expected_hit(prompt, a.chunk_size):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Readiness timeout')
            time.sleep(.2)
    return rows


def gpu_state():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu,'
        'temperature.gpu,clocks.current.sm,power.draw', '--format=csv'],
        capture_output=True, text=True, check=True)
    return result.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--reuse-passes', type=int, default=0)
    p.add_argument('--transport', choices=['dfs', 'object'], default='object')
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.repeats < 1 or a.reuse_passes < 0:
        p.error('repeats must be positive and reuse-passes nonnegative')
    a.profile, a.cpu_gb = True, 4
    a.model, a.max_tokens, a.max_model_len = 'Qwen/Qwen3-14B', 64, 8192
    a.context_tokens, a.chunk_size = [4096] * 4, 128
    a.startup_timeout, a.request_timeout, a.drain_timeout = 360, 180, 180
    folder = a.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source = ROOT/'lmcache_config_daosgds_unified.yaml'
    base = yaml.safe_load(source.read_text())
    base.update(chunk_size=128, local_cpu=False, max_local_cpu_size=4,
                enable_async_loading=True, use_layerwise=False, store_location=None)
    base['extra_config'].update({'daosgds.transport': a.transport,
        'daosgds.dfs_oclass': common.OC_SX, 'daosgds.gpu_buffer_gb': 10,
        'daosgds.io_workers': 16, 'daosgds.meta_workers': 16, 'daosgds.store': True})
    common.dump(folder/'plan.json', dict(args={k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()}, cases=list(schedule(a.repeats)), notes=[
        'Fresh vLLM process and UUID namespace for each repeat/concurrency/mode.',
        'No warmup inference before cold measurement. Model load is outside HTTP timings.',
        'Four distinct first-chunk hashes; all four cold requests must report zero cached tokens.',
        'At concurrency 1, only the first request is process-first; later distinct prompts are KV-cold.',
        'OS/model file caches, GPU clocks and DAOS server internal caches are not reset.',
        'DRAM off disables KV retention, not temporary host buffers or the CPU allocator.',
        'Cold latency includes inference and store submission, not proof of DAOS commit latency.',
        'Optional warm reuse passes are reported separately, never called cold.',
        'Readiness requests between cold and warm are recorded separately (max_tokens=1).',
        'Three conditions rotate order; each occupies each order position over three repeats.',
        'No DAOS deletion. CPU/GPU pool sizes remain equal across conditions.']))
    files = ['dram_cold_compare.py', 'dram_cache_bench.py', 'compare_e2e.py',
        'run_dram_cache.py', 'run_vllm.sh', 'lmcache_config_daosgds_unified.yaml',
        'lmcache_daos/gds_backend.py', 'lmcache_daos/dram_prefetch_backend.py',
        'lmcache_daos/object_binding.py', 'lmcache_daos/dfs_binding.py',
        'lmcache_daos/serde_v2.py', 'libdaosgdr.c', 'libdaosgdr.so']
    hashes = {}
    for name in files:
        target = folder/'executed_sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    common.dump(folder/'source_sha256.json', hashes)
    if a.dry_run:
        print(f'Dry run: {folder}')
        return
    cases, output_hashes = [], {}
    try:
        prompts, _, _ = common.workload(a)
        common.dump(folder/'prompts.json', prompts)
        for repeat, concurrency, condition in schedule(a.repeats):
            a.dram, a.gpu_prefetch = CONDITIONS[condition]
            case_dir = folder/f'r{repeat}_c{concurrency}_{condition}'
            case_dir.mkdir()
            cfg = copy.deepcopy(base)
            ns = 'minji-cold-' + uuid.uuid4().hex
            cfg['extra_config'].update({'daosgds.root': '/' + ns,
                                       'daosgds.object_namespace': ns + ':'})
            # Validate both profiles before launching the model.
            build_profile(cfg, a.dram, a.cpu_gb, a.gpu_prefetch)
            config = case_dir/'base.yaml'
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            case = dict(repeat=repeat, concurrency=concurrency, condition=condition, dram=a.dram,
                        gpu_prefetch=a.gpu_prefetch, namespace=ns, samples=[])
            common.dump(case_dir/'case.json', case)
            (case_dir/'gpu_before.csv').write_text(gpu_state())
            print(f'START {case_dir.name}', flush=True)
            with server(a, config, case_dir, 'main') as (client, log_path):
                cold = sample(client, log_path, prompts, a, case_dir, 'cold',
                              concurrency, False)
                validate_cold(cold)
                case['samples'].append(cold)
                if a.reuse_passes:
                    # Readiness requests are warm/unmeasured, documented separately.
                    common.dump(case_dir/'readiness.json', readiness(client, prompts, a))
                for reuse in range(1, a.reuse_passes + 1):
                    result = sample(client, log_path, prompts, a, case_dir,
                                    f'warm_{reuse}', concurrency, True)
                    if result['daos_prefetch_calls'] != (4 if a.dram == 'off' else 0):
                        raise RuntimeError('Unexpected warm read tier')
                    if a.gpu_prefetch == 'on' and (result['cpu_gpu_prefetch_calls'] != 4
                                                 or result['cpu_gpu_fallback_calls']):
                        raise RuntimeError('Expected GPU prefetch for all warm CPU hits')
                    case['samples'].append(result)
            (case_dir/'gpu_after.csv').write_text(gpu_state())
            log = log_path.read_text()
            if re.search(r'negative: -|Double free|Double release|GPU buffer full|'
                         r'DaosGdsBackend.*failed|Failed to create.*backend', log):
                raise RuntimeError(f'Backend/allocator failure in {log_path}')
            for result in case['samples']:
                for row in result['rows']:
                    key = (concurrency, result['tag'], row['request_id'])
                    output_hashes.setdefault(key, set()).add(row['output_token_sha256'])
            cases.append(case)
            common.dump(case_dir/'status.json', dict(status='completed'))
            common.dump(folder/'cases.json', cases)
            common.dump(folder/'summary.json', aggregate(cases))
            print(f'DONE {case_dir.name}', flush=True)
        differing = [str(key) for key, values in output_hashes.items() if len(values) > 1]
        common.dump(folder/'validation.json', dict(all_cold_requests_miss=True,
            output_token_hashes_match=not differing, differing_output_groups=differing))
        common.dump(folder/'status.json', dict(status='completed'))
    except BaseException as exc:
        common.dump(folder/'status.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
