#!/usr/bin/env python3
"""Isolated pre-change baseline / opt-in DRAM cache comparison.

Fixed prompts, real Qwen inference, new namespace, no cache deletion. Run each
condition in a fresh process. This is not the full DiscoveryBench workload.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import time
import uuid

import httpx
import yaml
import compare_e2e as common

ROOT = common.ROOT


def summarize(rows, seconds):
    return dict(requests=len(rows), wall_seconds=seconds,
                ttft_mean_ms=statistics.mean(r['ttft_ms'] for r in rows),
                ttft_p95_ms=common.percentile([r['ttft_ms'] for r in rows], .95),
                e2e_mean_ms=statistics.mean(r['e2e_ms'] for r in rows),
                requests_per_second=len(rows)/seconds,
                cached_tokens=sorted(set(r['cached_tokens'] for r in rows)))


def metric(text, name):
    return sum(float(line.split()[-1]) for line in text.splitlines()
               if line.startswith(name + '{') or line.startswith(name + ' '))


@contextmanager
def server(a, config, folder, phase, consumer=False):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', a.port))
    env = common.environment(config, a.transport)
    env.pop('DAOS_GDS_STAGING_TRACE', None)
    # The effective YAML is the authority for both conditions.
    for name in ('LMCACHE_LOCAL_CPU', 'LMCACHE_MAX_LOCAL_CPU_SIZE'):
        env.pop(name, None)
    command = common.server_command(a)
    if consumer:
        command[-1] = json.dumps(dict(kv_connector='LMCacheConnectorV1', kv_role='kv_consumer'))
    if a.profile:
        command = [str(ROOT/'venv/bin/python3'), str(ROOT/'run_dram_cache.py'),
                   '--dram', a.dram, '--cpu-gb', str(a.cpu_gb), '--config', str(config),
                   '--state-dir', str(folder/f'{phase}_profile'), '--', *command[1:]]
    common.dump(folder/f'{phase}_command.json', command)
    log_path = folder/f'{phase}_server.log'
    with log_path.open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        common.dump(folder/f'{phase}_pid.json', {'pid': proc.pid})
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{a.port}',
                              timeout=a.request_timeout, trust_env=False) as client:
                deadline = time.monotonic() + a.startup_timeout
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f'{phase} server exited; see {log_path}')
                    try:
                        if client.get('/health', timeout=2).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Server startup timeout')
                    time.sleep(1)
                common.dump(folder/f'{phase}_native_maps.json', common.native_maps(proc.pid))
                print(f'{phase}: ready PID={proc.pid}', flush=True)
                yield client, log_path
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


def sample(client, log_path, prompts, a, folder, tag, concurrency, require_hit):
    before = client.get('/metrics').text
    (folder/f'{tag}_metrics_before.txt').write_text(before)
    offset = log_path.stat().st_size
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda pr: common.request(client, pr, a), prompts))
    seconds = time.perf_counter() - start
    after = client.get('/metrics').text
    (folder/f'{tag}_metrics_after.txt').write_text(after)
    with log_path.open('rb') as f:
        f.seek(offset)
        text = f.read().decode(errors='replace')
    (folder/f'{tag}_server_slice.log').write_text(text)
    result = summarize(rows, seconds)
    result.update(tag=tag, concurrency=concurrency,
                  daos_prefetch_calls=text.count('DaosGdsBackend prefetch['),
                  cpu_hot_chunks=metric(after, 'lmcache:local_cpu_hot_cache_count'),
                  all_hit=all(r['cached_tokens'] >= common.expected_hit(pr, a.chunk_size)
                              for r, pr in zip(rows, prompts)), rows=rows)
    common.dump(folder/f'{tag}.json', result)
    if require_hit and not result['all_hit']:
        raise RuntimeError(f'{tag}: missing expected cached prefix')
    if not require_hit and any(r['cached_tokens'] for r in rows):
        raise RuntimeError('Fresh independent measurement prompts unexpectedly hit')
    if re.search(r'GPU buffer full|Double free|negative: -|DaosGdsBackend.*failed', text):
        raise RuntimeError(f'{tag}: allocator/backend error')
    print(f'{tag}: TTFT={result["ttft_mean_ms"]:.2f}ms, '
          f'DAOS reads={result["daos_prefetch_calls"]}, CPU entries={result["cpu_hot_chunks"]}', flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dram', choices=['off', 'on'], default='off')
    p.add_argument('--profile', action='store_true', help='Use optional launcher; false = original baseline')
    p.add_argument('--transport', choices=['dfs', 'object'], default='object')
    p.add_argument('--cpu-gb', type=float, default=4)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--skip-restart', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.dram == 'on' and not a.profile:
        p.error('--dram on requires --profile')
    if a.cpu_gb <= 0 or a.repeats < 1:
        p.error('cpu-gb and repeats must be positive')
    a.model, a.max_tokens, a.max_model_len = 'Qwen/Qwen3-14B', 64, 8192
    a.context_tokens, a.chunk_size = [4096]*4, 128
    a.startup_timeout, a.request_timeout, a.drain_timeout = 360, 180, 180
    folder = a.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    ns = 'minji-dram-' + uuid.uuid4().hex
    cfg = yaml.safe_load((ROOT/'lmcache_config_daosgds_unified.yaml').read_text())
    cfg.update(chunk_size=a.chunk_size, local_cpu=False, max_local_cpu_size=a.cpu_gb,
               enable_async_loading=True)
    cfg['extra_config'].update({'daosgds.transport': a.transport,
        'daosgds.root': '/'+ns, 'daosgds.object_namespace': ns+':',
        'daosgds.dfs_oclass': common.OC_SX, 'daosgds.gpu_buffer_gb': 10,
        'daosgds.io_workers': 16, 'daosgds.meta_workers': 16, 'daosgds.store': True})
    config = folder/'base.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    snapshot = folder/'executed_sources'
    snapshot.mkdir()
    files = ['dram_cache_bench.py', 'compare_e2e.py', 'run_vllm.sh',
             'lmcache_config_daosgds_unified.yaml', 'libdaosgdr.c', 'libdaosgdr.so',
             'lmcache_daos/gds_backend.py', 'lmcache_daos/object_binding.py',
             'lmcache_daos/dfs_binding.py', 'lmcache_daos/serde_v2.py']
    if a.profile:
        files.append('run_dram_cache.py')
    hashes = {}
    for name in files:
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    common.dump(folder/'manifest.json', {'args': {k: str(v) if isinstance(v, Path) else v
                                                for k, v in vars(a).items()},
        'namespace': ns, 'source_sha256': hashes,
        'notes': ['Synthetic cache-reuse workload, not full DiscoveryBench.',
                  'Both modes reserve equal CPU allocator capacity; only retention changes.',
                  'Distinct warmup prompt, fresh namespace, vLLM APC off.',
                  'Fill HTTP latency is not a DAOS commit-latency measurement.',
                  'No staging lifecycle trace; normal INFO logs in both modes.',
                  'Restart uses consumer-only mode: no new writes or CPU cache fill.']})
    if a.dry_run:
        print(f'Dry run: {folder}')
        return
    summaries = []
    try:
        prompts, warm, _ = common.workload(a)
        common.dump(folder/'prompts.json', dict(prompts=prompts, warm=warm))
        with server(a, config, folder, 'main') as (client, log_path):
            common.request(client, warm, a)
            common.await_hit(client, [warm], a)
            summaries.append(sample(client, log_path, prompts, a, folder, 'fill', 1, False))
            common.await_hit(client, prompts, a)
            for repeat in range(1, a.repeats+1):
                for concurrency in ([1, 4] if repeat % 2 else [4, 1]):
                    result = sample(client, log_path, prompts, a, folder,
                                    f'warm_r{repeat}_c{concurrency}', concurrency, True)
                    if result['daos_prefetch_calls'] != (0 if a.dram == 'on' else len(prompts)):
                        raise RuntimeError('Unexpected read tier; do not accept latency comparison')
                    summaries.append(result)
                    common.dump(folder/'summary.json', [{k:v for k,v in r.items() if k != 'rows'}
                                                       for r in summaries])
        if not a.skip_restart:
            with server(a, config, folder, 'restart', consumer=True) as (client, log_path):
                result = sample(client, log_path, prompts, a, folder, 'restart_hit', 1, True)
                if result['daos_prefetch_calls'] != len(prompts):
                    raise RuntimeError('Restart did not prove DAOS fallback')
                summaries.append(result)
        common.dump(folder/'summary.json', [{k:v for k,v in r.items() if k != 'rows'}
                                           for r in summaries])
        common.dump(folder/'status.json', {'status': 'completed'})
        print(f'Completed: {folder}', flush=True)
    except BaseException as exc:
        common.dump(folder/'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
