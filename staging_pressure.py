#!/usr/bin/env python3
"""One vLLM, persisted cache hits, concurrency sweep with opt-in staging tracing.

This is a fixed-token diagnostic, NOT full agentic DiscoveryBench. Creates a
private cache namespace, never deletes a container, and stops only its own server.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
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


@contextmanager
def server(a, config, folder, phase):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', a.port))
    env = common.environment(config, a.mode)
    # No tracing during fill; measured read-only phase has no save allocations.
    env.pop('DAOS_GDS_STAGING_TRACE', None)
    command = common.server_command(a)
    if phase == 'measure':
        env['DAOS_GDS_STAGING_TRACE'] = str(folder/'trace')
        command[-1] = json.dumps(dict(kv_connector='LMCacheConnectorV1', kv_role='kv_consumer'))
    common.dump(folder/f'{phase}_command.json', command)
    with (folder/f'{phase}_server.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        common.dump(folder/f'{phase}_pid.json', {'pid': proc.pid})
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{a.port}',
                              timeout=a.request_timeout, trust_env=False) as client:
                deadline = time.monotonic() + a.startup_timeout
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f'{phase} server exited; see server log')
                    try:
                        if client.get('/health', timeout=2).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Server startup timeout')
                    time.sleep(1)
                common.dump(folder/f'{phase}_native_maps.json', common.native_maps(proc.pid))
                print(f'{phase}: server ready (PID {proc.pid})', flush=True)
                yield client
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


def trace_summary(events):
    acquired = [e['wait_ms'] for e in events if e['event'] == 'serializer_acquired']
    ready_wait = [e['ready_wait_ms'] for e in events if e['event'] == 'retrieve_start'
                  and e.get('ready_wait_ms') is not None]
    active = peak_active = 0
    for e in events:
        if e['event'] == 'prefetch_start':
            active += 1
            peak_active = max(active, peak_active)
        elif e['event'] in ('prefetch_ready', 'prefetch_error'):
            active -= 1
    return dict(
        peak_used_gib=max((e['used_bytes'] for e in events), default=0) / 2**30,
        peak_ready_gib=max((e['ready_bytes'] for e in events), default=0) / 2**30,
        alloc_failures=sum(e['event'] in ('allocate', 'batched_allocate') and e['failed'] for e in events),
        partial_prefetches=sum(e['event'] == 'prefetch_ready' and e['chunks'] < e['requested_chunks'] for e in events),
        prefetches=sum(e['event'] == 'prefetch_start' for e in events),
        peak_active_prefetches=peak_active,
        serializer_wait_max_ms=max(acquired, default=0),
        serializer_wait_p95_ms=common.percentile(acquired, .95) if acquired else None,
        ready_wait_max_ms=max(ready_wait, default=0),
        ready_wait_p95_ms=common.percentile(ready_wait, .95) if ready_wait else None,
        final_used_bytes=events[-1]['used_bytes'] if events else None,
        trace_events=len(events))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['dfs', 'object'], default='object')
    p.add_argument('--context-length', type=int, default=8192)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--gpu-buffer-gb', type=float, default=10)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--port', type=int, default=8017)
    a = p.parse_args()
    if min(a.context_length, a.chunk_size, a.gpu_buffer_gb, a.repeats) <= 0:
        p.error('sizes and repeat count must be positive')
    if a.context_length <= a.chunk_size:
        p.error('context-length must exceed chunk-size')
    a.model = 'Qwen/Qwen3-14B'
    a.max_tokens = 64
    a.max_model_len = max(16384, a.context_length + a.max_tokens)
    a.context_tokens = [a.context_length] * 8
    a.startup_timeout, a.request_timeout, a.drain_timeout = 360, 180, 180
    a.pool, a.container = 'discospool', 'kvcache'
    a.io_workers = a.meta_workers = 16
    folder = a.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    ns = 'minji-staging-' + uuid.uuid4().hex
    cfg = yaml.safe_load((ROOT/'lmcache_config_daosgds_unified.yaml').read_text())
    cfg.update(chunk_size=a.chunk_size, enable_async_loading=True, local_cpu=False)
    cfg['extra_config'].update({
        'daosgds.transport': a.mode, 'daosgds.root': '/' + ns,
        'daosgds.object_namespace': ns + ':', 'daosgds.dfs_oclass': common.OC_SX,
        'daosgds.gpu_buffer_gb': a.gpu_buffer_gb,
        'daosgds.io_workers': a.io_workers, 'daosgds.meta_workers': a.meta_workers})
    fill_config, read_config = folder/'fill.yaml', folder/'read.yaml'
    fill_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cfg['extra_config']['daosgds.store'] = False
    read_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    files = ['staging_pressure.py', 'lmcache_daos/staging_trace.py',
             'lmcache_daos/gds_backend.py', 'compare_e2e.py', 'run_vllm.sh', 'libdaosgdr.so']
    common.dump(folder/'manifest.json', {
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'namespace': ns, 'measurement_role': 'kv_consumer',
        'source_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},
        'notes': ['Synthetic fixed-token cache-hit diagnostic, not DiscoveryBench.',
                  'Tracing adds overhead; latency is diagnostic, not an uninstrumented benchmark.',
                  'Cache fill then process restart; storage server cache is not flushed.',
                  'Eight distinct equal-length prompts at every concurrency level.',
                  'No cache/container deletion; generated experiment namespace is retained.']})
    if a.dry_run:
        print(f'Dry run: {folder}', flush=True)
        return
    try:
        check = subprocess.run([str(ROOT/'run_vllm.sh'), str(common.NATIVE/'bin/daos'),
                                'container', 'get-prop', a.pool, a.container],
                               env=common.environment(fill_config, a.mode),
                               capture_output=True, text=True, timeout=60)
        (folder/'container_properties.txt').write_text(check.stdout + check.stderr)
        check.check_returncode()
        prompts, warm, _ = common.workload(a)
        common.dump(folder/'prompts.json', {'prompts': prompts, 'warm': warm})
        with server(a, fill_config, folder, 'fill') as client:
            rows = []
            for prompt in [warm] + prompts:
                rows.append(common.request(client, prompt, a))
                print(f'fill: prompt {prompt["id"]} persisted request completed', flush=True)
            common.dump(folder/'fill_requests.json', rows)
            common.await_hit(client, [warm] + prompts, a)
            print('fill: all prefixes verified readable', flush=True)
        passes = []
        with server(a, read_config, folder, 'measure') as client:
            common.await_hit(client, [warm] + prompts, a)
            print('measure: restart cache hits verified; sweep starts', flush=True)
            for repeat in range(1, a.repeats + 1):
                order = [1, 2, 4, 8] if repeat % 2 else [8, 4, 2, 1]
                for concurrency in order:
                    start_ns = time.time_ns()
                    with ThreadPoolExecutor(max_workers=concurrency) as workers:
                        rows = list(workers.map(lambda pr: common.request(client, pr, a), prompts))
                    end_ns = time.time_ns()
                    item = dict(repeat=repeat, concurrency=concurrency,
                                start_ns=start_ns, end_ns=end_ns,
                                ttft_mean_ms=statistics.mean(r['ttft_ms'] for r in rows),
                                ttft_p95_ms=common.percentile([r['ttft_ms'] for r in rows], .95),
                                e2e_mean_ms=statistics.mean(r['e2e_ms'] for r in rows),
                                all_hit=all(r['cached_tokens'] >= common.expected_hit(pr, a.chunk_size)
                                            for r, pr in zip(rows, prompts)), rows=rows)
                    passes.append(item)
                    common.dump(folder/'passes.json', passes)
                    print(f'repeat {repeat} concurrency {concurrency}: TTFT {item["ttft_mean_ms"]:.1f}ms '
                          f'all_hit={item["all_hit"]}', flush=True)
        events = sorted((json.loads(line) for path in folder.glob('trace.*.jsonl')
                         for line in path.read_text().splitlines()), key=lambda e: e['time_ns'])
        if not any(e['event'] == 'prefetch_ready' for e in events):
            raise RuntimeError('No prefetch trace; diagnostic is not valid')
        summaries = []
        for item in passes:
            sub = [e for e in events if item['start_ns'] <= e['time_ns'] <= item['end_ns']]
            summaries.append({k: v for k, v in item.items() if k != 'rows'} | trace_summary(sub))
        common.dump(folder/'summary.json', summaries)
        common.dump(folder/'trace_totals.json', trace_summary(events))
        common.dump(folder/'status.json', {'status': 'completed'})
        print(f'Completed: {folder}', flush=True)
    except BaseException as exc:
        common.dump(folder/'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
