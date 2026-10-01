#!/usr/bin/env python3
"""Heavy fixed-token, changing DRAM/DAOS residency diagnostic (not DiscoveryBench)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
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


class ReadOnlyRequests:
    """Use LMCache's existing per-request skip_save; preserve resident CPU KV."""
    def __init__(self, client):
        self.client = client

    def stream(self, method, url, **kwargs):
        # LMCache's async lookup wire schema is dict[str, str]. The adapter
        # treats the nonempty string as the native per-request skip flag.
        kwargs['json'] = dict(kwargs['json'], kv_transfer_params={'lmcache.skip_save': 'true'})
        return self.client.stream(method, url, **kwargs)


def read_events(folder):
    events = []
    for path in folder.glob('trace.*.jsonl'):
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A live final line may not yet have been fully written.
    return sorted(events, key=lambda e: e['time_ns'])


def latest_sample(folder):
    for path in folder.glob('trace.*.jsonl'):
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('event') == 'occupancy_sample':
                return row
    return None


def wait_drained(folder, puts, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = latest_sample(folder)
        if row and row['daos_puts'] >= puts and row['used_bytes'] == 0:
            return row
        time.sleep(.2)
    raise RuntimeError(f'Could not verify DAOS puts / empty staging: {latest_sample(folder)}')


def batches(prompts):
    # Sequential fill leaves the most recent inputs resident in the 4 GiB CPU
    # cache. Actual prefix hits are traced and validated, not assumed from order.
    hot = prompts[-2:]
    cold = prompts[:8]
    return {
        'dram_heavy': hot * 8,
        'mixed': [p for i in range(8) for p in (hot[i % 2], cold[i])],
        'daos_heavy': cold * 2,
    }


def summarize(events, start_ns, end_ns, capacity):
    part = [e for e in events if start_ns <= e['time_ns'] <= end_ns]
    samples = [e for e in part if e['event'] == 'occupancy_sample']
    hits = {tier: sum(e['hit_chunks'] for e in part
                     if e['event'] == 'tier_lookup' and e['tier'] == tier)
            for tier in ('dram', 'daos')}
    total = sum(hits.values())
    cpu_gets = [e for e in part if e['event'] == 'cpu_get_ready']
    # Time-weight periodic samples within the HTTP batch window. Event peaks
    # capture shorter allocations; sampled averages are explicitly approximate.
    prior = next((e for e in reversed(events) if e['time_ns'] <= start_ns
                  and e['event'] == 'occupancy_sample'), None)
    used = prior['used_bytes'] if prior else 0
    prev = start_ns
    area = above5 = above9 = 0
    for row in samples + [dict(time_ns=end_ns, used_bytes=used)]:
        dt = row['time_ns'] - prev
        area += used * dt
        above5 += dt if used >= 5 * 2**30 else 0
        above9 += dt if used >= 9 * 2**30 else 0
        prev, used = row['time_ns'], row['used_bytes']
    duration = end_ns - start_ns
    return dict(hit_chunks=hits, dram_share_of_hit_chunks=hits['dram']/total if total else None,
        peak_used_gib=max((e['used_bytes'] for e in part), default=0)/2**30,
        mean_used_gib=area/duration/2**30,
        mean_free_gib=capacity-area/duration/2**30,
        fraction_time_ge_5gib=above5/duration, fraction_time_ge_9gib=above9/duration,
        peak_ready_gib=max((e['ready_bytes'] for e in part), default=0)/2**30,
        sampled_peak_cpu_ready_gib=max((e['cpu_ready_bytes'] for e in samples), default=0)/2**30,
        sampled_peak_daos_ready_gib=max((e['daos_ready_bytes'] for e in samples), default=0)/2**30,
        cpu_prefetch_batches=sum(e['staged_chunks'] > 0 for e in cpu_gets),
        cpu_unstaged_batches=sum(e['staged_chunks'] == 0 for e in cpu_gets),
        partial_daos_reads=sum(e['event'] == 'prefetch_ready' and e['chunks'] < e['requested_chunks']
                               for e in part),
        failed_alloc_events=sum(e['event'] in ('allocate', 'batched_allocate') and e['failed']
                                for e in part),
        sampling_points=len(samples), final_used_bytes=part[-1]['used_bytes'] if part else None)


@contextmanager
def server(a, config, folder):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', a.port))
    env = common.environment(config, 'object')
    for key in list(env):
        if key.startswith('LMCACHE_') and key != 'LMCACHE_LOG_LEVEL':
            env.pop(key)
    env['LMCACHE_CONFIG_FILE'] = str(config)
    env['DAOS_GDS_STAGING_TRACE'] = str(folder/'trace')
    env.pop('LMCACHE_FORCE_SKIP_SAVE', None)
    command = common.server_command(a) + ['--max-num-seqs', '16']
    common.dump(folder/'command.json', command)
    with (folder/'server.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        common.dump(folder/'pid.json', {'pid': proc.pid})
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{a.port}', timeout=180,
                              trust_env=False) as client:
                deadline = time.monotonic() + 420
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError('Server exited: inspect server.log')
                    try:
                        if client.get('/health', timeout=2).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Server startup timeout')
                    time.sleep(1)
                common.dump(folder/'native_maps.json', common.native_maps(proc.pid))
                print(f'{folder.name}: server ready PID={proc.pid}', flush=True)
                yield client
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.repeats < 1:
        p.error('repeats must be positive')
    a.model, a.max_tokens, a.max_model_len = 'Qwen/Qwen3-14B', 64, 16384
    a.context_tokens, a.chunk_size = [8192]*16, 128
    folder = a.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    ns = 'minji-mixed-pressure-' + uuid.uuid4().hex
    base = yaml.safe_load((ROOT/'lmcache_config_daosgds_unified.yaml').read_text())
    base.update(chunk_size=128, local_cpu=True, max_local_cpu_size=4,
                enable_async_loading=True, use_layerwise=False, store_location=None)
    base['extra_config'].update({'daosgds.transport': 'object', 'daosgds.gpu_buffer_gb': 10,
        'daosgds.cpu_prefetch_gpu_gb': 5, 'daosgds.io_workers': 16, 'daosgds.meta_workers': 16,
        'daosgds.store': True, 'storage_plugin.daosgds.module_path': 'lmcache_daos.staging_probe_backend'})
    hashes = {}
    for name in ['staging_mixed_pressure.py', 'compare_e2e.py', 'run_vllm.sh',
                 'lmcache_daos/staging_probe_backend.py', 'lmcache_daos/staging_trace.py',
                 'lmcache_daos/gds_backend.py', 'lmcache_daos/dram_prefetch_backend.py',
                 'lmcache_daos/object_binding.py', 'libdaosgdr.so', 'libdaosgdr.c']:
        dest = folder/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    common.dump(folder/'plan.json', dict(model=a.model, context_tokens=8192, inputs=16,
        output_tokens=64, cpu_gib=4, staging_gib=10, cpu_prefetch_watermark_gib=5,
        concurrency=[1, 4, 8, 16], repeats=a.repeats, source_sha256=hashes,
        notes=['Instrumented synthetic occupancy diagnostic, not DiscoveryBench or a definitive speed benchmark.',
               'Two fresh processes/namespaces (OFF then ON); repeats within a process are not independent trials.',
               '16 sequential fills; last two inputs selected as DRAM-hot. All actual tier hits are traced.',
               'Each batch has 16 requests: hot repeats 2 inputs, cold repeats 8 inputs, mixed alternates both.',
               'Per-request lmcache.skip_save during probes/measurements freezes cache contents.',
               'No eviction/admission policy changes. CPU prefetch uses existing 5 GiB shared watermark.',
               '5ms nominal occupancy sampling plus allocation/free event tracing, nonzero overhead.',
               'OS/DAOS internal caches are not flushed. New DAOS keys retained, no shared deletion.']))
    if a.dry_run:
        print(folder, flush=True)
        return
    prompts, _, _ = common.workload(a)
    common.dump(folder/'prompts.json', prompts)
    all_results = []
    try:
        for enabled in (False, True):
            case = folder/('prefetch_on' if enabled else 'prefetch_off')
            case.mkdir()
            cfg = json.loads(json.dumps(base))
            cfg['extra_config'].update({'daosgds.object_namespace': ns+case.name+':',
                'daosgds.root': '/'+ns+case.name,
                'storage_plugin.daosgds.class_name': 'ProbePrefetchBackend' if enabled else 'ProbeBackend'})
            config = case/'config.yaml'
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            results = []
            with server(a, config, case) as client:
                fill = []
                for prompt in prompts:
                    fill.append(common.request(client, prompt, a))
                    common.dump(case/'fill.json', fill)
                    if fill[-1]['cached_tokens']:
                        raise RuntimeError('Fresh fill unexpectedly hit')
                    print(f'{case.name}: fill {prompt["id"]}/16', flush=True)
                drained = wait_drained(case, 1024)
                common.dump(case/'after_fill.json', drained)
                readonly = ReadOnlyRequests(client)
                probes = []
                for prompt in prompts:
                    start = time.time_ns()
                    row = common.request(readonly, prompt, a, max_tokens=1)
                    probes.append(dict(start_ns=start, end_ns=time.time_ns(), **row))
                    common.dump(case/'probes.json', probes)
                    if row['cached_tokens'] != 8191:
                        raise RuntimeError(f'Probe {prompt["id"]} unexpectedly missed: {row["cached_tokens"]}')
                common.dump(case/'probes.json', probes)
                if any(r['cached_tokens'] != 8191 for r in probes):
                    raise RuntimeError('Not all prefilled DAOS prefixes readable')
                probe_events = read_events(case)
                for probe in probes:
                    hits = summarize(probe_events, probe['start_ns'], probe['end_ns'], 10)['hit_chunks']
                    probe['hit_chunks'] = hits
                common.dump(case/'probes.json', probes)
                if any(r['hit_chunks']['dram'] != 64 for r in probes[-2:]):
                    raise RuntimeError('Last two inputs not fully DRAM resident')
                if any(r['hit_chunks']['daos'] != 64 for r in probes[:8]):
                    raise RuntimeError('First eight inputs not fully DAOS-only')
                for repeat in range(1, a.repeats+1):
                    order = [1,4,8,16] if repeat % 2 else [16,8,4,1]
                    for concurrency in order:
                        workloads = batches(prompts)
                        names = list(workloads) if repeat % 2 else list(reversed(workloads))
                        for name in names:
                            before = wait_drained(case, 1024)
                            start = time.time_ns()
                            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                                rows = list(pool.map(lambda pr: common.request(readonly, pr, a), workloads[name]))
                            end = time.time_ns()
                            result = dict(condition=case.name, repeat=repeat, concurrency=concurrency,
                                workload=name, start_ns=start, end_ns=end, rows=rows,
                                mean_ttft_ms=statistics.mean(r['ttft_ms'] for r in rows),
                                mean_e2e_ms=statistics.mean(r['e2e_ms'] for r in rows),
                                wall_seconds=(end-start)/1e9,
                                all_hit=all(r['cached_tokens'] == 8191 for r in rows))
                            results.append(result)
                            common.dump(case/'passes.json', results)
                            after = wait_drained(case, 1024)
                            if after['daos_puts'] != before['daos_puts'] or after['cpu_hot_chunks'] != before['cpu_hot_chunks']:
                                raise RuntimeError('Read-only measurement changed stored cache')
                            print(f'{case.name} r{repeat} c{concurrency} {name}: '
                                  f'TTFT={result["mean_ttft_ms"]:.1f}ms all_hit={result["all_hit"]}', flush=True)
                common.dump(case/'after_measure.json', wait_drained(case, 1024))
            events = read_events(case)
            for result in results:
                stats = summarize(events, result['start_ns'], result['end_ns'], 10)
                result.update(stats)
                expected = {'dram_heavy': 1024, 'mixed': 512, 'daos_heavy': 0}[result['workload']]
                result['tier_mix_valid'] = stats['hit_chunks'] == {'dram': expected, 'daos': 1024-expected}
            common.dump(case/'passes.json', results)
            all_results.extend({k:v for k,v in row.items() if k != 'rows'} for row in results)
            common.dump(folder/'summary.json', all_results)
        common.dump(folder/'status.json', dict(status='completed',
            all_expected_hits=all(r['all_hit'] for r in all_results),
            all_tier_mixes_valid=all(r['tier_mix_valid'] for r in all_results)))
        print(f'Completed {folder}', flush=True)
    except BaseException as exc:
        common.dump(folder/'status.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
