#!/usr/bin/env python3
"""Fixed-count recorded DiscoveryBench prompt replay; no Python tool execution.

Each case starts a fresh vLLM/DRAM cache and a private DAOS key namespace.
Only the processes launched by staging_mixed_pressure.server are stopped.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import threading
import time
import uuid

import yaml

import compare_e2e as common
from staging_mixed_pressure import latest_sample, server

ROOT = common.ROOT
DEFAULT_SOURCE = ROOT/'discovery_async_dram_prefetch_20260926/prefetch_async_dram_on'


def extract_requests(source, count):
    records = []
    for path in sorted((source/'agents').glob('job_*/llm_calls.json')):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for call_id, call in json.loads(path.read_text()).items():
            if 'error' in call or 'end_ns' not in call:
                continue
            messages = call.get('messages')
            if not (isinstance(messages, list) and len(messages) == 1
                    and len(messages[0]) == 1 and isinstance(messages[0][0], str)):
                raise ValueError(f'Cannot reconstruct a single HumanMessage: {path}:{call_id}')
            prompt = messages[0][0]
            records.append(dict(source_file=str(path), source_sha256=digest,
                source_call_id=call_id, source_start_ns=call['start_ns'],
                prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
    records.sort(key=lambda r: (r['source_start_ns'], r['source_file'], r['source_call_id']))
    if len(records) < count:
        raise ValueError(f'Only {len(records)} recorded successful calls, need {count}')
    return [dict(index=i, **r) for i, r in enumerate(records[:count])]


def make_config(enabled, namespace, cpu_gib=8, staging_gib=None):
    cfg = yaml.safe_load((ROOT/'lmcache_config_daosgds_async_dram.yaml').read_text())
    cfg.update(max_local_cpu_size=cpu_gib)
    cfg['extra_config'].update({
        'storage_plugin.daosgds.module_path': 'lmcache_daos.store_probe_backend',
        'storage_plugin.daosgds.class_name': 'StoreProbeAsyncDramBackend',
        'daosgds.dram_prefetch': enabled, 'daosgds.dram_prefetch_policy': 'capacity',
        'daosgds.probe_interval_ms': 20,
        'daosgds.object_namespace': namespace+':', 'daosgds.root': '/'+namespace})
    if staging_gib is not None:
        if not math.isfinite(staging_gib) or staging_gib <= 0:
            raise ValueError('staging_gib must be finite and positive')
        cfg['extra_config']['daosgds.gpu_buffer_gb'] = staging_gib
    return cfg


def replay_one(client, record, barrier, *, max_tokens=2048, stop=('\nObservation:',)):
    barrier.wait(timeout=30)
    started = time.perf_counter()
    row = dict(index=record['index'], prompt_sha256=record['prompt_sha256'],
               start_ns=time.time_ns())
    parts, usage, finish, first = [], None, None, None
    try:
        with client.stream('POST', '/v1/chat/completions', json={
            'model': 'comparison-model', 'messages': [{'role': 'user', 'content': record['prompt']}],
            'temperature': 0, 'seed': 0, 'max_tokens': max_tokens,
            'stop': list(stop),
            'chat_template_kwargs': {'enable_thinking': False},
            'stream': True, 'stream_options': {'include_usage': True},
        }) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith('data: '):
                    continue
                data = line[6:]
                if data == '[DONE]':
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                row['server_request_id'] = event.get('id', row.get('server_request_id'))
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    finish = choice.get('finish_reason') or finish
                    text = choice.get('delta', {}).get('content') or ''
                    if text:
                        if first is None:
                            first = time.perf_counter()
                        parts.append(text)
        if not usage or first is None or finish is None:
            raise RuntimeError('Incomplete stream: text, usage or finish_reason missing')
        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        if cached is None:
            raise RuntimeError('Missing actual cached_tokens usage')
        row.update(ttft_ms=(first-started)*1000, usage=usage,
                   prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'],
                   cached_tokens=cached, finish_reason=finish,
                   output_sha256=hashlib.sha256(''.join(parts).encode()).hexdigest(),
                   output=''.join(parts))
    except Exception as exc:
        row['error'] = repr(exc)
    row.update(end_ns=time.time_ns(), elapsed_seconds=time.perf_counter()-started)
    return row


def await_empty(case):
    deadline = time.monotonic()+20
    while time.monotonic() < deadline:
        sample = latest_sample(case)
        if sample:
            if sample['used_bytes'] or sample['cpu_hot_bytes'] or sample['daos_puts']:
                raise RuntimeError(f'Case did not start cold: {sample}')
            return sample
        time.sleep(.1)
    raise RuntimeError('No initial occupancy sample')


def run_case(a, case, records, concurrency, enabled):
    namespace = 'minji-fixed-replay-'+uuid.uuid4().hex
    config = make_config(enabled, namespace, cpu_gib=a.cpu_gb,
                         staging_gib=getattr(a, 'staging_gib', None))
    if getattr(a, 'capacity_failure_probe', False):
        config['extra_config'].update({
            'storage_plugin.daosgds.module_path': 'lmcache_daos.capacity_probe_backend',
            'storage_plugin.daosgds.class_name': 'CapacityProbeBackend'})
    (case/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    calls, waves = [], []
    with server(a, case/'config.yaml', case) as client:
        common.dump(case/'initial_sample.json', await_empty(case))
        phase = dict(index=0, concurrency=concurrency, start_ns=time.time_ns())
        common.dump(case/'phases.json', [phase])
        offset = (case/'server.log').stat().st_size
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for start in range(0, len(records), concurrency):
                group = records[start:start+concurrency]
                barrier = threading.Barrier(len(group))
                wave = dict(index=len(waves), first_request=start, start_ns=time.time_ns())
                futures = [pool.submit(replay_one, client, r, barrier) for r in group]
                rows = [future.result() for future in futures]
                calls.extend(rows)
                wave.update(end_ns=time.time_ns(), requests=len(rows))
                waves.append(wave)
                common.dump(case/'replay_calls.json', calls)
                common.dump(case/'waves.json', waves)
                sample = latest_sample(case) or {}
                print(f'{case.name} {len(calls)}/{len(records)} '
                      f'TTFT={statistics.mean([r["ttft_ms"] for r in rows if "ttft_ms" in r] or [0]):.1f}ms '
                      f'staging={sample.get("used_bytes",0)/2**30:.2f}GiB '
                      f'DRAM={sample.get("cpu_hot_bytes",0)/2**30:.2f}GiB', flush=True)
                if any('error' in r for r in rows):
                    raise RuntimeError('Replay request failed; saved all responses, no retry')
                with (case/'server.log').open('rb') as log:
                    log.seek(offset)
                    new_log = log.read().decode(errors='replace')
                    offset = log.tell()
                if any(p in new_log for p in ('Double free', 'Double release', 'negative: -',
                        'CUDA error: an illegal memory access', 'CUDA out of memory')):
                    raise RuntimeError('Critical memory error; stopped own workload')
        phase['end_ns'] = time.time_ns()
        common.dump(case/'phases.json', [phase])
        (case/'metrics.txt').write_text(client.get('/metrics').text)
        # Do not extend measured elapsed time with final store/mirror draining.
        deadline, stable = time.monotonic()+60, 0
        while time.monotonic() < deadline:
            sample = latest_sample(case) or {}
            if sample and sample['used_bytes'] == 0 and not (sample.get('dram_mirror') or {}).get('pending_bytes', 0):
                stable += 1
            else:
                stable = 0
            if stable >= 10:
                break
            time.sleep(.2)
        common.dump(case/'final_sample.json', sample)
        if stable < 10:
            raise RuntimeError('Staging failed to drain after completed requests')
    common.dump(case/'status.json', dict(status='completed', requests=len(calls)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--port', type=int, default=8017)
    parser.add_argument('--cpu-gb', type=float, default=8)
    parser.add_argument('--concurrency', type=int, nargs='+', choices=(4,8,16), default=[4,8,16])
    parser.add_argument('--dry-run', action='store_true')
    a = parser.parse_args()
    if a.count < 16 or a.count % 16:
        parser.error('count must be a positive multiple of 16')
    if not math.isfinite(a.cpu_gb) or not 0 < a.cpu_gb <= 64:
        parser.error('cpu-gb must be finite and in (0,64]')
    if len(set(a.concurrency)) != len(a.concurrency):
        parser.error('concurrency values must be unique')
    a.model, a.max_model_len = 'Qwen/Qwen3-14B', 32768
    records = extract_requests(a.source, a.count)
    folder = a.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    common.dump(folder/'requests.json', records)
    hashes = {}
    names = ['discovery_fixed_replay.py', 'analyze_discovery_staging.py',
             'report_fixed_replay.py', 'compare_e2e.py', 'staging_mixed_pressure.py', 'run_vllm.sh',
             'lmcache_config_daosgds_async_dram.yaml', 'libdaosgdr.so', 'libdaosgdr.c']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'lmcache_daos').glob('*.py'))]
    for name in names:
        dst = folder/'executed_sources'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    cases = [dict(concurrency=c, prefetch=on, name=f'c{c:02d}_{"on" if on else "off"}')
             for c in a.concurrency for on in (False, True)]
    common.dump(folder/'plan.json', dict(model=a.model, requests_per_case=a.count,
        cases=cases, source=str(a.source), source_sha256=hashes,
        requests_sha256=hashlib.sha256((folder/'requests.json').read_bytes()).hexdigest(),
        cpu_gib=a.cpu_gb, staging_gib=make_config(False, 'plan', a.cpu_gb)['extra_config']['daosgds.gpu_buffer_gb'],
        chunk_tokens=128, max_model_len=a.max_model_len,
        generation=dict(temperature=0, seed=0, max_tokens=2048, enable_thinking=False, stop=['\nObservation:']),
        notes=['Replay of the earliest successful recorded model calls, ordered by start timestamp.',
               'One HumanMessage reconstructed from the recorded ReAct prompt. No new Python executions.',
               'Fixed input count/order and wave membership; response-paced waves, NOT fixed wall-clock arrivals.',
               'EOS enabled: generated length/output may differ; do not claim fixed generation work.',
               'Every case: fresh vLLM, empty DRAM, fresh DAOS namespace; no model-request warmup.',
               'OS/server hardware caches not flushed. Cases executed OFF then ON; one trial each.',
               'ON removes DRAM soft watermark, not physical capacity or existing DAOS serializer limits.',
               'Actual GPU allocation failure falls back to CPU for DRAM hits; DAOS may return a shorter hit prefix.',
               'DAOS prefetch, GPU-direct stores, async DRAM write mirror/read promotion enabled in all cases.',
               'No inter-wave draining, natural stores and promotions can overlap subsequent waves.',
               '20ms occupancy samples plus allocation/free events; instrumentation overhead included.']))
    if a.dry_run:
        print(folder, flush=True)
        return
    try:
        for spec in cases:
            case = folder/spec['name']
            case.mkdir()
            common.dump(folder/'status.json', dict(status='running', case=spec['name']))
            run_case(a, case, records, spec['concurrency'], spec['prefetch'])
        common.dump(folder/'status.json', dict(status='completed'))
    except BaseException as exc:
        common.dump(folder/'status.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
