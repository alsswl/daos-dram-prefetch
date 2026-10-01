#!/usr/bin/env python3
"""Sequential phases, identical replay, HTTP-streaming E2E comparison.

No container deletion. Each invocation creates private DFS roots/object dkey
prefixes. --dry-run writes a plan/configs without connecting to DAOS or a GPU.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid

import httpx
import yaml

ROOT = Path(__file__).resolve().parent
OC_SX = (1 << 24) | 0xFFFF
NATIVE = Path('/opt/daos-gds-gpu')


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--model', default='Qwen/Qwen3-14B')
    p.add_argument('--task-root', type=Path,
                   default=ROOT/'discoverybench/discoverybench/synth/train')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--steps', type=int, default=6)
    p.add_argument('--context-tokens', type=int, nargs='+',
                   help='Fixed independent token prompts, e.g. 8192 16384 31744; replaces --steps')
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--chunk-size', type=int, default=2048)
    p.add_argument('--pad-tokens', type=int, default=3000)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--max-tokens', type=int, default=64)
    p.add_argument('--gpu-buffer-gb', type=float, default=8)
    p.add_argument('--io-workers', type=int, default=8)
    p.add_argument('--meta-workers', type=int, default=8)
    p.add_argument('--pool', default='discospool')
    p.add_argument('--container', default='kvcache')
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--startup-timeout', type=float, default=300)
    p.add_argument('--request-timeout', type=float, default=180)
    p.add_argument('--drain-timeout', type=float, default=120)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    for name in ('repeats', 'steps', 'concurrency', 'chunk_size', 'max_model_len',
                 'max_tokens', 'gpu_buffer_gb', 'io_workers', 'meta_workers',
                 'startup_timeout', 'request_timeout', 'drain_timeout'):
        if getattr(a, name) <= 0:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if a.pad_tokens < 0 or not 1 <= a.port <= 65535:
        p.error('invalid pad-tokens or port')
    if a.context_tokens and any(n <= a.chunk_size or n + a.max_tokens > a.max_model_len
                                for n in a.context_tokens):
        p.error('context-tokens must exceed chunk-size and fit context plus generation')
    return a


def configs(a, folder, run_id):
    base = yaml.safe_load((ROOT/'lmcache_config_daosgds_unified.yaml').read_text())
    paths = {}
    for repeat in range(1, a.repeats + 1):
        for mode in ('dfs', 'object'):
            cfg = json.loads(json.dumps(base))
            cfg.update(chunk_size=a.chunk_size, enable_async_loading=True, local_cpu=False)
            ec = cfg['extra_config']
            ns = f'minji-compare-{run_id}-r{repeat}-{mode}'
            ec.update({'daosgds.transport': mode, 'daosgds.pool': a.pool,
                       'daosgds.container': a.container, 'daosgds.root': '/' + ns,
                       'daosgds.object_namespace': ns + ':',
                       'daosgds.dfs_oclass': OC_SX,
                       'daosgds.gpu_buffer_gb': a.gpu_buffer_gb,
                       'daosgds.io_workers': a.io_workers,
                       'daosgds.meta_workers': a.meta_workers})
            path = folder/f'r{repeat}_{mode}.yaml'
            path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            paths[repeat, mode] = path
    return paths


def environment(config, mode):
    env = os.environ.copy()
    env.update(LMCACHE_CONFIG_FILE=str(config), DAOSGDS_TRANSPORT=mode,
               DAOS_PROBE_CHUNKS='0', DAOS_GDS_MULTI_PREFETCH='1',
               LMCACHE_LOG_LEVEL='INFO', PYTHONHASHSEED='0', DAOSGDR_TIMING='0')
    return env


def server_command(a):
    return [str(ROOT/'run_vllm.sh'), str(ROOT/'venv/bin/python3'), '-m',
            'vllm.entrypoints.openai.api_server', '--model', a.model,
            '--served-model-name', 'comparison-model', '--host', '127.0.0.1',
            '--port', str(a.port), '--max-model-len', str(a.max_model_len),
            '--gpu-memory-utilization', '0.75', '--enforce-eager', '--seed', '0',
            '--no-enable-prefix-caching', '--enable-prompt-tokens-details',
            '--kv-transfer-config', '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}']


def workload(a):
    # Fixed scripted history, not model-generated history: transport output
    # differences must never change subsequent inputs or invalidate A/B pairing.
    os.environ['HF_HOME'] = '/home/hf/hf_cache'
    from kv_measure import load_task, build_prompt, make_pad_text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if a.context_tokens:
        prompts = []
        for i, length in enumerate(a.context_tokens):
            prefix = tokenizer.encode(f'Independent context {i}: summarize the following data.\n',
                                      add_special_tokens=False)
            unit = tokenizer.encode(' The experiment records measurements and compares results.',
                                    add_special_tokens=False)
            ids = (prefix + unit * (length // len(unit) + 1))[:length]
            prompts.append({'id': i + 1, 'tokens': ids,
                            'sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
        warm = tokenizer.encode('WARMUP ONLY\n' + 'warmup sample context ' * min(a.context_tokens),
                                add_special_tokens=False)[:min(a.context_tokens)]
        assert len({tuple(p['tokens'][:a.chunk_size]) for p in prompts}) == len(prompts)
        assert all(warm[:a.chunk_size] != p['tokens'][:a.chunk_size] for p in prompts)
        return prompts, {'id': 0, 'tokens': warm}, 'fixed-token reference comparison'
    task = next((t for d in sorted(a.task_root.iterdir()) if d.is_dir()
                 and (t := load_task(str(d)))), None)
    if task is None:
        raise RuntimeError('No valid DiscoveryBench task')
    prompts = []
    history = []
    for step in range(a.steps):
        text = build_prompt(task, history, make_pad_text(a.pad_tokens))
        ids = tokenizer.encode('MEASUREMENT DATA\n' + text, add_special_tokens=False)
        if len(ids) <= a.chunk_size or len(ids) + a.max_tokens > a.max_model_len:
            raise ValueError(f'prompt {step}: {len(ids)} tokens; adjust padding/context')
        prompts.append({'id': step + 1, 'tokens': ids,
                        'sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
        history.append(f'Analysis {step + 1}: inspect missing values, describe columns, '
                       'and compare groups before testing the discovery hypothesis.')
    # Equal-sized but different first-chunk hash: warmup cannot populate the
    # measurement prefix. Complete warmup performs PUT then a verified GET.
    warm = tokenizer.encode('WARMUP ONLY\n' + ('warmup sample context ' * a.max_model_len),
                            add_special_tokens=False)[:len(prompts[0]['tokens'])]
    assert all(warm[:a.chunk_size] != p['tokens'][:a.chunk_size] for p in prompts)
    return prompts, {'id': 0, 'tokens': warm}, task['domain']


def request(client, prompt, a, max_tokens=None):
    started = time.perf_counter()
    first = None
    first_token = None
    parts = []
    token_ids = []
    usage = None
    with client.stream('POST', '/v1/completions', json={
        'model': 'comparison-model', 'prompt': prompt['tokens'],
        'max_tokens': max_tokens or a.max_tokens, 'temperature': 0, 'seed': 0,
        'ignore_eos': True, 'stream': True, 'stream_options': {'include_usage': True},
        'return_token_ids': True,
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
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('token_ids') or []
                if delta:
                    if first_token is None:
                        first_token = time.perf_counter()
                    token_ids.extend(delta)
                piece = choice.get('text', '')
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    parts.append(piece)
    ended = time.perf_counter()
    if first is None or usage is None:
        raise RuntimeError('Missing streamed text or usage; no valid latency sample')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    if cached is None:
        raise RuntimeError('cached_tokens unavailable; cannot verify DAOS hit')
    if usage['prompt_tokens'] != len(prompt['tokens']):
        raise RuntimeError('Server changed prompt token count; invalid paired input')
    if usage['completion_tokens'] != (max_tokens or a.max_tokens):
        raise RuntimeError('Unexpected generation length; invalid workload sample')
    if len(token_ids) != usage['completion_tokens']:
        raise RuntimeError('Missing output token IDs; cannot verify token-level equality')
    text = ''.join(parts)
    return {'request_id': prompt['id'], 'prompt_tokens': usage['prompt_tokens'],
            'completion_tokens': usage['completion_tokens'], 'cached_tokens': cached,
            'ttft_ms': (first - started) * 1000, 'e2e_ms': (ended - started) * 1000,
            'first_token_ms': (first_token - started) * 1000,
            'output_token_sha256': hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
            'output_token_ids': token_ids,
            'output_sha256': hashlib.sha256(text.encode()).hexdigest(), 'text': text}


def expected_hit(prompt, chunk):
    # vLLM must compute at least one token even if the prompt is chunk-aligned.
    n = len(prompt['tokens'])
    return min(n - 1, (n // chunk) * chunk)


def await_hit(client, prompts, a):
    deadline = time.monotonic() + a.drain_timeout
    for prompt in prompts:
        while True:
            result = request(client, prompt, a, max_tokens=1)
            if result['cached_tokens'] >= expected_hit(prompt, a.chunk_size):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Cache readiness timeout for {prompt["id"]}')
            time.sleep(0.2)


def sample_pass(client, prompts, a, folder, tag, repeat, mode, phase):
    (folder/f'{tag}_metrics_before.txt').write_text(client.get('/metrics').text)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        rows = list(pool.map(lambda p: request(client, p, a), prompts))
    elapsed = time.perf_counter() - started
    (folder/f'{tag}_metrics_after.txt').write_text(client.get('/metrics').text)
    for row, prompt in zip(rows, prompts):
        row.update(repeat=repeat, mode=mode, phase=phase, prompt_sha256=prompt['sha256'],
                   expected_cached_tokens=expected_hit(prompt, a.chunk_size))
    dump(folder/f'{tag}_responses.json', rows)
    flat = [{k: v for k, v in row.items() if k not in ('text', 'output_token_ids')} for row in rows]
    with (folder/f'{tag}.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    valid = all(r['cached_tokens'] >= r['expected_cached_tokens'] for r in rows)
    if phase == 'fill':
        # Later requests share prefixes and can hit during fill. Only the
        # first submitted prompt is required to start cold in serial mode.
        valid = (rows[0]['cached_tokens'] == 0 if a.concurrency == 1
                 else any(r['cached_tokens'] == 0 for r in rows))
        if a.context_tokens:
            valid = all(r['cached_tokens'] == 0 for r in rows)
    summary = {'repeat': repeat, 'mode': mode, 'phase': phase, 'cache_check_pass': valid,
               'requests': len(rows), 'wall_seconds': elapsed,
               'requests_per_second': len(rows) / elapsed,
               'output_tokens_per_second': sum(r['completion_tokens'] for r in rows)/elapsed}
    for field in ('ttft_ms', 'first_token_ms', 'e2e_ms'):
        values = [r[field] for r in rows]
        summary.update({field+'_p50': statistics.median(values),
                        field+'_p95': percentile(values, 0.95),
                        field+'_mean': statistics.mean(values),
                        field+'_stdev': statistics.stdev(values) if len(values) > 1 else 0})
    dump(folder/f'{tag}_summary.json', summary)
    if not valid:
        raise RuntimeError(f'{tag}: cache validation failed; inspect CSV')
    return summary


def native_maps(pid):
    todo, paths = [pid], set()
    while todo:
        current = todo.pop()
        try:
            todo += [int(s) for s in Path(f'/proc/{current}/task/{current}/children').read_text().split()]
            for line in Path(f'/proc/{current}/maps').read_text().splitlines():
                p = line.split()[-1]
                if any(n in p for n in ('libdaos.so', 'libdfs.so', 'libmercury.so', 'libfabric.so')):
                    paths.add(p)
        except FileNotFoundError:
            pass
    for name, prefix in [('libdaos.so', str(NATIVE)), ('libmercury.so', str(NATIVE)),
                         ('libfabric.so', '/opt/ofi-cuda/')]:
        matches = [p for p in paths if name in Path(p).name]
        if not matches or not all(p.startswith(prefix) for p in matches):
            raise RuntimeError(f'Unexpected loaded stack {name}: {matches}')
    return sorted(paths)


def run_server(a, config, mode, repeat, stage, folder, prompts, warm):
    # Never reuse or terminate somebody else's server on this port.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', a.port))
    tag = f'r{repeat}_{mode}_{stage}'
    log_path = folder/f'{tag}_server.log'
    summaries = []
    with log_path.open('w') as log:
        proc = subprocess.Popen(server_command(a), cwd=ROOT, env=environment(config, mode),
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{a.port}',
                              timeout=a.request_timeout, trust_env=False) as client:
                deadline = time.monotonic() + a.startup_timeout
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f'Server exited; see {log_path}')
                    try:
                        if client.get('/health', timeout=2).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'Server startup timeout; see {log_path}')
                    time.sleep(1)
                print(f'{tag}: ready; warming model and DAOS', flush=True)
                # Warmup uses same isolated prompt for both implementations.
                request(client, warm, a)
                await_hit(client, [warm], a)
                dump(folder/f'{tag}_native_maps.json', native_maps(proc.pid))
                phases = ['fill'] if stage == 'fill' else ['restart_hit', 'same_process_hit']
                for phase in phases:
                    print(f'{tag}: measuring {phase}', flush=True)
                    summaries.append(sample_pass(client, prompts, a, folder,
                                                 f'r{repeat}_{mode}_{phase}',
                                                 repeat, mode, phase))
                if stage == 'fill':
                    # Real hit verification is a storage-readiness barrier,
                    # not a fixed sleep. These requests are outside measurement.
                    await_hit(client, prompts, a)
                    print(f'{tag}: persisted-prefix readiness verified', flush=True)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
    text = log_path.read_text(errors='replace')
    if re.search(r'Double free|negative: -|DaosGdsBackend.*failed|Failed to create backend', text):
        raise RuntimeError(f'Backend error in {log_path}; samples not accepted')
    return summaries


def main():
    a = parser()
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    folder = (a.output or ROOT/f'comparison_{run_id}').resolve()
    folder.mkdir(parents=True, exist_ok=False)
    paths = configs(a, folder, run_id)
    plan = {'run_id': run_id, 'args': {k: str(v) if isinstance(v, Path) else v
                                     for k, v in vars(a).items()},
            'server_command': server_command(a), 'order': [],
            'notes': ['Fixed schema plus scripted growing history; not full agentic.',
                      'TTFT is first nonempty HTTP text; includes local HTTP/client overhead.',
                      'fill may mix misses and shared-prefix hits.',
                      'restart_hit means client restart; DAOS server cache is not cleared.',
                      'same_process_hit still fetches from DAOS; vLLM prefix caching is off.',
                      'DFS OC_SX matches object redundancy/target class, not physical layout.',
                      'Only UUID-prefixed experiment namespaces are created; no cache deletion.'],
            'source_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                              for n in ['compare_e2e.py', 'run_vllm.sh', 'libdaosgdr.so',
                                        'lmcache_daos/gds_backend.py']}}
    for rep in range(1, a.repeats+1):
        modes = ['dfs', 'object'] if rep % 2 else ['object', 'dfs']
        plan['order'] += [{'repeat': rep, 'mode': m, 'config': str(paths[rep,m])} for m in modes]
    dump(folder/'manifest.json', plan)
    print(f'Results: {folder}', flush=True)
    if a.dry_run:
        print('DRY RUN: plan/configs written; no DAOS or GPU activity.')
        return
    summaries = []
    try:
        check = subprocess.run([str(ROOT/'run_vllm.sh'), str(NATIVE/'bin/daos'),
                                'container', 'get-prop', a.pool, a.container],
                               env=environment(paths[1,'dfs'], 'dfs'),
                               capture_output=True, text=True, timeout=60)
        (folder/'container_properties.txt').write_text(check.stdout + check.stderr)
        check.check_returncode()
        if (not re.search(r'\(layout_type\)\s+POSIX', check.stdout)
                or not re.search(r'\(rd_fac\)\s+0\b', check.stdout)
                or not re.search(r'\(status\)\s+HEALTHY', check.stdout)):
            raise RuntimeError('Comparison requires HEALTHY POSIX container with rd_fac=0')
        for name, command in [('pool_query', ['pool', 'query', a.pool]),
                              ('dfs_root_attributes', ['filesystem', 'get-attr', a.pool,
                                                       a.container, '--dfs-path', '/'])]:
            result = subprocess.run([str(ROOT/'run_vllm.sh'), str(NATIVE/'bin/daos'), *command],
                                    env=environment(paths[1, 'dfs'], 'dfs'),
                                    capture_output=True, text=True, timeout=60)
            (folder/f'{name}.txt').write_text(result.stdout + result.stderr)
            result.check_returncode()
        prompts, warm, domain = workload(a)
        dump(folder/'workload.json', {'domain': domain, 'prompts': prompts, 'warmup': warm})
        for item in plan['order']:
            rep, mode = item['repeat'], item['mode']
            for stage in ('fill', 'restart'):
                summaries.extend(run_server(a, paths[rep, mode], mode, rep, stage,
                                            folder, prompts, warm))
                dump(folder/'summary.json', summaries)
        with (folder/'summary.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
        # Text equality is a diagnostic, not a byte-level KV correctness gate.
        equality = []
        for rep in range(1, a.repeats+1):
            baseline = json.loads((folder/f'r{rep}_dfs_fill_responses.json').read_text())
            for mode in ('dfs', 'object'):
                for phase in ('fill', 'restart_hit', 'same_process_hit'):
                    rows = json.loads((folder/f'r{rep}_{mode}_{phase}_responses.json').read_text())
                    equality.append({'repeat': rep, 'mode': mode, 'phase': phase,
                                     'tokens_match_dfs_fill': all(x['output_token_ids']==y['output_token_ids']
                                                                  for x,y in zip(baseline, rows)),
                                     'text_matches_dfs_fill': all(x['output_sha256']==y['output_sha256']
                                                                 for x,y in zip(baseline, rows))})
        dump(folder/'output_text_checks.json', equality)
        dump(folder/'status.json', {'status': 'complete', 'cache_checks': 'passed',
                                    'all_output_tokens_equal': all(x['tokens_match_dfs_fill'] for x in equality),
                                    'all_output_text_equal': all(x['text_matches_dfs_fill'] for x in equality)})
        print(f'Completed: {folder}/summary.csv', flush=True)
    except BaseException as exc:
        dump(folder/'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
