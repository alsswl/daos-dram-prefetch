"""Full-agentic initial fill and repeated process restarts for DFS and object."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import shutil
import subprocess
import time
from types import SimpleNamespace
import uuid

import httpx
import compare_e2e as common

ROOT = Path(__file__).resolve().parent


def run_schedule(restart_runs, interleave_modes=False):
    """Index zero fills a fresh namespace; later indices reuse that namespace."""
    if restart_runs < 1:
        raise ValueError('restart_runs must be at least 1')
    modes = ('dfs', 'object')
    if not interleave_modes:
        return [(mode, index) for mode in modes for index in range(restart_runs + 1)]
    return [(mode, index) for index in range(restart_runs + 1)
            for mode in (modes if index % 2 == 0 else modes[::-1])]


def snapshot_metrics(client, path):
    response = client.get('/metrics'); response.raise_for_status()
    path.write_text(response.text)
    values = {}
    for line in response.text.splitlines():
        match = re.match(r'(vllm:time_to_first_token_seconds_(?:sum|count))(?:\{[^}]*\})? ([\d.eE+-]+)$', line)
        if match:
            values[match[1]] = values.get(match[1], 0) + float(match[2])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--restart-runs', type=int, default=1,
                        help='Number of fresh-vLLM cache-preserving replays after initial fill, per mode')
    parser.add_argument('--interleave-modes', action='store_true',
                        help='Alternate DFS/object within each replay index and reverse their order each time')
    args = parser.parse_args()
    if args.restart_runs < 1 or args.chunk_size < 1:
        parser.error('restart-runs and chunk-size must be positive')
    schedule = run_schedule(args.restart_runs, args.interleave_modes)
    folder = args.output.resolve(); folder.mkdir(parents=True, exist_ok=False)
    a = SimpleNamespace(repeats=1, chunk_size=args.chunk_size, pool='discospool',
                        container='kvcache', gpu_buffer_gb=10, io_workers=16, meta_workers=16,
                        model='Qwen/Qwen3-14B', port=8017, max_model_len=16384,
                        restart_runs=args.restart_runs, interleave_modes=args.interleave_modes)
    run_id = 'agent-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    configs = common.configs(a, folder, run_id)
    command = common.server_command(a)
    common.dump(folder / 'manifest.json', {'args': vars(a), 'server_command': command,
        'protocol': 'adventure-travel_0_0, react; initial fill plus cache-preserving fresh-vLLM replays; no grading',
        'schedule': [{'mode': mode, 'run': f'run{index + 1}', 'restart_index': index,
                      'phase': 'fill' if index == 0 else 'restart_hit'} for mode, index in schedule],
        'cache_policy': 'Fresh namespace per mode; all replay indices within a mode share and extend it. DAOS server cache is not reset.',
        'differences_from_document': ['unified DFS/object backends and common native stack',
          'chunk 128, staging 10GiB, workers 16; eager, GPU utilization 0.75',
          'temperature 0, max output 2048 tokens/call, max iterations 25',
          'network-isolated Python tool with read-only task data at /data',
          'fresh namespace instead of deleting shared container'],
        'source_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
          ['full_agent_bench.py', 'run_full_agent.py', 'agent_python_worker.py',
           'run_vllm.sh', 'lmcache_daos/gds_backend.py', 'libdaosgdr.so',
           'compare_e2e.py', 'discoverybench/agents/react_agent.py', 'discoverybench/agents/react_utils.py']}})
    sources = folder / 'executed_sources'; sources.mkdir()
    for name in ('full_agent_bench.py', 'run_full_agent.py', 'agent_python_worker.py',
                 'compare_e2e.py', 'run_vllm.sh'):
        shutil.copy2(ROOT / name, sources / name)
    dataset = ROOT / 'discoverybench/discoverybench/synth/train/adventure-travel_0_0'
    common.dump(folder / 'dataset_sha256.json', {
        str(path.relative_to(dataset)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(dataset.rglob('*')) if path.is_file()})
    results = []
    try:
        for sequence, (mode, restart_index) in enumerate(schedule, 1):
            run = f'run{restart_index + 1}'
            tag = f'{mode}_{run}'
            common.dump(folder / 'status.json', {'status': 'running', 'current': tag,
                        'completed_runs': len(results), 'planned_runs': len(schedule)})
            print(f'[{sequence}/{len(schedule)}] {tag}: starting fresh vLLM process', flush=True)
            with socket.socket() as port_check:
                port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                port_check.bind(('127.0.0.1', a.port))
            with (folder / f'{tag}_server.log').open('w') as log:
                server = subprocess.Popen(command, cwd=ROOT, env=common.environment(configs[1, mode], mode),
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    with httpx.Client(base_url=f'http://127.0.0.1:{a.port}', timeout=30, trust_env=False) as client:
                        deadline = time.monotonic() + 300
                        while True:
                            if server.poll() is not None:
                                raise RuntimeError(f'{tag}: server exited')
                            try:
                                if client.get('/health', timeout=2).status_code == 200:
                                    break
                            except httpx.TransportError:
                                pass
                            if time.monotonic() > deadline:
                                raise TimeoutError(f'{tag}: server startup timeout')
                            time.sleep(1)
                        print(f'{tag}: full agentic workflow starting', flush=True)
                        before = snapshot_metrics(client, folder / f'{tag}_metrics_before.txt')
                        with (folder / f'{tag}_agent_stdout.log').open('w') as agent_log:
                            agent = subprocess.run([str(ROOT/'agent-venv/bin/python3'),
                                str(ROOT/'run_full_agent.py'), '--folder', str(folder / tag),
                                '--port', str(a.port)], cwd=ROOT, stdout=agent_log,
                                stderr=subprocess.STDOUT, timeout=1800)
                        after = snapshot_metrics(client, folder / f'{tag}_metrics_after.txt')
                        common.dump(folder / f'{tag}_native_maps.json', common.native_maps(server.pid))
                        agent.check_returncode()
                        row = json.loads((folder / tag / 'status.json').read_text())
                        row.update(mode=mode, run=run, restart_index=restart_index,
                                   phase='fill' if restart_index == 0 else 'restart_hit',
                                   execution_sequence=sequence)
                        count_key, sum_key = 'vllm:time_to_first_token_seconds_count', 'vllm:time_to_first_token_seconds_sum'
                        count = after.get(count_key, 0) - before.get(count_key, 0)
                        if count <= 0 or count != row['llm_calls']:
                            raise RuntimeError(f'{tag}: TTFT metrics missing or model-call count mismatch')
                        row['ttft_count'] = count
                        row['mean_server_ttft_ms'] = (after[sum_key] - before.get(sum_key, 0)) / count * 1000
                finally:
                    if server.poll() is None:
                        os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL); server.wait(timeout=10)
            text = (folder / f'{tag}_server.log').read_text(errors='replace')
            hits = re.findall(r'Reqid: ([^,]+), Total tokens (\d+), Inference Engine computed tokens: (\d+), LMCache hit tokens: (\d+), need to load: (\d+)', text)
            row['cache_requests'] = [dict(request_id=h[0], prompt_tokens=int(h[1]),
                engine_computed_tokens=int(h[2]), lmcache_hit_tokens=int(h[3]), load_tokens=int(h[4])) for h in hits]
            if not hits or (restart_index == 0 and int(hits[0][3]) != 0) or (restart_index > 0 and int(hits[0][3]) <= 0):
                raise RuntimeError(f'{tag}: initial cold/restart-hit check failed')
            if len(hits) != row['llm_calls']:
                raise RuntimeError(f'{tag}: cache/model-call count mismatch')
            if re.search(r'Double free|negative: -|DaosGdsBackend.*failed|Failed to create backend', text):
                raise RuntimeError(f'{tag}: backend error')
            pre_shutdown = text.partition('[shutdown]')[0]
            if re.search(r'\bERROR\b', pre_shutdown):
                raise RuntimeError(f'{tag}: pre-shutdown errors require review')
            row['prefetch'] = re.findall(r'prefetch\[[^]]+\]: .*?GPU-direct\)', text)
            results.append(row)
            common.dump(folder / 'summary.json', results)
            print(f'{tag}: complete; {row["llm_calls"]} LLM calls, '
                  f'{row["python_calls"]} Python calls, {row["workflow_seconds"]:.2f}s, '
                  f'TTFT {row["mean_server_ttft_ms"]:.1f}ms; '
                  f'first hit {row["cache_requests"][0]["lmcache_hit_tokens"]}', flush=True)
        common.dump(folder / 'status.json', {'status': 'complete', 'runs': len(results), 'grading': 'not run'})
        print(f'ALL COMPLETE: {folder}', flush=True)
    except BaseException as exc:
        common.dump(folder / 'status.json', {'status': 'failed', 'error': repr(exc), 'completed_runs': len(results)})
        raise


if __name__ == '__main__':
    main()
