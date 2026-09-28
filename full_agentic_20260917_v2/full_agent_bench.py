"""Document's full-agentic run1/restart/run2 protocol for DFS and object."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
from types import SimpleNamespace
import uuid

import httpx
import compare_e2e as common

ROOT = Path(__file__).resolve().parent


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
    args = parser.parse_args()
    folder = args.output.resolve(); folder.mkdir(parents=True, exist_ok=False)
    a = SimpleNamespace(repeats=1, chunk_size=args.chunk_size, pool='discospool',
                        container='kvcache', gpu_buffer_gb=10, io_workers=16, meta_workers=16,
                        model='Qwen/Qwen3-14B', port=8017, max_model_len=16384)
    run_id = 'agent-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    configs = common.configs(a, folder, run_id)
    command = common.server_command(a)
    common.dump(folder / 'manifest.json', {'args': vars(a), 'server_command': command,
        'protocol': 'supplied document full agentic: adventure-travel_0_0, react, run1/restart/run2; no grading',
        'differences_from_document': ['unified DFS/object backends and common native stack',
          'chunk 128, staging 10GiB, workers 16; eager, GPU utilization 0.75',
          'temperature 0, max output 2048 tokens/call, max iterations 25',
          'network-isolated Python tool with read-only task data at /data',
          'fresh namespace instead of deleting shared container'],
        'source_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
          ['full_agent_bench.py', 'run_full_agent.py', 'agent_python_worker.py',
           'run_vllm.sh', 'lmcache_daos/gds_backend.py', 'libdaosgdr.so',
           'discoverybench/agents/react_agent.py', 'discoverybench/agents/react_utils.py']}})
    results = []
    try:
        for mode in ('dfs', 'object'):
            for run in ('run1', 'run2'):
                tag = f'{mode}_{run}'
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
                            row.update(mode=mode, run=run)
                            count_key, sum_key = 'vllm:time_to_first_token_seconds_count', 'vllm:time_to_first_token_seconds_sum'
                            count = after.get(count_key, 0) - before.get(count_key, 0)
                            if count <= 0:
                                raise RuntimeError('TTFT metrics missing')
                            row['ttft_count'] = count
                            row['mean_server_ttft_ms'] = (after[sum_key] - before.get(sum_key, 0)) / count * 1000
                            results.append(row)
                            common.dump(folder / 'summary.json', results)
                            print(f'{tag}: complete; {row["llm_calls"]} LLM calls, '
                                  f'{row["python_calls"]} Python calls, {row["workflow_seconds"]:.2f}s', flush=True)
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
                if not hits or (run == 'run1' and int(hits[0][3]) != 0) or (run == 'run2' and int(hits[0][3]) <= 0):
                    raise RuntimeError(f'{tag}: initial cold/restart-hit check failed')
                if re.search(r'Double free|negative: -|DaosGdsBackend.*failed|Failed to create backend', text):
                    raise RuntimeError(f'{tag}: backend error')
                row['prefetch'] = re.findall(r'prefetch\[[^]]+\]: .*?GPU-direct\)', text)
                common.dump(folder / 'summary.json', results)
        common.dump(folder / 'status.json', {'status': 'complete', 'runs': len(results), 'grading': 'not run'})
        print(f'ALL COMPLETE: {folder}', flush=True)
    except BaseException as exc:
        common.dump(folder / 'status.json', {'status': 'failed', 'error': repr(exc), 'completed_runs': len(results)})
        raise


if __name__ == '__main__':
    main()
