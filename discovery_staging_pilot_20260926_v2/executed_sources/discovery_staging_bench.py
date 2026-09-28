#!/usr/bin/env python3
"""Sustained real-data DiscoveryBench agents on one vLLM; instrumented KV tiers.

Only own vLLM/task processes and private Python tool containers are stopped.
No DAOS deletion, no forced hit ratios, no padding or cached-read-only replay.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
import uuid

import yaml

import compare_e2e as common
from staging_mixed_pressure import server, latest_sample

ROOT = common.ROOT


def select_tasks(count=32, seed=926):
    groups = defaultdict(list)
    data_root = ROOT/'discoverybench/discoverybench/real'
    for path in sorted(data_root.rglob('metadata*.json')):
        data = json.loads(path.read_text())
        if not data.get('queries') or not data.get('datasets'):
            continue
        if not all((path.parent/d['name']).is_file() for d in data['datasets']):
            continue
        groups[data.get('domain', path.parent.name)].append(path)
    rng = random.Random(seed)
    for paths in groups.values():
        rng.shuffle(paths)
    selected = []
    while len(selected) < count:
        changed = False
        for domain in sorted(groups):
            if groups[domain] and len(selected) < count:
                path = groups[domain].pop()
                data = json.loads(path.read_text())
                files = [path] + [path.parent/d['name'] for d in data['datasets']]
                selected.append(dict(metadata=str(path), domain=domain,
                    hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
                changed = True
        if not changed:
            raise ValueError('Not enough local tasks')
    return selected


def parse_phases(value):
    phases = []
    for text in value.split(','):
        n, seconds = text.split(':')
        n, seconds = int(n), float(seconds)
        if not 1 <= n <= 16 or seconds <= 0:
            raise ValueError('Concurrency must be 1..16; duration positive')
        phases.append((n, seconds))
    return phases


def stop_job(job):
    proc = job['process']
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    # A forced process kill must not strand its own Python tool container.
    config = job['folder']/'sandbox_command.json'
    if config.exists():
        command = json.loads(config.read_text())
        name = command[command.index('--name')+1]
        if not name.startswith('minji-agent-'):
            raise RuntimeError('Unexpected sandbox name; refusing cleanup')
        subprocess.run(['podman', 'stop', '--time', '2', name], capture_output=True, timeout=15)


def run_phases(a, folder, tasks, client):
    jobs, active, phases = [], [], []
    counter = 0
    event_file = (folder/'controller.jsonl').open('w', buffering=1)
    def emit(event, **fields):
        event_file.write(json.dumps(dict(event=event, time_ns=time.time_ns(), **fields))+'\n')
    def finish(job):
        result_path = job['folder']/'status.json'
        status = json.loads(result_path.read_text()) if result_path.exists() else {'status': 'no_status'}
        row = dict(index=job['index'], task=job['task'], phase=job['phase'],
                   start_ns=job['start_ns'], end_ns=time.time_ns(),
                   folder=str(job['folder'].relative_to(folder)),
                   returncode=job['process'].returncode, forced=job.get('forced', False), **status)
        jobs.append(row)
        common.dump(folder/'jobs.json', jobs)
        job['log'].close()
        emit('agent_end', index=row['index'], status=row['status'], phase=row['phase'])
    try:
        for phase_index, (concurrency, duration) in enumerate(a.phases):
            started = time.monotonic()
            phase = dict(index=phase_index, concurrency=concurrency,
                         requested_seconds=duration, start_ns=time.time_ns())
            phases.append(phase)
            common.dump(folder/'phases.json', phases)
            emit('phase_start', **phase)
            next_report = next_metrics = next_state = 0
            while True:
                now = time.monotonic()
                accepting = now-started < duration
                for job in active[:]:
                    if job['process'].poll() is None and (now-job['started'] > a.task_timeout or
                            now-started > duration+a.drain_seconds):
                        job['forced'] = True
                        stop_job(job)
                    if job['process'].poll() is not None:
                        finish(job)
                        active.remove(job)
                if not accepting and 'admission_end_ns' not in phase:
                    phase['admission_end_ns'] = time.time_ns()
                    common.dump(folder/'phases.json', phases)
                    emit('phase_admission_end', phase=phase_index)
                while accepting and len(active) < concurrency:
                    task = tasks[counter % len(tasks)]
                    job_folder = folder/'agents'/f'job_{counter:05d}'
                    job_folder.parent.mkdir(exist_ok=True)
                    log = (job_folder.parent/f'job_{counter:05d}.log').open('w')
                    command = [str(ROOT/'agent-venv/bin/python3'), str(ROOT/'run_full_agent.py'),
                        '--folder', str(job_folder), '--port', str(a.port),
                        '--metadata', task['metadata'], '--dataset-type', 'real', '--stream',
                        '--max-iterations', '25', '--disable-thinking']
                    if a.tool_site:
                        command += ['--tool-site', str(a.tool_site)]
                    env = os.environ.copy()
                    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    job = dict(index=counter, folder=job_folder, phase=phase_index,
                        task=task['metadata'], process=proc, log=log,
                        started=time.monotonic(), start_ns=time.time_ns())
                    active.append(job)
                    emit('agent_start', index=counter, pid=proc.pid, phase=phase_index,
                         metadata=task['metadata'], folder=str(job_folder))
                    counter += 1
                if now >= next_metrics:
                    response = client.get('/metrics', timeout=10)
                    response.raise_for_status()
                    with (folder/'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(time_ns=time.time_ns(), text=response.text))+'\n')
                    next_metrics = now+5
                if now >= next_state:
                    state = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,'
                        'temperature.gpu,clocks.current.sm,power.draw', '--format=csv,noheader'],
                        capture_output=True, text=True, timeout=10)
                    emit('gpu_state', output=state.stdout.strip())
                    next_state = now+30
                if now >= next_report:
                    sample = latest_sample(folder) or {}
                    completed = sum(j['status'] == 'complete' for j in jobs)
                    print(f'{folder.name} C{concurrency} {now-started:.0f}/{duration:.0f}s '
                          f'active_agents={len(active)} finished={len(jobs)} complete={completed} '
                          f'staging={sample.get("used_bytes",0)/2**30:.2f}GiB '
                          f'cpu={sample.get("cpu_hot_bytes",0)/2**30:.2f}GiB', flush=True)
                    next_report = now+30
                if len(jobs) >= 8 and all(j.get('llm_calls', 0) == 0 for j in jobs[-8:]):
                    raise RuntimeError('Eight consecutive workflows failed before any model call')
                if not accepting and not active:
                    break
                time.sleep(.3)
            phase['end_ns'] = time.time_ns()
            common.dump(folder/'phases.json', phases)
            emit('phase_end', phase=phase_index)
        return jobs
    finally:
        for job in active:
            stop_job(job)
            finish(job)
        event_file.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-gb', type=float, required=True)
    p.add_argument('--conditions', default='off,on')
    p.add_argument('--phases', default='4:300,8:300,16:600')
    p.add_argument('--tasks', type=int, default=32)
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--task-timeout', type=float, default=300)
    p.add_argument('--drain-seconds', type=float, default=90)
    p.add_argument('--tool-site', type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    a.phases = parse_phases(a.phases)
    conditions = a.conditions.split(',')
    if not 0 < a.cpu_gb <= 64 or any(c not in ('off','on') for c in conditions):
        p.error('Invalid CPU budget or condition')
    a.model, a.max_model_len = 'Qwen/Qwen3-14B', 32768
    folder = a.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    tasks = select_tasks(a.tasks)
    common.dump(folder/'tasks.json', tasks)
    hashes = {}
    for name in ['discovery_staging_bench.py', 'run_full_agent.py', 'agent_python_worker.py',
            'compare_e2e.py', 'staging_mixed_pressure.py', 'run_vllm.sh',
            'lmcache_daos/gds_backend.py', 'lmcache_daos/staging_probe_backend.py',
            'lmcache_daos/staging_trace.py', 'lmcache_daos/dram_prefetch_backend.py',
            'discoverybench/agents/react_agent.py', 'discoverybench/agents/react_utils.py',
            'discoverybench/utils/autonomous_single_agent.py', 'libdaosgdr.so']:
        dst = folder/'executed_sources'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    common.dump(folder/'plan.json', dict(args={k:str(v) if isinstance(v,Path) else v
        for k,v in vars(a).items()}, source_sha256=hashes, notes=[
        'Real-data DiscoveryBench subset; scientific-answer grading not run.',
        '25 agent steps, 2048 output tokens/call; original prompts, no padding.',
        'One live vLLM process per condition. Closed-loop agent arrivals, Python tool pauses included.',
        'No artificial tier ratios; CPU LRU and async DAOS stores remain active.',
        'New DAOS namespace per condition; no cache deletion or server cache flush.',
        'OFF/ON have same task catalogue, but actual model calls and arrival timing can diverge.',
        'Each phase admits new workflows for requested duration, then drains for at most drain_seconds.',
        '20ms nominal allocator sampling plus allocation/free events; instrumentation has overhead.',
        'Failures, context overflows and forced timeout stops are retained, never silently discarded.']))
    if a.dry_run:
        print(folder)
        return
    cfg = yaml.safe_load((ROOT/'lmcache_config_daosgds_unified.yaml').read_text())
    cfg.update(chunk_size=128, local_cpu=True, max_local_cpu_size=a.cpu_gb,
               enable_async_loading=True, use_layerwise=False, store_location=None)
    cfg['extra_config'].update({'daosgds.transport':'object', 'daosgds.gpu_buffer_gb':10,
        'daosgds.cpu_prefetch_gpu_gb':5, 'daosgds.io_workers':16, 'daosgds.meta_workers':16,
        'daosgds.store':True, 'daosgds.probe_interval_ms':20,
        'storage_plugin.daosgds.module_path':'lmcache_daos.staging_probe_backend'})
    try:
        for condition in conditions:
            case = folder/f'prefetch_{condition}'
            case.mkdir()
            ns = 'minji-discovery-live-'+uuid.uuid4().hex
            cfg['extra_config'].update({'daosgds.root':'/'+ns, 'daosgds.object_namespace':ns+':',
                'storage_plugin.daosgds.class_name':'ProbePrefetchBackend' if condition=='on' else 'ProbeBackend'})
            config = case/'config.yaml'
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            common.dump(folder/'status.json', dict(status='running', condition=condition))
            with server(a, config, case) as client:
                run_phases(a, case, tasks, client)
            common.dump(case/'status.json', dict(status='completed'))
        common.dump(folder/'status.json', dict(status='completed'))
    except BaseException as exc:
        common.dump(folder/'status.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
