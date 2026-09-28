#!/usr/bin/env python3
"""Fresh 1 vs 2 CPU-prefetch workers, C16/D8/S8, cold256 then warm256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

import yaml

import compare_e2e as common
from cold_warm_prefetch import run_case
from prefetch_capacity_sweep import pool_query, check_space
from report_cold_warm_prefetch import select_events
from report_prefetch_timing import join, stats
from report_prefetch_capacity_sweep import timeline
from staging_mixed_pressure import read_events


def read(path): return json.loads(path.read_text())


def overlap(timings, start, end):
    points = [(r[start], 1) for r in timings if start in r and end in r]
    points += [(r[end], -1) for r in timings if start in r and end in r]
    active = maximum = 0
    for _, delta in sorted(points):
        active += delta
        maximum = max(maximum, active)
    assert active == 0
    return maximum


def report(root):
    assert read(root/'status.json')['status'] == 'completed'
    plan = read(root/'plan.json')
    records = read(root/'requests.json')
    expected = [(r['index'],r['prompt_sha256']) for r in records]
    configs, natives, namespaces, entries, details = [], [], [], [], {}
    for workers in (1,2):
        case = root/f'w{workers}'
        assert read(case/'status.json')['status'] == 'completed'
        cfg = yaml.safe_load((case/'config.yaml').read_text())
        ec = cfg['extra_config']
        assert ec.pop('daosgds.dram_prefetch_workers') == workers
        assert cfg['max_local_cpu_size'] == ec['daosgds.gpu_buffer_gb'] == 8
        assert ec['daosgds.dram_prefetch'] and ec['daosgds.dram_prefetch_policy'] == 'capacity'
        assert not ec['daosgds.dram_prefetch_cancel_queued'] and not ec['daosgds.dram_prefetch_early_ready']
        namespaces.append(ec.pop('daosgds.object_namespace')); ec.pop('daosgds.root')
        configs.append(cfg); natives.append(read(case/'native_maps.json'))
        cmd = read(case/'command.json')
        assert cmd[cmd.index('--max-num-seqs')+1] == '16'
        initial = read(case/'initial_sample.json')
        assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
        events = read_events(case)
        assert len({r['pid'] for r in events}) == 1
        for phase in ('cold','warm'):
            folder = case/phase
            before, after = read(folder/'initial_sample.json'), read(folder/'final_sample.json')
            assert before == (initial if phase=='cold' else read(case/'cold/final_sample.json'))
            assert after['used_bytes'] == after['dram_mirror']['pending_bytes'] == after['dram_mirror']['errors'] == 0
            pf = after['cpu_prefetch']; prior = before['cpu_prefetch']
            assert pf['copy_errors'] == pf['watermark_rejections'] == pf['deferred_pending_batches'] == 0
            calls = read(folder/'replay_calls.json')
            assert len(calls) == 256 and not any('error' in c for c in calls)
            assert [(c['index'],c['prompt_sha256']) for c in calls] == expected
            window = read(folder/'phase.json'); assert window['concurrency'] == 16
            selected = select_events(events,calls); rows = join(calls,selected)
            assert not any(r['other_failed_chunks'] for r in rows)
            timing = [r for r in selected if r['event']=='cpu_prefetch_timing']
            mapping = {}
            for row in timing:
                if 'cuda_stream_id' in row:
                    mapping.setdefault(row['worker_thread_id'],set()).add(row['cuda_stream_id'])
            assert all(len(streams)==1 for streams in mapping.values())
            assert len(set().union(*mapping.values())) == len(mapping) <= workers
            scoped = [e for e in events if window['start_ns']<=e['time_ns']<=window['end_ns']]
            samples = [e['used_bytes']/2**30 for e in scoped if e['event']=='occupancy_sample']
            candidates = sum(e['queried_chunks'] for e in selected if e['event']=='tier_lookup' and e['tier']=='dram')
            dram = sum(r['dram_lookup_chunks'] for r in rows); daos = sum(r['daos_lookup_chunks'] for r in rows)
            staged = sum(r.get('cpu_staged_chunks',0) for r in rows)
            returned = sum(r['daos_returned_chunks'] for r in rows)
            lost = sum(r['capacity_recomputed_tokens'] for r in rows)
            entry = dict(workers=workers,phase=phase,requests=256,
                ttft_ms=stats(r['ttft_ms'] for r in rows),queue_ms=stats(r.get('queue_ms') for r in rows),
                copy_ms=stats(r.get('copy_ms') for r in rows),retrieve_ms=stats(r.get('retrieve_ms') for r in rows),
                elapsed_seconds=(window['end_ns']-window['start_ns'])/1e9,
                computed_tokens=sum(r['computed_prompt_tokens'] for r in rows),
                completion_tokens=sum(r['completion_tokens'] for r in rows),
                dram_hit_pct=100*dram/candidates,daos_hit_pct=100*daos/candidates,
                capacity_recomputed_tokens=lost,capacity_affected_requests=sum(r['capacity_recomputed_tokens']>0 for r in rows),
                capacity_recompute_input_pct=100*lost/sum(r['prompt_tokens'] for r in rows),
                mean_staging_gib=statistics.mean(samples),peak_staging_gib=max(e['used_bytes'] for e in scoped)/2**30,
                dram_staged_pct=100*staged/dram,all_staging_pct=100*(staged+returned)/(dram+returned),
                worker_streams={str(k):list(v) for k,v in mapping.items()},
                max_overlapping_workers=overlap(timing,'worker_start_ns','worker_end_ns'),
                max_overlapping_copy_calls=overlap(timing,'copy_start_ns','copy_end_ns'),
                counters={k:v-prior.get(k,0) for k,v in pf.items()})
            entries.append(entry); details[workers,phase] = rows
            common.dump(folder/'timing_by_request.json',rows)
            timeline(folder,events,window['start_ns'],window['end_ns'],8)
    assert configs[0] == configs[1] and natives[0] == natives[1] and len(set(namespaces)) == 2
    for n,h in plan['source_sha256'].items():
        assert hashlib.sha256((root/'executed_sources'/n).read_bytes()).hexdigest() == h
    pairs = {}
    for phase in ('cold','warm'):
        zipped = list(zip(details[1,phase],details[2,phase],strict=True))
        pairs[phase] = dict(same_cached_requests=sum(a['cached_tokens']==b['cached_tokens'] for a,b in zipped),
            same_tier_requests=sum((a['dram_lookup_chunks'],a['daos_lookup_chunks'])==(b['dram_lookup_chunks'],b['daos_lookup_chunks']) for a,b in zipped))
    common.dump(root/'summary.json',entries); common.dump(root/'paired_checks.json',pairs)
    common.dump(root/'validation.json',dict(completed_cases=2,inputs_match=True,same_native_stack=True,
         independent_namespaces=True,buffers_drained=True,distinct_thread_local_streams=True))
    lines = ['# DRAM 프리페치 작업자 1개 vs 2개', '',
        'Qwen3-14B BF16, DRAM/staging 각 8GiB, 청크128, 동시 요청16, max-num-seqs16.',
        '두 조건 모두 DRAM/DAOS 프리페치 ON, 취소 OFF, 조기 준비 알림 OFF, 물리적 staging 용량만 적용.',
        '공통 대기열 + 작업자별 독립 CUDA stream. 대기열을 물리적으로 2개로 나누지는 않았다.',
        '각 조건은 새 프로세스·빈 DRAM·새 DAOS namespace에서 cold256 → 저장 완료 확인 → warm256.',
        '고정 DiscoveryBench 입력 재생이며 full agentic 도구 실행은 아니다. 조건당 1회, rolling 도착 시간·생성량은 달라질 수 있다.', '',
        '|단계|작업자|평균 TTFT ms|p95 ms|작업 대기 ms|copy 호출 ms|staging 평균/최대 GiB|DRAM hit %|DAOS hit %|추가 재계산 토큰|',
        '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for e in entries:
        lines.append(f"|{e['phase']}|{e['workers']}|{e['ttft_ms']['mean']:.2f}|{e['ttft_ms']['p95']:.2f}|{e['queue_ms']['mean']:.3f}|{e['copy_ms']['mean']:.3f}|{e['mean_staging_gib']:.3f}/{e['peak_staging_gib']:.3f}|{e['dram_hit_pct']:.2f}|{e['daos_hit_pct']:.2f}|{e['capacity_recomputed_tokens']}|")
    lines += ['', 'copy 시간은 CPU 측 제출+stream 완료 대기이며 순수 GPU DMA 시간이 아니다. 작업 중첩 역시 CPU 타임스탬프 기준이다.',
        '그래프의 0초 점은 시작 순간이 아니라 첫 0~2초 구간 통계다. 각 단계의 시작 직전 staging=0을 검증했다.', '',
        '## 시간 그래프', '',
        '- [작업자1 cold](w1/cold/staging_hits.png), [작업자1 warm](w1/warm/staging_hits.png)',
        '- [작업자2 cold](w2/cold/staging_hits.png), [작업자2 warm](w2/warm/staging_hits.png)', '',
        '[세부 수치](summary.json) · [요청별 재사용량 대조](paired_checks.json) · [검증 결과](validation.json)', '',
        '## 설정과 원복', '',
        '`extra_config`의 `daosgds.dram_prefetch_workers: 2`로 활성화한다. `1`로 바꾸거나 항목을 제거하고 프로세스를 재시작하면 원복된다. 기본값은 1이며 공용 YAML은 바꾸지 않았다.']
    (root/'RESULT_KO.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--port',type=int,default=8017)
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--report-only',action='store_true')
    a = p.parse_args(); root = a.output.resolve()
    if a.report_only: report(root); return
    root.mkdir(parents=True,exist_ok=False)
    a.model='Qwen/Qwen3-14B'; a.max_model_len=32768
    a.cpu_gib=a.staging_gib=8; a.cancel_queued=a.early_ready=False
    records = read(common.ROOT/'discovery_capacity_matrix_20260927/requests.json')
    assert len(records)==256 and all(hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256'] for r in records)
    common.dump(root/'requests.json',records)
    names = ['compare_prefetch_workers.py','prefetch_capacity_sweep.py','cold_warm_prefetch.py',
        'report_prefetch_capacity_sweep.py','report_prefetch_timing.py','report_cold_warm_prefetch.py',
        'report_capacity_matrix.py','analyze_discovery_staging.py','staging_mixed_pressure.py',
        'discovery_fixed_replay.py','discovery_rolling_replay.py','compare_e2e.py','run_vllm.sh',
        'libdaosgdr.so','lmcache_config_daosgds_async_dram.yaml','tests/prefetch_workers_gpu_copy.py']
    names += [str(f.relative_to(common.ROOT)) for f in sorted((common.ROOT/'lmcache_daos').glob('*.py'))]
    hashes = {}
    for name in names:
        dest=root/'executed_sources'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(common.ROOT/name,dest); hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    common.dump(root/'plan.json',dict(workers=[1,2],concurrency=16,max_num_seqs=16,cpu_gib=8,staging_gib=8,
        prefetch=True,cancel_queued=False,early_ready=False,requests_per_phase=256,requests_total=1024,source_sha256=hashes))
    if a.dry_run: common.dump(root/'status.json',dict(status='dry_run')); return
    os.environ['DAOS_GDS_PREFETCH_TIMING']='1'
    completed=[]
    try:
        space=pool_query(); common.dump(root/'pool_before.json',space); check_space(space)
        for label,args in [('gpu_copy',['tests/prefetch_workers_gpu_copy.py']),
                           ('daos_roundtrip',['tests/object_gpu_roundtrip.py','--size-mib','20'])]:
            env=dict(os.environ,DAOSGDS_TRANSPORT='object',PYTHONPATH=str(common.ROOT))
            r=subprocess.run([str(common.ROOT/'run_vllm.sh'),str(common.ROOT/'venv/bin/python3'),*args],
                cwd=common.ROOT,env=env,capture_output=True,text=True,timeout=120)
            (root/f'{label}_preflight.log').write_text(r.stdout+r.stderr)
            if r.returncode: raise RuntimeError(label+' preflight failed')
        for workers in (1,2):
            assert all(hashlib.sha256((common.ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
            case=root/f'w{workers}'; case.mkdir()
            q=pool_query(); common.dump(case/'pool_before.json',q); nvme=check_space(q)
            common.dump(root/'status.json',dict(status='running',workers=workers,completed=completed))
            print(f"WORKERS {workers} free={nvme['free']/1e9:.1f}GB min={nvme['min']/1e9:.1f}GB",flush=True)
            a.prefetch_workers=workers
            run_case(a,case,records,16,True)
            common.dump(case/'pool_after.json',pool_query()); completed.append(workers)
        common.dump(root/'status.json',dict(status='completed',completed=completed))
        report(root)
    except BaseException as exc:
        common.dump(root/'status.json',dict(status='failed',error=repr(exc),completed=completed)); raise


if __name__ == '__main__': main()
