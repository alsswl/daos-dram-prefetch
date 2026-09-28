#!/usr/bin/env python3
"""LongBench GovReport200: C16/D8/S8, DRAM prefetch OFF/ON, cold then warm.

All original rows, original order; normal EOS and 512 output-token cap. No
synthetic repetition, forced generation, cache pinning or admission delays.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import zipfile

import yaml

import compare_e2e as common
from cold_warm_prefetch import run_case
from discovery_fixed_replay import replay_one
from prefetch_capacity_sweep import pool_query, check_space
from report_capacity_matrix import attribute_recomputation
from report_cold_warm_prefetch import select_events
from report_prefetch_timing import stats
from report_prefetch_capacity_sweep import timeline
from staging_mixed_pressure import read_events

ARCHIVE = Path('/root/.cache/huggingface/hub/datasets--THUDM--LongBench/snapshots/'
               '5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip')
PROMPT = ('You are given a report by a government agency. Write a one-page summary of the report.\n\n'
          'Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:')


def read(path):
    return json.loads(path.read_text())


def prepare():
    os.environ['HF_HOME'] = '/home/hf/hf_cache'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-14B', local_files_only=True)
    assert len(tok) > 100000 and tok.encode('Hello world')
    def length(prompt):
        result = tok.apply_chat_template([dict(role='user', content=prompt)],
                   tokenize=True, add_generation_prompt=True, enable_thinking=False,
                   return_dict=False)
        assert isinstance(result, list) and all(isinstance(i, int) for i in result)
        return len(result)
    with zipfile.ZipFile(ARCHIVE) as archive:
        raw = archive.read('data/gov_report.jsonl')
    source = [json.loads(line) for line in raw.splitlines()]
    assert len(source) == 200
    records = []
    for i, item in enumerate(source):
        context = item['context']
        tokens = tok.encode(context, add_special_tokens=False)
        original = length(PROMPT.format(context=context))
        kept = len(tokens)
        prompt = PROMPT.format(context=context)
        while length(prompt) > 32768 - 512:
            kept -= max(1, length(prompt) - (32768 - 512) + 8)
            assert kept > 0
            context = tok.decode(tokens[:kept//2], skip_special_tokens=False) + tok.decode(tokens[-(kept-kept//2):], skip_special_tokens=False)
            prompt = PROMPT.format(context=context)
        records.append(dict(index=i, dataset_id=item['_id'], prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            original_context_tokens=len(tokens), original_prompt_tokens=original,
            expected_prompt_tokens=length(prompt), truncated=kept<len(tokens),
            retained_context_tokens=kept, reference_answers=item['answers']))
    return records, dict(archive=str(ARCHIVE), archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        member='data/gov_report.jsonl', member_sha256=hashlib.sha256(raw).hexdigest(),
        prompt_template=PROMPT, max_tokens=512, max_model_len=32768,
        truncation='Head/tail of context only, preserving instructions and reserving output/chat tokens',
        truncated_ids=[r['dataset_id'] for r in records if r['truncated']],
        prompt_tokens=stats(r['expected_prompt_tokens'] for r in records),
        total_prompt_tokens=sum(r['expected_prompt_tokens'] for r in records))


def request(client, record, barrier):
    row = replay_one(client, record, barrier, max_tokens=512, stop=())
    if 'error' not in row and row['prompt_tokens'] != record['expected_prompt_tokens']:
        row['error'] = f"Tokenization mismatch: {row['prompt_tokens']} != {record['expected_prompt_tokens']}"
    return row


def report(root):
    assert read(root/'status.json')['status'] == 'completed'
    records = read(root/'requests.json')
    expected = [(r['index'], r['prompt_sha256']) for r in records]
    entries, configs, natives, namespaces = [], [], [], []
    for label, enabled in [('off', False), ('on', True)]:
        case = root/label
        assert read(case/'status.json')['status'] == 'completed'
        cfg = yaml.safe_load((case/'config.yaml').read_text()); ec = cfg['extra_config']
        assert ec.pop('daosgds.dram_prefetch') is enabled
        namespaces.append(ec.pop('daosgds.object_namespace')); ec.pop('daosgds.root')
        configs.append(cfg); natives.append(read(case/'native_maps.json'))
        initial = read(case/'initial_sample.json')
        assert initial['cpu_hot_bytes'] == initial['used_bytes'] == initial['daos_puts'] == 0
        events = read_events(case)
        for phase in ('cold', 'warm'):
            folder = case/phase; window = read(folder/'phase.json')
            calls = read(folder/'replay_calls.json')
            assert len(calls) == 200 and not any('error' in c for c in calls)
            assert [(c['index'], c['prompt_sha256']) for c in calls] == expected
            assert all(c['prompt_tokens'] == r['expected_prompt_tokens'] for c,r in zip(calls,records))
            before = read(folder/'initial_sample.json'); after = read(folder/'final_sample.json')
            assert before == (initial if phase == 'cold' else read(case/'cold/final_sample.json'))
            assert before['used_bytes'] == after['used_bytes'] == after['dram_mirror']['pending_bytes'] == 0
            selected = select_events(events, calls)
            rows = attribute_recomputation(calls, selected, 128)
            common.dump(folder/'attribution_by_request.json', rows)
            # Preempted long requests may have multiple lookup/read batches.
            # Never present an unvalidated attribution as capacity recomputation.
            issues = [r['index'] for r in rows if r['attribution_issues'] or r['unattributed_shortfall_tokens']]
            scoped = [e for e in events if window['start_ns'] <= e['time_ns'] <= window['end_ns']]
            samples = [e['used_bytes']/2**30 for e in scoped if e['event'] == 'occupancy_sample']
            candidates = sum(e['queried_chunks'] for e in selected if e['event']=='tier_lookup' and e['tier']=='dram')
            dram = sum(r['dram_lookup_chunks'] for r in rows); daos = sum(r['daos_lookup_chunks'] for r in rows)
            staged = sum(e['staged_chunks'] for e in selected if e['event']=='cpu_get_ready')
            lost = sum(r['capacity_recomputed_tokens'] for r in rows)
            total = sum(c['prompt_tokens'] for c in calls)
            pf = after['cpu_prefetch']; prior = before['cpu_prefetch']
            entry = dict(prefetch=enabled, phase=phase, requests=len(calls),
                elapsed_seconds=(window['end_ns']-window['start_ns'])/1e9,
                ttft_ms=stats(c['ttft_ms'] for c in calls),
                prompt_tokens=total, cached_tokens=sum(c['cached_tokens'] for c in calls),
                computed_tokens=sum(c['prompt_tokens']-c['cached_tokens'] for c in calls),
                completion_tokens=sum(c['completion_tokens'] for c in calls),
                output_tokens=stats(c['completion_tokens'] for c in calls),
                finish_reasons=dict(Counter(c['finish_reason'] for c in calls)),
                lookup_candidates=candidates, dram_lookup_chunks=dram, daos_lookup_chunks=daos,
                dram_hit_pct=100*dram/candidates if candidates else None,
                daos_hit_pct=100*daos/candidates if candidates else None,
                mean_staging_gib=statistics.mean(samples), peak_staging_gib=max(e['used_bytes'] for e in scoped)/2**30,
                dram_staged_chunks=staged, dram_staged_pct=100*staged/dram if dram else None,
                capacity_recomputed_tokens=None if issues else lost,
                capacity_recompute_input_pct=None if issues else 100*lost/total,
                attribution_issue_request_indices=issues,
                counters={k:v-prior.get(k,0) for k,v in pf.items()})
            entries.append(entry)
            timeline(folder, events, window['start_ns'], window['end_ns'], 8)
    assert configs[0] == configs[1] and natives[0] == natives[1] and len(set(namespaces)) == 2
    common.dump(root/'summary.json', entries)
    common.dump(root/'validation.json',dict(inputs_match=True, same_native_stack=True,
        independent_namespaces=True, cold_empty=True, phases_drained=True,
        attribution_complete=not any(e['attribution_issue_request_indices'] for e in entries)))
    lines = ['# GovReport 프리페치 OFF/ON 비교', '',
        'Qwen3-14B BF16 · DRAM8GiB / staging8GiB · 동시 요청16 / max-num-seqs16 · 청크128.',
        'LongBench GovReport200개를 원래 순서로 cold200 → 같은 프로세스에서 warm200으로 실행했다.',
        'OFF/ON은 DRAM 프리페치만 변경. DAOS 프리페치 ON, 복사 작업자1, 대기열 취소/조기 알림 OFF.',
        '각 조건은 새 프로세스·빈 DRAM·새 DAOS namespace. 요청이 끝날 때 다음 요청을 투입하는 rolling 방식.',
        '한 페이지 요약, 출력 최대512토큰, 정상 EOS. 원문이 긴 일부 입력은 32K 한도에 맞춰 앞/뒤를 보존했다.',
        '원본 전체 GovReport 학습/검증셋이 아니라 LongBench의 GovReport200이며, 요약 품질 점수는 평가하지 않았다.', '',
        '|단계|프리페치|평균 TTFT ms|p95 ms|전체 s|DRAM hit %|DAOS hit %|staging 평균/최대 GiB|생성 토큰|',
        '|---|---|---:|---:|---:|---:|---:|---|---:|']
    for e in entries:
        lines.append(f"|{e['phase']}|{'ON' if e['prefetch'] else 'OFF'}|{e['ttft_ms']['mean']:.2f}|{e['ttft_ms']['p95']:.2f}|{e['elapsed_seconds']:.2f}|{e['dram_hit_pct']:.2f}|{e['daos_hit_pct']:.2f}|{e['mean_staging_gib']:.3f}/{e['peak_staging_gib']:.3f}|{e['completion_tokens']}|")
    lines += ['', '## 해석 시 주의', '',
        '- 조건당 1회다. 처리 속도에 따라 실제 도착 시각·완료 순서·DRAM 보관 상태·생성량은 달라질 수 있다.',
        '- hit 비율은 최초 CPU-tier 조회 후보 청크 기준이다. 재조회가 있으면 조회 횟수 기준이며 실제 재사용 토큰 비율과 다르다.',
        '- 서로 다른 긴 문서200개의 KV는 DRAM8GiB보다 훨씬 크다. warm도 DRAM all-hit가 아니며 DAOS 중심일 수 있다.',
        '- staging 그래프는 2초 구간의 샘플 평균/최대다. 첫 점은 0초 순간이 아니라 첫2초 구간이다.',
        '- 재계산 원인 검증에 문제가 있는 요청은 summary.json에 표시하며 해당 단계의 원인별 재계산율은 단정하지 않는다.', '',
        '## 그래프', '',
        '- [OFF cold](off/cold/staging_hits.png) · [OFF warm](off/warm/staging_hits.png)',
        '- [ON cold](on/cold/staging_hits.png) · [ON warm](on/warm/staging_hits.png)', '',
        '[전체 수치](summary.json) · [입력/절단 정보](dataset.json) · [조건 검증](validation.json)']
    (root/'RESULT_KO.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--report-only', action='store_true')
    a = p.parse_args(); root = a.output.resolve()
    if a.report_only:
        report(root); return
    root.mkdir(parents=True, exist_ok=False)
    a.model='Qwen/Qwen3-14B'; a.max_model_len=32768
    a.cpu_gib=a.staging_gib=8; a.prefetch_workers=1
    a.cancel_queued=a.early_ready=False
    records, data = prepare()
    common.dump(root/'requests.json', records); common.dump(root/'dataset.json', data)
    names = ['compare_govreport.py','cold_warm_prefetch.py','discovery_fixed_replay.py',
             'discovery_rolling_replay.py','staging_mixed_pressure.py','compare_e2e.py',
             'prefetch_capacity_sweep.py','report_prefetch_capacity_sweep.py','report_prefetch_timing.py',
             'report_cold_warm_prefetch.py','report_capacity_matrix.py','analyze_discovery_staging.py',
             'run_vllm.sh','libdaosgdr.so','lmcache_config_daosgds_async_dram.yaml']
    names += [str(f.relative_to(common.ROOT)) for f in sorted((common.ROOT/'lmcache_daos').glob('*.py'))]
    hashes = {}
    for name in names:
        dest=root/'executed_sources'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(common.ROOT/name,dest); hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    common.dump(root/'plan.json',dict(model=a.model,concurrency=16,max_num_seqs=16,
        cpu_gib=8,staging_gib=8,prefetch_workers=1,chunk_tokens=128,max_tokens=512,
        requests_per_phase=200,requests_total=800,case_order=['off','on'],source_sha256=hashes))
    if a.dry_run:
        common.dump(root/'status.json',dict(status='dry_run')); return
    os.environ['DAOS_GDS_PREFETCH_TIMING']='1'
    completed=[]
    try:
        space=pool_query(); common.dump(root/'pool_before.json',space); nvme=check_space(space)
        required=(data['total_prompt_tokens']+200*512)*163840*2+220_000_000_000
        if nvme['free'] < required:
            raise RuntimeError(f"Need conservative free {required/1e9:.1f}GB, found {nvme['free']/1e9:.1f}GB")
        r=subprocess.run([str(common.ROOT/'run_vllm.sh'),str(common.ROOT/'venv/bin/python3'),
            'tests/object_gpu_roundtrip.py','--size-mib','20'],cwd=common.ROOT,
            env=dict(os.environ,DAOSGDS_TRANSPORT='object'),capture_output=True,text=True,timeout=120)
        (root/'storage_preflight.log').write_text(r.stdout+r.stderr)
        if r.returncode: raise RuntimeError('DAOS preflight failed')
        for label, enabled in [('off',False),('on',True)]:
            assert all(hashlib.sha256((common.ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
            case=root/label; case.mkdir()
            q=pool_query(); common.dump(case/'pool_before.json',q); check_space(q)
            common.dump(root/'status.json',dict(status='running',case=label,completed=completed))
            run_case(a,case,records,16,enabled,request_fn=request)
            common.dump(case/'pool_after.json',pool_query()); completed.append(label)
        common.dump(root/'status.json',dict(status='completed',completed=completed))
    except BaseException as exc:
        common.dump(root/'status.json',dict(status='failed',completed=completed,error=repr(exc)))
        raise
    report(root)


if __name__ == '__main__':
    main()
