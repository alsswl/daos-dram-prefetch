#!/usr/bin/env python3
"""Cold-start EQ-Bench longform generation pilot, first four original stories.

Retains 13 original prompts, history and normal EOS. No judge, retries, warm
replay, padding, skip-save or intermediate cache drain. Separate conversations
advance independently. Adapted local performance run, not a leaderboard score.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time
import uuid

import yaml

import compare_e2e as common
from cold_warm_prefetch import drain
from discovery_fixed_replay import make_config, await_empty
from discovery_rolling_replay import LogHealth
from prefetch_capacity_sweep import pool_query, check_space
from report_capacity_matrix import attribute_recomputation
from report_cold_warm_prefetch import select_events
from report_prefetch_timing import stats
from report_prefetch_capacity_sweep import timeline
from staging_mixed_pressure import server, read_events

UPSTREAM = common.ROOT/'eqbench_longform_upstream_20260928'
COMMIT = '34f60a028c3f973c19cde98dc5a9e8f9875a87e3'


def read(path):
    return json.loads(path.read_text())


def load_templates(data):
    templates = {i: (data/f'prompt{i}.txt').read_text() for i in range(1, 6)}
    templates[6] = (data/'prompt_chapter_first.txt').read_text()
    middle = (data/'prompt_chapter_intermediate.txt').read_text()
    for step in range(7, 13):
        templates[step] = middle.format(chapter_number=step-5)
    templates[13] = (data/'prompt_chapter_last.txt').read_text().format(chapter_number=8)
    return templates


def build_messages(story, templates, outputs, step):
    messages = []
    for i in range(1, step+1):
        prompt = templates[i].replace('{writing_prompt}', story['writing_prompt']).replace('{n_chapters}', '8')
        messages.append(dict(role='user', content=prompt))
        if i < step:
            messages.append(dict(role='assistant', content=outputs[str(i)]))
    return messages


def common_prefix(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def streaming_call(client, messages, seed):
    payload = dict(model='comparison-model', messages=messages,
        temperature=.7, min_p=.1, top_p=1.0, top_k=-1, max_tokens=4000, seed=seed,
        chat_template_kwargs=dict(enable_thinking=False), stream=True,
        stream_options=dict(include_usage=True))
    row = dict(start_ns=time.time_ns(), sampling={k:v for k,v in payload.items() if k!='messages'})
    start = time.perf_counter(); first = None; parts = []; usage = None; finish = None
    try:
        with client.stream('POST', '/v1/chat/completions', json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith('data: '): continue
                raw = line[6:]
                if raw == '[DONE]': break
                e = json.loads(raw)
                if e.get('error'): raise RuntimeError(str(e['error']))
                row['server_request_id'] = e.get('id', row.get('server_request_id'))
                usage = e.get('usage') or usage
                for choice in e.get('choices', []):
                    finish = choice.get('finish_reason') or finish
                    text = choice.get('delta', {}).get('content') or ''
                    if text:
                        if first is None: first = time.perf_counter()
                        parts.append(text)
        if first is None or not usage or not finish: raise RuntimeError('Incomplete stream')
        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        if cached is None: raise RuntimeError('Actual cached_tokens missing')
        output = ''.join(parts)
        row.update(ttft_ms=(first-start)*1000, usage=usage, prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'], cached_tokens=cached,
            finish_reason=finish, output=output, output_words=len(output.split()),
            output_sha256=hashlib.sha256(output.encode()).hexdigest(),
            upstream_short_response=len(output.strip())<500)
    except Exception as exc:
        row['error'] = repr(exc)
    row.update(end_ns=time.time_ns(), elapsed_seconds=time.perf_counter()-start)
    return row


def run_story(client, tok, story_id, story, templates, root, barrier, stop):
    folder = root/'stories'/story_id; folder.mkdir(parents=True)
    outputs = {}; calls = []; previous = []
    barrier.wait(timeout=30)
    try:
        for step in range(1, 14):
            if stop.is_set(): raise RuntimeError('Pilot stopped after another task/server failure')
            messages = build_messages(story, templates, outputs, step)
            tokens = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=False)
            assert isinstance(tokens, list) and all(isinstance(t,int) for t in tokens)
            if len(tokens)+4000 > 32768:
                raise RuntimeError(f'Context budget exceeded at step{step}: {len(tokens)}+4000>32768; no truncation')
            request_record = dict(messages=messages, expected_prompt_tokens=len(tokens),
                previous_prompt_tokens=len(previous), common_previous_prompt_tokens=common_prefix(previous,tokens))
            common.dump(folder/f'request_{step:02d}.json', request_record)
            row = streaming_call(client, messages, seed=int(story_id)*100+step)
            row.update(index=(int(story_id)-1)*13+step-1, story_id=story_id, step=step,
                kind='planning' if step<=5 else 'chapter', chapter=step-5 if step>5 else None,
                expected_prompt_tokens=len(tokens), common_previous_prompt_tokens=common_prefix(previous,tokens),
                prompt_sha256=hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest())
            if 'error' not in row and row['prompt_tokens'] != len(tokens):
                row['error'] = f"Tokenizer mismatch {row['prompt_tokens']} != {len(tokens)}"
            calls.append(row); common.dump(folder/'calls.json', calls)
            if 'error' in row: raise RuntimeError(row['error'])
            outputs[str(step)] = row['output'].strip()
            common.dump(folder/'outputs.json', outputs)
            print(f"story{story_id} step{step}/13 input={row['prompt_tokens']} output={row['completion_tokens']} "
                  f"words={row['output_words']} reused={row['cached_tokens']} TTFT={row['ttft_ms']:.1f}ms", flush=True)
            previous = tokens
        common.dump(folder/'status.json', dict(status='completed', requests=13))
    except BaseException as exc:
        stop.set(); common.dump(folder/'status.json', dict(status='failed', error=repr(exc), requests=len(calls)))
        raise


def report(root):
    assert read(root/'status.json')['status']=='completed'
    calls = sorted([c for p in (root/'stories').glob('*/calls.json') for c in read(p)], key=lambda c:c['index'])
    assert len(calls)==52 and not any('error' in c for c in calls)
    events = read_events(root/'server'); selected = select_events(events,calls)
    attributed = attribute_recomputation(calls, selected, 128)
    by_index = {r['index']:r for r in attributed}
    rows=[]
    for c in calls:
        row=dict(c); row.pop('output'); row.pop('sampling'); row.pop('usage')
        row.update(by_index[c['index']]); rows.append(row)
    common.dump(root/'requests_summary.json',rows)
    window = read(root/'phase.json')
    scoped = [e for e in events if window['start_ns']<=e['time_ns']<=window['end_ns']]
    samples = [e['used_bytes']/2**30 for e in scoped if e['event']=='occupancy_sample']
    candidate = sum(e['queried_chunks'] for e in selected if e['event']=='tier_lookup' and e['tier']=='dram')
    dram = sum(r['dram_lookup_chunks'] for r in rows); daos = sum(r['daos_lookup_chunks'] for r in rows)
    issues = [r['index'] for r in rows if r['attribution_issues'] or r['unattributed_shortfall_tokens']]
    summary=dict(requests=len(rows),stories=4,steps_per_story=13,
        elapsed_seconds=(window['end_ns']-window['start_ns'])/1e9,
        input_tokens=stats(r['prompt_tokens'] for r in rows),output_tokens=stats(r['completion_tokens'] for r in rows),
        chapter_output_tokens=stats(r['completion_tokens'] for r in rows if r['kind']=='chapter'),
        chapter_output_words=stats(r['output_words'] for r in rows if r['kind']=='chapter'),
        ttft_ms=stats(r['ttft_ms'] for r in rows),
        cached_tokens=sum(r['cached_tokens'] for r in rows),total_input_tokens=sum(r['prompt_tokens'] for r in rows),
        hit_requests=sum(r['cached_tokens']>0 for r in rows),
        first_turn_cached=[r['cached_tokens'] for r in rows if r['step']==1],
        dram_lookup_chunks=dram,daos_lookup_chunks=daos,lookup_candidates=candidate,
        dram_hit_pct=100*dram/candidate,daos_hit_pct=100*daos/candidate,
        mean_staging_gib=statistics.mean(samples),peak_staging_gib=max(e['used_bytes'] for e in scoped)/2**30,
        dram_staged_chunks=sum(e['staged_chunks'] for e in selected if e['event']=='cpu_get_ready'),
        capacity_recomputed_tokens=None if issues else sum(r['capacity_recomputed_tokens'] for r in rows),
        attribution_issue_indices=issues,
        length_capped_requests=sum(r['finish_reason']=='length' for r in rows),
        short_response_requests=sum(r['upstream_short_response'] for r in rows))
    common.dump(root/'summary.json',summary)
    timeline(root,events,window['start_ns'],window['end_ns'],8)
    lines=['# EQ-Bench Longform cold-start 예비 실험','',
        'Qwen3-14B BF16 · 독립 이야기4개 동시 진행 · 이야기당13단계 · DRAM/staging 각8GiB · 청크128.',
        'DRAM/DAOS 프리페치ON, 복사 작업자1, 취소/조기 준비 알림OFF. vLLM 내장 prefix cache OFF.',
        '새 프로세스·빈 DRAM·새 DAOS namespace에서 한 번만 실행. warm 재실행과 사전 캐시 채우기 없음.',
        '공식 첫4개 이야기와 준비5단계/장 작성8단계 프롬프트를 유지했다. 각 요청에 실제 이전 답변을 누적했다.',
        '공식 temperature0.7/min_p0.1/max_tokens4000, 로컬 top_p1/top_k비활성/고정 요청 seed, thinking OFF.',
        'EOS 억제·최소 출력 강제·입력 padding·중간 캐시 drain 없음. 요청/짧은 출력 재시도 및 외부 품질 채점은 생략.',
        '따라서 공식 전체 품질 평가가 아니라 생성 워크로드를 이용한 캐시·길이 예비 실험이다.','',
        f"- 본문32개 출력 평균: {summary['chapter_output_tokens']['mean']:.1f}토큰 / {summary['chapter_output_words']['mean']:.1f}단어.",
        f"- 실제 재사용이 있었던 요청: {summary['hit_requests']}/52. 각 이야기 첫 요청 재사용 토큰: {summary['first_turn_cached']}.",
        f"- 조회 후보 청크 기준 DRAM hit {summary['dram_hit_pct']:.2f}%, DAOS hit {summary['daos_hit_pct']:.2f}%.",
        f"- staging 평균 {summary['mean_staging_gib']:.3f}GiB, 최대 {summary['peak_staging_gib']:.3f}GiB.",
        f"- 길이 상한 종료 {summary['length_capped_requests']}회. 원본의 짧은 출력 조건(<500자) {summary['short_response_requests']}회(재시도하지 않음).",'',
        '|단계|내용|평균 입력 토큰|평균 출력 토큰|재사용 토큰 비율|DRAM hit 청크|DAOS hit 청크|',
        '|---:|---|---:|---:|---:|---:|---:|']
    for step in range(1,14):
        part=[r for r in rows if r['step']==step]; total=sum(r['prompt_tokens'] for r in part)
        lines.append(f"|{step}|{'준비' if step<=5 else str(step-5)+'장'}|{statistics.mean(r['prompt_tokens'] for r in part):.0f}|{statistics.mean(r['completion_tokens'] for r in part):.0f}|{100*sum(r['cached_tokens'] for r in part)/total:.1f}%|{sum(r['dram_lookup_chunks'] for r in part)}|{sum(r['daos_lookup_chunks'] for r in part)}|")
    lines += ['', 'hit는 조회 시점의 존재 확인이고 실제 토큰 재사용과 구분한다. DRAM/DAOS 비율을 강제로 맞추지 않았다.',
        '출력 길이와 처리 순서가 달라지는 실제 다중 턴 실행이며 ON/OFF 성능 비교 또는 통계적 결론이 아니다.',
        'staging 그래프 첫 점은 시작 순간이 아니라 첫2초 구간 통계다. 입력은 대화가 진행되면서 누적된다.', '',
        '[시간 그래프](staging_hits.png) · [전체 수치](summary.json) · [요청별 수치](requests_summary.json)', '',
        '실제 입력/출력은 stories 하위에 저장했다. 실행 전후 용량과 환경, 원본 버전 및 코드 스냅샷도 보존했다.']
    (root/'RESULT_KO.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--port',type=int,default=8017)
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--report-only',action='store_true')
    a=p.parse_args(); root=a.output.resolve()
    if a.report_only: report(root); return
    root.mkdir(parents=True,exist_ok=False)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=UPSTREAM,text=True).strip()
    stories=read(UPSTREAM/'data/longform_creative_writing_prompts_minimalist.json')
    stories={k:stories[k] for k in ('1','2','3','4')}
    templates=load_templates(UPSTREAM/'data')
    common.dump(root/'stories.json',stories); common.dump(root/'templates.json',templates)
    a.model='Qwen/Qwen3-14B'; a.max_model_len=32768
    namespace='minji-eqbench-pilot-'+uuid.uuid4().hex
    cfg=make_config(True,namespace,cpu_gib=8,staging_gib=8)
    cfg['extra_config'].update({'storage_plugin.daosgds.module_path':'lmcache_daos.capacity_probe_backend',
        'storage_plugin.daosgds.class_name':'CapacityProbeBackend','daosgds.dram_prefetch_workers':1,
        'daosgds.dram_prefetch_cancel_queued':False,'daosgds.dram_prefetch_early_ready':False})
    case=root/'server'; case.mkdir(); (case/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    files=['eqbench_longform_pilot.py','cold_warm_prefetch.py','discovery_fixed_replay.py','discovery_rolling_replay.py',
        'staging_mixed_pressure.py','compare_e2e.py','run_vllm.sh','libdaosgdr.so',
        'prefetch_capacity_sweep.py','report_capacity_matrix.py','report_cold_warm_prefetch.py',
        'report_prefetch_timing.py','report_prefetch_capacity_sweep.py','analyze_discovery_staging.py']
    files += [str(f.relative_to(common.ROOT)) for f in sorted((common.ROOT/'lmcache_daos').glob('*.py'))]
    hashes={}
    for name in files:
        dest=root/'executed_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(common.ROOT/name,dest);hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    shutil.copytree(UPSTREAM/'data',root/'upstream_data')
    common.dump(root/'plan.json',dict(upstream_commit=COMMIT,story_ids=list(stories),stories=4,requests=52,
        concurrency=4,max_num_seqs=16,model=a.model,max_model_len=32768,cpu_gib=8,staging_gib=8,
        prefetch=True,retries=0,quality_judging=False,source_sha256=hashes,
        notes=['Normal EOS. Abort on context overflow rather than truncate or change original tasks.',
               '4 independent conversation chains, no barrier/drain between subsequent turns.']))
    os.environ['HF_HOME']='/home/hf/hf_cache'
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);assert len(tok)>100000
    if a.dry_run:
        print({k:len(tok.apply_chat_template(build_messages(s,templates,{},1),tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)) for k,s in stories.items()})
        common.dump(root/'status.json',dict(status='dry_run'));return
    os.environ['DAOS_GDS_PREFETCH_TIMING']='1'
    try:
        q=pool_query();common.dump(root/'pool_before.json',q);space=check_space(q)
        # Upper bound for 4 histories, 13x4000 outputs + prompt overhead, with safety headroom.
        assert space['free']>300_000_000_000
        r=subprocess.run([str(common.ROOT/'run_vllm.sh'),str(common.ROOT/'venv/bin/python3'),
            'tests/object_gpu_roundtrip.py','--size-mib','20'],cwd=common.ROOT,
            env=dict(os.environ,DAOSGDS_TRANSPORT='object'),capture_output=True,text=True,timeout=120)
        (root/'storage_preflight.log').write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('DAOS preflight failed')
        common.dump(root/'status.json',dict(status='running'))
        with server(a,case/'config.yaml',case) as client:
            common.dump(root/'initial_sample.json',await_empty(case))
            health=LogHealth(case/'server.log');health()
            (root/'metrics_before.txt').write_text(client.get('/metrics').text)
            phase=dict(start_ns=time.time_ns());common.dump(root/'phase.json',phase)
            stop=threading.Event();barrier=threading.Barrier(4)
            with ThreadPoolExecutor(max_workers=4) as pool:
                pending={pool.submit(run_story,client,tok,k,s,templates,root,barrier,stop) for k,s in stories.items()}
                try:
                    while pending:
                        done,pending=wait(pending,timeout=.2,return_when=FIRST_COMPLETED)
                        health()
                        for f in done:f.result()
                except BaseException:
                    stop.set();raise
            phase['end_ns']=time.time_ns();common.dump(root/'phase.json',phase)
            (root/'metrics_after.txt').write_text(client.get('/metrics').text)
            common.dump(root/'final_sample.json',drain(case,health))
        common.dump(root/'pool_after.json',pool_query())
        common.dump(root/'status.json',dict(status='completed',requests=52))
    except BaseException as exc:
        common.dump(root/'status.json',dict(status='failed',error=repr(exc)));raise
    report(root)


if __name__=='__main__':main()
