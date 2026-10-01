#!/usr/bin/env python3
"""Pinned L-Eval document-QA subset; cold-start, fixed inputs, rolling clients.

Performance adaptation, not the official full L-Eval quality evaluation.
No truncation, repeated questions, forced output length or cache warmup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import urllib.request
import uuid

import yaml
import eqbench_longform_matrix as matrix
from discovery_fixed_replay import replay_one
from discovery_rolling_replay import rolling_requests

ROOT = matrix.common.ROOT
COMMIT = 'cd34b050269148aed75acbbe4a599873ad0f37e9'
DATASETS = ('scientific_qa', 'financial_qa', 'legal_contract_qa', 'narrative_qa')
MODEL = 'Qwen/Qwen3-14B'
MAX_OUTPUT = 1024
MAX_CONTEXT = 32768
HEADER = 'Read the following document and answer the question using the document.\n\nDocument:\n'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    result = []
    # Reverse arm order on alternate repeats; never select by performance.
    for repeat in range(1, 4):
        for concurrency in (8, 16):
            for enabled in ((False, True) if repeat % 2 else (True, False)):
                mode = 'wait' if enabled else 'off'
                result.append(dict(name=f'c{concurrency}_r{repeat}_{mode}', repeat=repeat,
                    concurrency=concurrency, cpu_gib=8, staging_gib=8, mode=mode,
                    prefetch=enabled, cancel_queued=False, early_ready=enabled))
    return result


def cases_from_plan(plan):
    assert plan['workload_kind'] == 'leval_document_qa'
    assert plan['cases'] == cases()
    return plan['cases']


def prompt_for(document, question):
    # Questions follow the document, preserving its exact reusable prefix.
    return HEADER + document + '\n\nQuestion:\n' + question + '\n\nAnswer:'


def tokenize(tok, prompt):
    return tok.apply_chat_template([dict(role='user', content=prompt)], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)


def distinct_questions(row):
    assert len(row['instructions']) == len(row['outputs']), 'Question/reference mismatch'
    seen = set(); pairs = []
    for i, (question, reference) in enumerate(zip(row['instructions'], row['outputs'])):
        if question in seen: continue
        seen.add(question)
        pairs.append((i, question, reference))
    return pairs


def interleave(documents, seed=20260928):
    # Random interleaving, preserving the original order of each document's Qs.
    # Fixed admission list, not fixed wall-clock arrivals (closed-loop clients).
    slots = [i for i, d in enumerate(documents) for _ in d['questions']]
    random.Random(seed).shuffle(slots)
    next_q = [0]*len(documents)
    records = []
    for i in slots:
        d = documents[i]; q = next_q[i]; next_q[i] += 1
        item = d['questions'][q]
        prompt = prompt_for(d['text'], item['question'])
        records.append(dict(index=len(records), document_id=d['document_id'], question_index=q,
            prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            expected_prompt_tokens=item['prompt_tokens']))
    return records


def snapshot(root):
    names = ['leval_prefetch.py', 'supervise_eqbench_matrix.py', 'eqbench_longform_matrix.py',
        'eqbench_longform_pilot.py', 'discovery_rolling_replay.py', 'discovery_fixed_replay.py',
        'staging_mixed_pressure.py', 'cold_warm_prefetch.py', 'compare_e2e.py',
        'report_prefetch_timing.py', 'report_capacity_matrix.py', 'report_cold_warm_prefetch.py',
        'report_prefetch_capacity_sweep.py', 'analyze_discovery_staging.py',
        'prefetch_capacity_sweep.py', 'cleanup_experiment_cache.py', 'run_vllm.sh',
        'libdaosgdr.c', 'libdaosgdr.so', 'list_experiment_dkeys',
        'lmcache_config_daosgds_async_dram.yaml', 'tests/test_leval_prefetch.py']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'lmcache_daos').glob('*.py'))]
    hashes = {}
    for name in names:
        src = ROOT/name; dst = root/'executed_sources'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst); hashes[name] = digest(dst)
    return hashes


def prepare(root, tok):
    root.mkdir(exist_ok=False); data = root/'source_data'; data.mkdir()
    candidates = []; audit = []; sources = []; seen_documents = set()
    for name in DATASETS:
        url = f'https://raw.githubusercontent.com/OpenLMLab/LEval/{COMMIT}/LEval-data/Open-ended-tasks/{name}.jsonl'
        path = data/(name+'.jsonl')
        with urllib.request.urlopen(url, timeout=90) as response:
            path.write_bytes(response.read())
        sources.append(dict(dataset=name, url=url, sha256=digest(path)))
        for i, line in enumerate(path.read_text().splitlines()):
            if not line.strip(): continue
            row = json.loads(line); text = row['input']
            length = len(tok.encode(text, add_special_tokens=False, verbose=False))
            reasons = []
            if not 4096 <= length <= 8192: reasons.append('document_outside_4k_8k')
            pairs = distinct_questions(row)
            if len(pairs) < 4: reasons.append('fewer_than_four_distinct_questions')
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if text_hash in seen_documents: reasons.append('duplicate_document')
            info = dict(document_id=f'{name}:{i}', document_tokens=length,
                available_questions=len(row['instructions']), distinct_questions=len(pairs), exclusion_reasons=reasons)
            audit.append(info)
            if reasons: continue
            seen_documents.add(text_hash)
            questions = []
            for original_index, q, ref in pairs[:4]:
                n = len(tokenize(tok, prompt_for(text, q)))
                assert n+MAX_OUTPUT <= MAX_CONTEXT-4096, 'Input budget lacks 4K headroom'
                questions.append(dict(question=q, reference=ref, prompt_tokens=n, original_question_index=original_index))
            assert len(questions) == 4 and len({x['question'] for x in questions}) == 4
            candidates.append(dict(**info, text=text, questions=questions))
    matrix.common.dump(root/'selection_audit.json', audit)
    matrix.common.dump(root/'sources.json', sources)
    # All eligible documents, not a performance-selected subset.
    assert len(candidates) >= 12, f'Only {len(candidates)} eligible documents; review selection before running'
    hashes = [hashlib.sha256(d['text'].encode()).hexdigest() for d in candidates]
    assert len(hashes) == len(set(hashes)), 'Duplicate documents'
    estimated_gib = sum(d['document_tokens'] for d in candidates)*163840/2**30
    assert estimated_gib > 12, 'Selected document KV working set too small'
    records = interleave(candidates)
    matrix.common.dump(root/'documents.json', candidates)
    matrix.common.dump(root/'requests.json', records)
    plan = dict(workload_kind='leval_document_qa', cases=cases(), model=MODEL,
        upstream_commit=COMMIT, requests_per_case=len(records), total_requests=len(records)*len(cases()),
        documents=len(candidates), questions_per_document=4,
        document_token_range=[min(d['document_tokens'] for d in candidates), max(d['document_tokens'] for d in candidates)],
        estimated_document_kv_gib=estimated_gib, max_model_len=MAX_CONTEXT,
        cpu_gib=8, staging_gib=8, chunk_tokens=128, max_num_seqs=16,
        generation=dict(temperature=0, seed=0, max_tokens=MAX_OUTPUT, enable_thinking=False, stop=[]),
        requests_sha256=digest(root/'requests.json'), documents_sha256=digest(root/'documents.json'),
        source_sha256=snapshot(root), cleanup_completed_case_kv=True,
        cleanup_authority='User authorized experiment-only DAOS KV cleanup; preserve unrelated keys and all logs.',
        notes=['Fixed shuffled admission list, rolling closed-loop concurrency; wall-clock arrivals depend on completions.',
            'First four distinct original questions per eligible document; original duplicate questions skipped, no truncation or replication.',
            'Fresh process/DRAM/staging and private DAOS namespace for every arm and repeat; no prefill warmup.',
            'EOS enabled; output tokens may differ. Not official L-Eval quality evaluation.',
            'Question follows document; answers are not appended to future requests. Natural prefix reuse.',
            'KV size is estimated from current Qwen3-14B BF16 geometry, not measured resident memory.'])
    matrix.common.dump(root/'plan.json', plan)
    matrix.common.dump(root/'status.json', dict(status='dry_run', completed=[]))
    report(root)
    print(json.dumps({k:plan[k] for k in ['documents','requests_per_case','total_requests',
        'document_token_range','estimated_document_kv_gib']}, indent=2), flush=True)


def report(root):
    plan = matrix.read(root/'plan.json')
    summaries = [matrix.read(root/s['name']/'summary.json') for s in cases_from_plan(plan)
        if (root/s['name']/'summary.json').exists() and
        matrix.read(root/s['name']/'status.json')['status'] == 'completed']
    matrix.common.dump(root/'summary.json', summaries)
    lines = ['# L-Eval 문서 QA 프리페치 비교', '',
        f"원본 문서 {plan['documents']}개 × 서로 다른 질문 4개 = 조건당 {plan['requests_per_case']}회. 12조건(8/16 동시 요청 × OFF/ON × 3회).",
        'DRAM 8GiB / staging 8GiB / Qwen3-14B / 청크128. 모든 조건 cold 시작, 별도 warm 반복 없음.',
        '고정 입력·순서, 완료 시 다음 요청을 넣는 rolling 방식. 도착 시각과 실제 생성 길이는 고정하지 않는다.',
        '원문 4K~8K 필터, 원본 중복 질문을 제외한 앞의 질문4개 선택, 고정 seed 순서 혼합. 원문 절단·질문 복제·출력 길이 강제 없음.',
        '공식 전체 L-Eval/품질 채점이 아니라 문서 QA 기반 성능 실험. 실제 DRAM/DAOS 비율은 결과값이다.', '',
        '|조건|시간(s)|평균 TTFT(ms)|DRAM hit %|DAOS hit %|staging 최대 GiB|용량 부족 재계산 토큰|생성 토큰|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"|[{s['name']}]({s['name']}/staging_hits.png)|{s['elapsed_seconds']:.2f}|{s['ttft_ms']['mean']:.2f}|{s['dram_hit_pct']:.2f}|{s['daos_hit_pct']:.2f}|{s['peak_staging_gib']:.3f}|{s['capacity_recomputed_tokens']}|{s['total_output_tokens']}|")
    (root/'RESULT_KO.md').write_text('\n'.join(lines)+'\n')


def finish_report(case, spec):
    summary=matrix.report_case(case, spec)
    # Shared reporter's legacy story count is unrelated to this document workload.
    summary.pop('stories', None)
    summary['documents']=matrix.read(case.parent/'plan.json')['documents']
    summary['questions_per_document']=4
    matrix.common.dump(case/'summary.json',summary)
    # Host intervals, not pure DMA durations or proof of compute overlap.
    from report_prefetch_timing import join, stats
    calls = matrix.read(case/'stories/requests/calls.json')
    events = matrix.pilot.select_events(matrix.pilot.read_events(case/'server'), calls)
    try:
        rows = join(calls, events)
        matrix.common.dump(case/'timing_requests.json', rows)
        keys = ['ttft_ms','http_to_retrieve_start_ms','retrieve_ms','queue_ms','copy_ms',
                'cpu_ready_to_retrieve_ms','worker_end_to_retrieve_ms']
        matrix.common.dump(case/'timing_summary.json', {k:stats(r.get(k) for r in rows) for k in keys})
    except (AssertionError, ValueError) as exc:
        matrix.common.dump(case/'timing_analysis_error.json', dict(error=repr(exc),
            note='Timing attribution invalid; do not infer overlap. Raw trace preserved.'))


def run(root):
    import supervise_eqbench_matrix as supervisor
    plan = matrix.read(root/'plan.json'); specs = cases_from_plan(plan)
    assert digest(root/'requests.json') == plan['requests_sha256']
    assert digest(root/'documents.json') == plan['documents_sha256']
    for name, sha in plan['source_sha256'].items():
        assert digest(ROOT/name) == sha, f'Source changed since preparation: {name}'
    records = matrix.read(root/'requests.json'); completed = []
    os.environ['HF_HOME']='/home/hf/hf_cache'; os.environ['DAOS_GDS_PREFETCH_TIMING']='1'
    a = argparse.Namespace(model=MODEL,max_model_len=MAX_CONTEXT,port=8017)
    for spec in specs:
        case = root/spec['name']
        matrix.common.dump(root/'status.json', dict(status='running', current=spec['name'], completed=completed))
        try:
            if case.exists():
                assert 'end_ns' in matrix.read(case/'phase.json') and (case/'final_sample.json').exists()
                finish_report(case,spec)
            else:
                case.mkdir(); srv=case/'server';srv.mkdir()
                namespace='minji-leval-'+uuid.uuid4().hex
                matrix.common.dump(case/'identity.json',dict(namespace=namespace+':'))
                cfg=matrix.config_for(spec,namespace)
                (srv/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
                matrix.common.dump(case/'status.json',dict(status='running'))
                q=matrix.pool_query();matrix.common.dump(case/'pool_before.json',q);matrix.check_space(q)
                print(f"CASE {len(completed)+1}/{len(specs)} {spec['name']}",flush=True)
                with matrix.pilot.server(a,srv/'config.yaml',srv) as client:
                    matrix.common.dump(case/'initial_sample.json',matrix.pilot.await_empty(srv))
                    health=matrix.pilot.LogHealth(srv/'server.log');health()
                    calls=[]; folder=case/'stories/requests';folder.mkdir(parents=True)
                    phase=dict(start_ns=time.time_ns());matrix.common.dump(case/'phase.json',phase)
                    def invoke(record,barrier):
                        row=replay_one(client,record,barrier,max_tokens=MAX_OUTPUT,stop=())
                        row.update(document_id=record['document_id'],question_index=record['question_index'],
                            expected_prompt_tokens=record['expected_prompt_tokens'])
                        if 'error' not in row and row['prompt_tokens'] != record['expected_prompt_tokens']:
                            row['error']='Tokenizer mismatch'
                        return row
                    def save(row):
                        calls.append(row)
                        matrix.common.dump(folder/'calls.json',sorted(calls,key=lambda c:c['index']))
                        print(f"{spec['name']} {len(calls)}/{len(records)} completed",flush=True)
                    rolling_requests(records,spec['concurrency'],invoke,save,health)
                    assert len(calls)==len(records) and not any('error' in c for c in calls)
                    phase['end_ns']=time.time_ns();matrix.common.dump(case/'phase.json',phase)
                    (case/'metrics_after.txt').write_text(client.get('/metrics').text)
                    matrix.common.dump(case/'final_sample.json',matrix.pilot.drain(srv,health))
                matrix.common.dump(case/'pool_after.json',matrix.pool_query())
                finish_report(case,spec)
            matrix.common.dump(case/'status.json',dict(status='completed',requests=len(records)))
            report(root)
            supervisor.cleanup_failed(case)
            completed.append(spec['name'])
            print(f"COMPLETE {len(completed)}/{len(specs)} {spec['name']} KV cleaned",flush=True)
        except BaseException as exc:
            if case.exists() and not (case/'status.json').exists():
                matrix.common.dump(case/'status.json',dict(status='failed',error=repr(exc)))
            elif case.exists() and matrix.read(case/'status.json')['status']!='completed':
                matrix.common.dump(case/'status.json',dict(status='failed',error=repr(exc)))
            matrix.common.dump(root/'status.json',dict(status='failed',current=spec['name'],completed=completed,error=repr(exc)))
            raise
    matrix.common.dump(root/'status.json',dict(status='completed',completed=completed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    assert root.parent==ROOT and root.name.startswith('leval_prefetch_')
    if args.prepare:
        os.environ['HF_HOME']='/home/hf/hf_cache'
        from transformers import AutoTokenizer
        prepare(root,AutoTokenizer.from_pretrained(MODEL,local_files_only=True))
    else:
        assert args.resume
        run(root)


if __name__=='__main__':main()
