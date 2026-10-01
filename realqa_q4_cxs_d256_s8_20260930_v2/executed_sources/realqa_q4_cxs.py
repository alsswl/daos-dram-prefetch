#!/usr/bin/env python3
"""Audited upstream real-multi-round-qa CxS, actual generated multi-turn history.

Keep upstream ChatSession/run_group/run_turn; add deterministic book selection,
context validation, passive stream telemetry, fail-fast errors and DAOS probes.
"""
import argparse
import asyncio
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from types import SimpleNamespace

import yaml
import sharegpt_async_threeway as three
from report_cold_warm_prefetch import select_events
from report_prefetch_timing import stats
from report_staging_fine import bins, header, axes
from report_capacity_matrix import save_chart

ROOT, base, early = three.ROOT, three.base, three.early
MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
KV_BYTES = 36 * 2 * 8 * 128 * 2
UPSTREAM = ROOT/'LMBenchmark/real-multi-round-qa/multi-round-qa.py'


def dump(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def upstream():
    spec = importlib.util.spec_from_file_location('realqa_upstream', UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.model_max_length = 10**9
    return tok


def prepare(root):
    assert root.parent == ROOT and not root.exists()
    root.mkdir()
    (root/'documents').mkdir()
    dump(root/'status.json', {'status':'preparing'})
    tok, up = tokenizer(), upstream()
    # Reuse already downloaded Gutenberg originals; no invented padding.
    raw_dirs = [ROOT/'realqa_gutenberg_16k_20260929/raw',
                ROOT/'realqa_gutenberg_64k_20260929/raw']
    raws = {p.name:p for d in reversed(raw_dirs) for p in d.glob('*.txt')}
    docs = []
    for name in sorted(raws, key=lambda s:int(s.split('.')[0])):
        raw = raws[name]
        text = raw.read_text(encoding='utf-8-sig')
        # Only need a prefix; avoid tokenizing multi-megabyte complete novels.
        tokens = tok.encode(text[:400000], add_special_tokens=False)
        if len(tokens) < 28672:
            continue
        prefix = tok.decode(tokens[:28672])
        target = root/'documents'/name
        target.write_text(prefix)
        prompt = up.FIRST_PROMPT.format(prefix)
        n = len(tok.apply_chat_template([{'role':'user','content':prompt}],
            tokenize=True, add_generation_prompt=True, return_dict=False))
        assert 28000 <= n < 29000
        docs.append(dict(file=name, raw_file=str(raw), raw_sha256=base.digest(raw),
                         sha256=base.digest(target), initial_prompt_tokens=n))
        if len(docs) % 25 == 0:
            print(f'Prepared {len(docs)} eligible 28K books', flush=True)
        if len(docs) == 200:
            break
    assert len(docs) >= 100, 'Insufficient real books; do not substitute artificial text'
    rng = random.Random(20260930)
    assignments = [dict(index=i, group=i//10, slot=i%10, **rng.choice(docs)) for i in range(80)]
    unique = {r['file']:r for r in assignments}
    # Upper bound: no sharing between books, all generated/question suffixes stored.
    first_gib = sum(r['initial_prompt_tokens'] for r in unique.values())*KV_BYTES/2**30
    required = first_gib+80*6*(256+128)*KV_BYTES/2**30
    plan = dict(model=MODEL, cpu_gib=256, staging_gib=8, concurrency=8,
        session_depth=10, sessions=80, num_rounds=5, turns_per_session=6,
        expected_requests=480, answer_len=256, max_model_len=32768, max_num_seqs=8,
        chunk_tokens=128, seed=20260930, document_tokens=28672,
        cases=[dict(name='c8_s10_none',policy='none')], source_sha256={},
        upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM.parent,text=True).strip(),
        capacity=dict(eligible_books=len(docs), unique_selected_books=len(unique),
            initial_unique_kv_upper_gib=first_gib, stored_kv_upper_gib=required),
        notes=['Upstream real-multi-round-qa fixed CxS scheduler, prompts and actual answer feedback.',
            'Local adaptations: fixed seeded selection from sorted books, 28K document prefixes, telemetry and fail-fast bounds.',
            'C8 groups each cycle through S10 sessions, one outstanding request/group. No global wave barrier.',
            'num_rounds=5 means initial summary plus FIVE follow-ups: six model calls/session.',
            'Fresh process and private empty DAOS namespace at start only. Natural within-run reuse; no warm replay.',
            'Both payload prefetch tiers OFF; async metadata-first lookup and 1ms backoff unchanged.',
            'GPU-direct stores, asynchronous DRAM mirror and read promotion enabled; native vLLM prefix cache OFF.',
            'Upstream TTFT includes pre-request metrics GET; also report HTTP-send-to-first-content TTFT separately.',
            'Actual input length checked outside request timer; no silent history truncation or question replacement.',
            'Staging includes demand reads AND writes, not GPU compute utilization. No answer-quality scoring.'])
    plan['served_model_name'] = 'comparison-model'
    dump(root/'documents.json',docs)
    dump(root/'sessions.json',assignments)
    files = set(three.read(three.BASELINE/'plan.json')['source_sha256'])
    files.update(['realqa_q4_cxs.py','sharegpt_async_threeway.py',
                  'lmcache_daos/tier_payload_backend.py','cleanup_q4_32k_namespace.py',
                  str(UPSTREAM.relative_to(ROOT)), 'report_staging_fine.py',
                  'tests/test_realqa_q4_cxs.py'])
    for name in sorted(files):
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
        plan['source_sha256'][name] = base.digest(dest)
    dump(root/'plan.json',plan)
    dump(root/'status.json',dict(status='prepared',capacity=plan['capacity']))
    print(json.dumps(plan['capacity'],indent=2),flush=True)


async def workload(root,case,health):
    plan, assignments, tok, up = read(root/'plan.json'), read(root/'sessions.json'), tokenizer(), upstream()
    original_session, original_turn = up.ChatSession, up.run_turn
    cursor = iter(assignments)
    completed = []
    (case/'metrics').mkdir()
    (case/'histories').mkdir()

    class FixedSession(original_session):
        def _load_random_file(self):
            self.assignment = next(cursor)
            return up.FIRST_PROMPT.format((root/'documents'/self.assignment['file']).read_text())

    async def observed_turn(session, client, http_client, base_url, gap):
        health()
        messages = session.messages+[dict(role='user',content=session.get_next_prompt())]
        ids = await asyncio.to_thread(tok.apply_chat_template,messages,
            tokenize=True,add_generation_prompt=True,return_dict=False)
        assert len(ids)+session.answer_len <= plan['max_model_len'], 'Context budget exceeded; stop without truncation'
        row = dict(index=session.assignment['index']*6+session.turns,
            session_index=session.assignment['index'], session_id=session.session_id,
            book=session.assignment['file'],turn=session.turns,
            input_tokens_validated=len(ids),
            messages_sha256=hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest())

        async def observed_create(**kwargs):
            row['http_start_ns'] = time.time_ns()
            stream = await client.chat.completions.create(**kwargs)

            async def chunks():
                try:
                    async for chunk in stream:
                        row['server_request_id'] = chunk.id
                        if chunk.choices and chunk.choices[0].delta.content and 'first_token_ns' not in row:
                            row['first_token_ns'] = time.time_ns()
                        if chunk.usage:
                            row['usage'] = chunk.usage.model_dump()
                            details = row['usage'].get('prompt_tokens_details') or {}
                            row['cached_tokens'] = details.get('cached_tokens',0)
                        yield chunk
                finally:
                    await stream.close()
            return chunks()

        proxy = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=observed_create)))
        result = await original_turn(session,proxy,http_client,base_url,gap)
        row['end_ns'] = time.time_ns()
        row.update(asdict(result))
        metric_text = row.pop('metrics')
        (case/'metrics'/f'{row["index"]:04d}.txt').write_text(metric_text)
        row['ttft_http_ms'] = ((row['first_token_ns']-row['http_start_ns'])/1e6
            if 'first_token_ns' in row else None)
        # One event loop: each append is atomic with respect to the other coroutines.
        with (case/'calls.jsonl').open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
        dump(case/'histories'/f'{session.assignment["index"]:03d}.json',session.messages)
        completed.append(row)
        dump(root/'status.json',dict(status='running',completed=len(completed),
            total=plan['expected_requests'],updated_ns=time.time_ns()))
        print(f'Completed {len(completed)}/480; session={row["session_index"]} turn={row["turn"]} '
              f'prompt={result.prompt_tokens} cached={row.get("cached_tokens")} ttft={row["ttft_http_ms"]}',flush=True)
        assert result.status == 'success', result.error
        assert result.prompt_tokens == len(ids), 'Client/server tokenization differs'
        health()
        return result

    up.ChatSession, up.run_turn = FixedSession, observed_turn
    args = SimpleNamespace(concurrent=8,session_depth=10,model='comparison-model',
        src_dir=str(root/'documents'),num_rounds=5,answer_len=256,
        base_url='http://127.0.0.1:8017',gap_between_requests=0,time=None,
        timeout=180,skip_ssl_verify=False)
    results = await up.run_all_concurrent(args)
    assert len(results)==480 and len(completed)==480
    assert len({(c['session_index'],c['turn']) for c in completed})==480
    dump(case/'calls.json',completed)
    # Metrics already stored separately; avoid embedding duplicate large texts.
    dump(case/'upstream_results.json',dict(params=vars(args),results=[
        {k:v for k,v in r.items() if k!='metrics'} for r in results]))


def report(root):
    case = root/read(root/'plan.json')['cases'][0]['name']
    events = base.read_events(case)
    window = read(case/'workload.json')
    calls = read(case/'calls.json')
    selected = select_events(events,calls)
    rows = []
    for label, cc in [('all',calls)]+[(str(i),[c for c in calls if c['turn']==i]) for i in range(6)]:
        ev = select_events(selected,cc)
        queried = sum(e['queried_chunks'] for e in ev if e['event']=='tier_lookup' and e['tier']=='dram')
        hits = {t:sum(e['hit_chunks'] for e in ev if e['event']=='tier_lookup' and e['tier']==t) for t in ('dram','daos')}
        inp = sum(c['prompt_tokens'] for c in cc)
        cached = sum(c['cached_tokens'] for c in cc)
        rows.append(dict(turn=label,requests=len(cc),input_tokens=inp,cached_tokens=cached,
            cached_token_pct=100*cached/inp,queried_chunks=queried,hit_chunks=hits,
            dram_candidate_pct=100*hits['dram']/queried if queried else 0,
            daos_candidate_pct=100*hits['daos']/queried if queried else 0,
            ttft_http_ms=stats([c['ttft_http_ms'] for c in cc]),
            ttft_upstream_ms=stats([c['ttft']*1000 for c in cc]),
            completion_tokens=sum(c['completion_tokens'] for c in cc)))
    duration = (window['end_ns']-window['start_ns'])/1e9
    fine = bins(events,window['start_ns'],window['end_ns'],100_000_000)
    dump(root/'occupancy_100ms.json',fine)
    peak = max(b['peak_gib'] for b in fine)
    mean = sum(b['time_weighted_mean_gib']*b['duration_seconds'] for b in fine)/duration
    dump(root/'summary.json',dict(duration_s=duration,by_turn=rows,
        staging_peak_gib=peak,staging_time_weighted_mean_gib=mean,
        staging_peak_pct=100*peak/8,staging_mean_pct=100*mean/8,
        hit_denominator='Initial DRAM-tier candidate chunks; DAOS counted over same denominator',
        initial=read(case/'initial_sample.json'),final=read(case/'final_sample.json')))
    base.timeline(case,events,window['start_ns'],window['end_ns'],8)
    shutil.copy2(case/'staging_hits.png',root/'staging_hits.png')
    svg = header(405,'Real multi-round QA | Qwen3-4B | C8 x S10 x 6 turns',
        'DRAM 256GiB / staging 8GiB / both prefetch OFF | blue=peak, green=time-weighted mean, 100ms bins')
    axes(svg,100,225,duration,'Staging occupancy (reads + writes), not GPU utilization')
    for key,color in [('peak_gib','#0072b2'),('time_weighted_mean_gib','#009e73')]:
        pts = ' '.join(f'{75+1125*b["start_seconds"]/duration:.2f},{100+225*(1-b[key]/8):.2f}' for b in fine)
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    svg += ['<text x="70" y="385">Fresh caches at start only; real answers feed subsequent turns; no artificial warm replay.</text></g></svg>']
    save_chart(root,'staging_100ms',svg)


def run(root):
    plan = read(root/'plan.json')
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,digest in plan['source_sha256'].items():
            assert base.digest(ROOT/name)==digest, f'Source changed: {name}'
        for doc in read(root/'documents.json'):
            assert base.digest(root/'documents'/doc['file'])==doc['sha256']
        assert early.idle_gpu(), 'GPU occupied; do not stop unrelated jobs'
        assert shutil.disk_usage(ROOT).free > 6*2**30
        mem = {s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
        assert mem['MemAvailable'] > 320*2**30
        case = root/plan['cases'][0]['name']
        case.mkdir()
        try:
            os.environ.update(DAOS_GDS_PREFETCH_TIMING='1',DAOS_LOOKUP_READY_RETURN='0',DAOS_LOOKUP_LOCK_PROBE='0')
            dump(root/'status.json',dict(status='starting',updated_ns=time.time_ns()))
            base.storage_guard(case,plan['capacity']['stored_kv_upper_gib'])
            cfg = three.config('none')
            (case/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
            args = SimpleNamespace(model=MODEL,max_model_len=32768,max_num_seqs=8,port=8017)
            with base.server(args,case/'config.yaml',case) as client:
                initial = base.await_empty(case)
                assert initial['used_bytes']==initial['cpu_hot_bytes']==initial['daos_puts']==0
                dump(case/'initial_sample.json',initial)
                health = base.LogHealth(case/'server.log')
                window = dict(start_ns=time.time_ns())
                dump(case/'workload.json',window)
                asyncio.run(workload(root,case,health))
                window['end_ns'] = time.time_ns()
                dump(case/'workload.json',window)
                final = base.drain(case,health,timeout=180)
                dump(case/'final_sample.json',final)
            three.validate_policy(case,'none')
            dump(case/'status.json',dict(status='completed',requests=480))
            report(root)
            dump(root/'status.json',dict(status='completed',requests=480,updated_ns=time.time_ns(),
                cache_cleanup='not performed; namespace retained for possible follow-up comparison'))
        except BaseException as exc:
            dump(root/'status.json',dict(status='failed',error=repr(exc),updated_ns=time.time_ns()))
            raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','report'])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    {'prepare':prepare,'run':run,'report':report}[args.action](args.output.resolve())
