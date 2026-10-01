#!/usr/bin/env python3
"""Natural 28-32Ki-token ShareGPT histories, Qwen3-4B, demand-only reads.

Separate diagnostic; does not change any completed experiment or backend.
One selected history per conversation, replayed cold then warm, not live agents.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import yaml
import sharegpt_async_threeway as three
from prepare_sharegpt_scale import prefix_chunks
from report_early_lookup import summarize_case
from report_capacity_matrix import save_chart

early, base = three.early, three.base
ROOT, read, dump = three.ROOT, three.read, three.dump
MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
KV_BYTES = 36 * 2 * 8 * 128 * 2


def prepare(root, count):
    from transformers import AutoTokenizer
    assert root.parent == ROOT and not root.exists()
    assert count >= 8
    root.mkdir()
    dump(root/'status.json', {'status': 'preparing'})
    sources = sorted((ROOT/'sharegpt_dataset_20260929').glob('*html_cleaned.json'))
    assert len(sources) == 2
    entries = []
    for source in sources:
        entries.extend(json.loads(source.read_text()))
    order = list(range(len(entries)))
    random.Random(20260930).shuffle(order)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    records, selected, seen = [], [], set()
    for scanned, source_index in enumerate(order, 1):
        entry = entries[source_index]
        messages = entry.get('conversations', [])
        if sum(len(m.get('value', '')) for m in messages) < 28672:
            continue
        history = []
        for pos in range(0, len(messages)-1, 2):
            user, answer = messages[pos:pos+2]
            if (user.get('from') != 'human' or answer.get('from') != 'gpt'
                    or not user.get('value') or not answer.get('value')):
                break
            history.append(user['value'])
            prompt = '\n'.join(history).strip()
            if len(prompt) > 480000:
                break
            if len(prompt) >= 28672:
                ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False,
                    return_dict=False)
                if len(ids) > 32768:
                    break
                if len(ids) >= 28672:
                    chunks = prefix_chunks(ids)
                    # Exclude duplicate/reposted histories without artificially padding.
                    if len(chunks-seen) < .95*len(chunks):
                        break
                    i = len(records)
                    records.append(dict(index=i, session=i, turn=0, prompt=prompt,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        expected_prompt_tokens=len(ids), max_tokens=256))
                    selected.append(dict(session=i, source_index=source_index,
                        source_id=entry.get('id'), original_turn=pos//2,
                        input_tokens=len(ids)))
                    seen.update(chunks)
                    print(f'selected={len(records)}/{count} scanned={scanned} tokens={len(ids)}', flush=True)
                    dump(root/'status.json',dict(status='preparing',selected=len(records),scanned=scanned))
                    break
            history.append(answer['value'])
        if len(records) == count:
            break
    assert len(records) == count, f'Only {len(records)} natural long histories; no padding fallback'
    dump(root/'requests.json', records)
    dump(root/'sessions.json', selected)
    tokens = [r['expected_prompt_tokens'] for r in records]
    unique = len(seen)*128*KV_BYTES/2**30
    cap = dict(requests=count, sessions=count, input_tokens_min=min(tokens),
        input_tokens_max=max(tokens), input_tokens_mean=sum(tokens)/count,
        input_tokens_sum=sum(tokens), unique_input_kv_gib=unique,
        conservative_stored_kv_gib=unique+count*(256+127)*KV_BYTES/2**30)
    spec = dict(name='c8_none', policy='none', prefetch=False, concurrency=8)
    plan = dict(model=MODEL, cpu_gib=256, staging_gib=8, max_model_len=33792,
        max_num_seqs=8, chunk_tokens=128, kv_bytes_per_token=KV_BYTES,
        phases=['cold','warm1'], cases=[spec], capacity=cap,
        request_sha256=base.digest(root/'requests.json'), source_sha256={},
        dataset_sources=[dict(path=str(p),sha256=base.digest(p)) for p in sources],
        notes=['Natural 28-32Ki-token histories; one late turn per conversation; no padding/truncation.',
               'History replay, NOT a full multi-turn agent benchmark; 256 output-token cap, EOS enabled.',
               'C8 rolling requests; cold then identical warm inputs in same process and retained caches.',
               'Both prefetch tiers OFF; async metadata lookup still ON, 1ms backoff.',
               'DRAM256/staging8GiB. GPU-direct stores, async DRAM mirror and read promotion remain enabled.',
               'Demand DAOS reads are batch-based, NOT a newly implemented streaming window.',
               'Cold namespace is empty; DAOS/OS physical caches are not flushed.',
               'Occupancy includes read and store buffers, not GPU compute utilization.'])
    files = set(read(three.BASELINE/'plan.json')['source_sha256'])
    files.update(['sharegpt_32k_demand.py','sharegpt_async_threeway.py','lmcache_daos/tier_payload_backend.py',
                  'cleanup_q4_32k_namespace.py'])
    for name in sorted(files):
        dst = root/'executed_sources'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dst)
        plan['source_sha256'][name] = base.digest(dst)
    dump(root/'capacity_estimate.json', cap)
    dump(root/'plan.json', plan)
    dump(root/'status.json',dict(status='prepared', capacity=cap))
    print(json.dumps(cap, indent=2), flush=True)


def run(root):
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(root/'plan.json')
        early.verify_sources(root, plan)
        assert early.idle_gpu(), 'GPU occupied; do not disturb other jobs'
        assert shutil.disk_usage(ROOT).free > 6*2**30
        mem = {s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
        assert mem['MemAvailable'] > 320*2**30
        os.environ.update(DAOS_GDS_PREFETCH_TIMING='1', DAOS_LOOKUP_READY_RETURN='0',
                          DAOS_LOOKUP_LOCK_PROBE='0')
        spec = plan['cases'][0]
        case = root/spec['name']
        case.mkdir()
        cap = plan['capacity']
        try:
            dump(root/'status.json', dict(status='starting',updated_ns=time.time_ns()))
            early.cw.wait_space(case, 2*cap['conservative_stored_kv_gib']-cap['unique_input_kv_gib'])
            cfg = three.config('none')
            (case/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            args = SimpleNamespace(model=MODEL, max_model_len=plan['max_model_len'],
                                   max_num_seqs=8, port=8017)
            with base.server(args, case/'config.yaml', case) as client:
                initial = base.await_empty(case)
                assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
                dump(case/'initial_sample.json', initial)
                dump(root/'status.json', dict(status='running', current=spec['name'],updated_ns=time.time_ns()))
                health = base.LogHealth(case/'server.log')
                health()
                final = early.cw.run_phases(case,read(root/'requests.json'),8,client,health,initial,
                    phases=plan['phases'],warm_extra_gib=cap['conservative_stored_kv_gib']-cap['unique_input_kv_gib'])
                dump(case/'final_sample.json', final)
            dump(case/'status.json',dict(status='completed',requests=2*cap['requests']))
            summarize_case(root,spec)
            three.validate_policy(case,'none')
            dump(root/'summary.json',read(case/'summary.json'))
            plot(root)
            with (case/'cleanup_q4.log').open('x') as log:
                command = [str(ROOT/'run_vllm.sh'), str(ROOT/'venv/bin/python3'),
                    str(ROOT/'cleanup_q4_32k_namespace.py'), '--case', str(case)]
                for flags in ([], ['--execute']):
                    subprocess.run(command+flags,cwd=ROOT,
                        env=dict(os.environ,DAOSGDS_TRANSPORT='object'),
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            dump(root/'status.json',dict(status='completed',updated_ns=time.time_ns()))
        except BaseException as exc:
            dump(root/'status.json',dict(status='failed',error=repr(exc),updated_ns=time.time_ns()))
            # Preserve incomplete traces and render available staging history as evidence.
            events = base.read_events(case)
            for folder in case.iterdir():
                if folder.is_dir() and (folder/'phase.json').exists():
                    phase = read(folder/'phase.json')
                    end = phase.get('end_ns',max((e['time_ns'] for e in events),default=0))
                    if events and end > phase['start_ns']:
                        base.timeline(folder,events,phase['start_ns'],end,8)
            raise


def plot(root):
    """Also usable while running; unfinished panels explicitly labelled partial."""
    plan = read(root/'plan.json')
    case = root/plan['cases'][0]['name']
    events = base.read_events(case)
    if not events:
        return
    panels = []
    for name in plan['phases']:
        folder = case/name
        if not (folder/'phase.json').exists():
            continue
        phase = read(folder/'phase.json')
        end = phase.get('end_ns',events[-1]['time_ns'])
        if end <= phase['start_ns']:
            continue
        base.timeline(folder,events,phase['start_ns'],end,8)
        panel = ET.fromstring((folder/'staging_hits.svg').read_text())
        panel.set('y',str(80+560*len(panels)))
        if not (folder/'status.json').exists() or read(folder/'status.json')['status'] != 'completed':
            for text in panel.iter('{http://www.w3.org/2000/svg}text'):
                if text.get('y') == '28':
                    text.text = (text.text or '')+' [PARTIAL]'
        panels.append(ET.tostring(panel,encoding='unicode'))
    save_chart(root,'staging_overview',[
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{80+560*len(panels)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="55" y="30" font-family="sans-serif" font-size="20">Qwen3-4B | 28-32K input | both prefetch OFF</text>',
        '<text x="55" y="57" font-family="sans-serif" font-size="14">DRAM 256GiB | staging 8GiB | concurrency 8 | demand reads + stores</text>',
        *panels,'</svg>'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','plot'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--requests',type=int,default=80)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.output.resolve(),args.requests)
    elif args.action == 'run':
        run(args.output.resolve())
    else:
        plot(args.output.resolve())
