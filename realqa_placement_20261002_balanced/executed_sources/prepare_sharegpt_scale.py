#!/usr/bin/env python3
"""Freeze an auditable, unpadded ShareGPT history replay for a large CPU cache.

Only prepares data; never touches DAOS or launches inference. Original answers,
not benchmark-generated answers, form subsequent inputs (LMBenchmark style).
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import struct
import time

MODEL = 'Qwen/Qwen3-14B'
CHUNK = 128
BYTES_PER_TOKEN = 163840


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def prefix_chunks(ids, chunk=CHUNK):
    """Content+entire-prefix identity; equality matches chained KV chunk keys."""
    prev = b''
    result = set()
    for offset in range(0, len(ids) - chunk + 1, chunk):
        prev = hashlib.sha256(prev + struct.pack(f'<{chunk}I', *ids[offset:offset+chunk])).digest()
        result.add(prev)
    return result


def history_pairs(entry, turns):
    messages = entry.get('conversations', [])
    if len(messages) < 2 * turns:
        return None
    history, result = [], []
    for i in range(turns):
        user, answer = messages[2*i:2*i+2]
        if (user.get('from') != 'human' or answer.get('from') != 'gpt'
                or not isinstance(user.get('value'), str)
                or not isinstance(answer.get('value'), str)
                or not user['value'].strip() or not answer['value'].strip()):
            return None
        history.append(user['value'])
        result.append(('\n'.join(history).strip(), answer['value']))
        history.append(answer['value'])
    return result


def prepare(a):
    from transformers import AutoTokenizer
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    params = dict(model=MODEL, seed=a.seed, turns=a.turns,
                  min_final_input=a.min_final_input, max_input=a.max_input,
                  max_output=a.max_output, target_gib=a.target_gib,
                  source=str(a.source.resolve()), source_sha256=digest(a.source),
                  extra_sources=[dict(path=str(p.resolve()), sha256=digest(p)) for p in a.extra_source],
                  chunk_tokens=CHUNK, kv_bytes_per_token=BYTES_PER_TOKEN)
    dump(root/'params.json', params)
    try:
        entries = json.loads(a.source.read_text())
        for source in a.extra_source:
            entries.extend(json.loads(source.read_text()))
        order = list(range(len(entries)))
        random.Random(a.seed).shuffle(order)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        tokenizer.model_max_length = 10**9

        def encode(text):
            return tokenizer.apply_chat_template(
                [{'role': 'user', 'content': text}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)

        selected, seen, reasons = [], set(), Counter()
        scanned = 0
        for source_index in order:
            scanned += 1
            entry = entries[source_index]
            pairs = history_pairs(entry, a.turns)
            if pairs is None:
                reasons['too_few_or_invalid_turns'] += 1
                continue
            if len(pairs[-1][0]) > 200000:
                reasons['over_200k_char_safety_bound'] += 1
                continue
            final_ids = encode(pairs[-1][0])
            if not a.min_final_input <= len(final_ids) <= a.max_input:
                reasons['final_length_outside_range'] += 1
                continue
            rows, chunk_set = [], set()
            for turn, (prompt, original_answer) in enumerate(pairs):
                ids = final_ids if turn == a.turns - 1 else encode(prompt)
                if len(ids) > a.max_input:
                    break
                answer_tokens = len(tokenizer.encode(original_answer, add_special_tokens=False))
                rows.append(dict(turn=turn, prompt=prompt,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    expected_prompt_tokens=len(ids),
                    original_answer_tokens=answer_tokens,
                    max_tokens=max(1, min(a.max_output, answer_tokens))))
                chunk_set.update(prefix_chunks(ids))
            if len(rows) != a.turns:
                reasons['earlier_turn_too_long'] += 1
                continue
            added = chunk_set - seen
            if not added:
                reasons['duplicate_prefixes'] += 1
                continue
            sid = len(selected)
            for row in rows:
                row['session'] = sid
            selected.append(dict(session=sid, source_index=source_index,
                source_id=entry.get('id'), unique_new_chunks=len(added), records=rows))
            seen.update(added)
            if len(selected) % 16 == 0:
                state = dict(status='preparing', scanned=scanned, sessions=len(selected),
                    estimated_unique_input_kv_gib=len(seen)*CHUNK*BYTES_PER_TOKEN/2**30,
                    updated_ns=time.time_ns())
                dump(root/'status.json', state)
                print(json.dumps(state), flush=True)
            if len(seen)*CHUNK*BYTES_PER_TOKEN >= a.target_gib*2**30:
                break
        dump(root/'selection_diagnostics.json', dict(scanned=scanned, sessions=len(selected),
             reasons=dict(reasons), unique_input_kv_gib=len(seen)*CHUNK*BYTES_PER_TOKEN/2**30))
        if len(seen)*CHUNK*BYTES_PER_TOKEN < a.target_gib*2**30:
            raise RuntimeError('Not enough eligible data; do not silently pad, duplicate, or widen limits')
        # Round-robin input order. The runner prevents overlapping turns of one
        # session, but does not impose a global wave/round completion barrier.
        records = [dict(index=i, **s['records'][turn])
                   for i, (turn, s) in enumerate(
                       (turn, s) for turn in range(a.turns) for s in selected)]
        dump(root/'requests.json', records)
        dump(root/'sessions.json', [{k:v for k,v in s.items() if k != 'records'} for s in selected])
        inputs = [r['expected_prompt_tokens'] for r in records]
        input_bytes = len(seen)*CHUNK*BYTES_PER_TOKEN
        # Conservative upper bound on additionally stored generated/tail chunks.
        # Runtime writes may be lower (capacity, EOS, duplicates, async admission).
        extra_bytes = sum(r['max_tokens']+CHUNK-1 for r in records)*BYTES_PER_TOKEN
        capacity = dict(sessions=len(selected), requests=len(records), turns=a.turns,
            scanned=scanned, source_entries=len(entries), excluded=dict(reasons),
            input_tokens_min=min(inputs), input_tokens_mean=sum(inputs)/len(inputs),
            input_tokens_max=max(inputs), input_tokens_sum=sum(inputs),
            unique_input_chunks=len(seen), unique_input_kv_gib=input_bytes/2**30,
            requested_output_tokens=sum(r['max_tokens'] for r in records),
            conservative_stored_kv_gib=(input_bytes+extra_bytes)/2**30,
            request_sha256=digest(root/'requests.json'),
            notes=['Fixed-seed, first eligible conversations; no performance-based selection.',
                   'First four original turns, cumulative text like upstream LMBenchmark; no padding or truncation.',
                   'Final turn 4K-8K; earlier turns may be shorter. Qwen chat-template token counts.',
                   'Unique input KV is a full-prefix chunk estimate, not actual DRAM residency.',
                   'Original answers form later prompts. Generated benchmark answers are NOT fed back.',
                   'Output cap 256; source-answer token length used when smaller. EOS remains enabled.'])
        dump(root/'capacity_estimate.json', capacity)
        dump(root/'status.json', dict(status='prepared', **capacity,
             elapsed_seconds=time.time()-started, updated_ns=time.time_ns()))
        print(json.dumps(capacity, indent=2), flush=True)
    except BaseException as exc:
        dump(root/'status.json', dict(status='failed', error=repr(exc), updated_ns=time.time_ns()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--extra-source', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-gib', type=float, default=384)
    p.add_argument('--turns', type=int, default=4)
    p.add_argument('--min-final-input', type=int, default=4096)
    p.add_argument('--max-input', type=int, default=8192)
    p.add_argument('--max-output', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260929)
    a = p.parse_args()
    if not (0 < a.target_gib <= 512 and 2 <= a.turns <= 8
            and 128 <= a.min_final_input <= a.max_input <= 8192
            and 1 <= a.max_output <= 512):
        p.error('Invalid preparation bounds')
    prepare(a)


if __name__ == '__main__':
    main()
