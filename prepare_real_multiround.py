#!/usr/bin/env python3
"""Prepare auditable Gutenberg prefixes for LMBenchmark real-multi-round-qa.

No GPU/DAOS access. Serial downloads, resumable files, fixed token-length category.
Unlike upstream's largest-category choice, use the first 16K tokens of each
eligible book to keep the model's 32K context safe as conversation grows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from transformers import AutoTokenizer


def dump(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--books', type=int, default=200)
    p.add_argument('--tokens', type=int, default=16384)
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--session-depth', type=int, default=24)
    p.add_argument('--last-id', type=int, default=800)
    p.add_argument('--reuse-raw-from', type=Path)
    p.add_argument('--local-only', action='store_true')
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if a.tokens not in (16384, 65536):
        raise ValueError('Supported document sizes: 16384 or 65536 tokens')
    category = f'{a.tokens//1024}k'
    for folder in ('raw', category):
        (root/folder).mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-14B', local_files_only=True)
    tok.model_max_length = 10**9  # Tokenization only; inference retains a 32768-token limit.
    upstream = Path(__file__).resolve().parent/'LMBenchmark'
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()
    params = dict(books=a.books, tokens=a.tokens, concurrency=a.concurrency,
                  session_depth=a.session_depth, last_id=a.last_id,
                  model='Qwen/Qwen3-14B', upstream_commit=commit, seed=20260929)
    if a.reuse_raw_from is not None:
        params['reuse_raw_from'] = str(a.reuse_raw_from.resolve())
    if a.local_only:
        params['local_only'] = True
    if (root/'params.json').exists():
        assert json.loads((root/'params.json').read_text()) == params
    else:
        dump(root/'params.json', params)
    audit_path = root/'download_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else []
    manifest_path = root/'documents.json'
    docs = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for doc in docs:
        assert sha((root/doc['file']).read_bytes()) == doc['sha256']
    processed = {r['book_id'] for r in audit}
    processed.update(d['book_id'] for d in docs)
    seen = {d['sha256'] for d in docs}
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(
        total=3, connect=3, read=3, status=3, backoff_factor=2,
        status_forcelist=(500, 502, 503, 504), allowed_methods=('GET',),
        respect_retry_after_header=True))
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers['User-Agent'] = 'LMBenchmark-research-data-preparation/1.0 (serial download)'
    started = time.time()
    consecutive_network_failures = 0
    try:
        for book_id in range(1, a.last_id+1):
            if len(docs) >= a.books:
                break
            if book_id in processed:
                continue
            url = f'https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8'
            dump(root/'status.json', dict(status='downloading', eligible_books=len(docs),
                target_books=a.books, current_book_id=book_id, updated_ns=time.time_ns()))
            row = dict(book_id=book_id, url=url)
            raw = root/'raw'/f'{book_id}.txt'
            if not raw.exists() and a.reuse_raw_from is not None:
                cached = a.reuse_raw_from.resolve()/'raw'/f'{book_id}.txt'
                if cached.exists():
                    shutil.copy2(cached, raw)
            if not raw.exists() and a.local_only:
                row['reason'] = 'not_available_locally'
                audit.append(row); dump(audit_path, audit)
                continue
            if raw.exists():
                content = raw.read_bytes()
            else:
                time.sleep(1)
                try:
                    response = session.get(url, timeout=(10, 30), stream=True)
                except (requests.ConnectionError, requests.Timeout) as exc:
                    consecutive_network_failures += 1
                    row.update(reason='network_failure_after_retries', error=repr(exc))
                    audit.append(row); dump(audit_path, audit)
                    if consecutive_network_failures >= 5:
                        raise RuntimeError('Five consecutive network failures; stop downloads') from exc
                    print(f'Skipping unavailable book {book_id} after bounded retries', flush=True)
                    continue
                consecutive_network_failures = 0
                with response:
                    row['http_status'] = response.status_code
                    if response.status_code in (403, 429):
                        raise RuntimeError(f'Gutenberg throttled/denied HTTP {response.status_code}; stop downloads')
                    if response.status_code == 404:
                        row['reason'] = 'not_found'
                        audit.append(row); dump(audit_path, audit)
                        continue
                    response.raise_for_status()
                    if 'text/plain' not in response.headers.get('Content-Type',''):
                        raise RuntimeError(f'Expected plain text for book {book_id}')
                    data = bytearray()
                    for block in response.iter_content(65536):
                        data.extend(block)
                        if len(data) > 20*2**20:
                            raise RuntimeError(f'Book {book_id} exceeds 20MiB download safety bound')
                    content = bytes(data)
                    raw.write_bytes(content)
            text = content.decode('utf-8-sig')
            if 'Project Gutenberg' not in text[:10000]:
                raise RuntimeError(f'Missing expected Gutenberg header in {book_id}')
            tokens = tok.encode(text, add_special_tokens=False)
            row.update(raw_tokens=len(tokens), raw_bytes=len(content), raw_sha256=sha(content))
            if len(tokens) < a.tokens:
                row['reason'] = f'shorter_than_{category}'
            else:
                prefix = tok.decode(tokens[:a.tokens])
                data = prefix.encode()
                digest = sha(data)
                if digest in seen:
                    row['reason'] = 'duplicate_prefix'
                else:
                    filename = f'{category}/{book_id}.txt'
                    (root/filename).write_bytes(data)
                    actual = len(tok.encode(prefix, add_special_tokens=False))
                    assert abs(actual-a.tokens) <= 4
                    docs.append(dict(book_id=book_id, file=filename, sha256=digest,
                        tokens=actual, original_tokens=len(tokens), source=url, raw_sha256=sha(content)))
                    seen.add(digest)
                    dump(manifest_path, docs)
                    row['reason'] = 'selected'
                    print(f'Prepared {len(docs)}/{a.books}: Gutenberg {book_id}, {actual} tokens', flush=True)
            audit.append(row); dump(audit_path, audit)
        if len(docs) < a.books:
            raise RuntimeError(f'Only {len(docs)} eligible books through ID {a.last_id}; do not auto-expand scope')
        # Preserve upstream's with-replacement sampling, but freeze it once for
        # all policies. Never choose a seed based on measured performance.
        rng = random.Random(params['seed'])
        ordered = sorted(docs, key=lambda d:d['book_id'])
        assignments = [dict(group=g, slot=s, **rng.choice(ordered))
                       for g in range(a.concurrency) for s in range(a.session_depth)]
        unique = {d['sha256']:d for d in assignments}
        raw_kv_gib = sum(d['tokens'] for d in unique.values())*163840/2**30
        dump(root/'sessions.json', assignments)
        dump(root/'capacity_estimate.json', dict(sessions=len(assignments), unique_documents=len(unique),
            initial_document_kv_gib=raw_kv_gib, dram_capacity_gib=256,
            caveat='Estimate before shared-prefix deduplication; actual DRAM retention and later growth are measured.',
            sampling='Fixed seed, with replacement like upstream; identical assignments across policies.'))
        dump(root/'status.json', dict(status='prepared', books=len(docs), sessions=len(assignments),
            unique_documents=len(unique), initial_document_kv_gib=raw_kv_gib,
            elapsed_seconds=time.time()-started, updated_ns=time.time_ns()))
        print(f'Data ready: {len(unique)} unique documents, estimated {raw_kv_gib:.2f}GiB initial document KV',flush=True)
    except BaseException as exc:
        dump(root/'status.json', dict(status='failed', eligible_books=len(docs),error=repr(exc),updated_ns=time.time_ns()))
        raise


if __name__ == '__main__':
    main()
