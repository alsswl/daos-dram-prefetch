import asyncio
import threading
from types import SimpleNamespace as NS

from test_dram_prefetch import fixture, discard
from lmcache_daos.prefetch_timing import install


def test_timing_stage_and_fallback_preserve_lifetime():
    pf, cpu, inner, sources = fixture()
    rows = []
    install(pf, NS(emit=lambda event, **row: rows.append(dict(event=event, **row))))
    try:
        out = asyncio.run(cpu.batched_get_non_blocking('first', ['key']))
        discard(out)
        row = rows[-1]
        assert row['outcome'] == 'staged'
        names = ['queued_ns', 'worker_start_ns', 'reserve_start_ns', 'reserve_end_ns',
                 'copy_start_ns', 'copy_end_ns', 'worker_end_ns']
        assert [row[n] for n in names] == sorted(row[n] for n in names)
        inner.total_allocated_size = 128
        out = asyncio.run(cpu.batched_get_non_blocking('second', ['key']))
        assert out is sources
        discard(out)
        assert rows[-1]['outcome'] == 'fallback' and 'copy_start_ns' not in rows[-1]
        assert sources[0].refs == 1
    finally:
        pf.close()


def test_timing_records_executor_queue_without_changing_order():
    pf, cpu, inner, sources = fixture()
    rows = []
    install(pf, NS(emit=lambda event, **row: rows.append(row)))
    entered, release = threading.Event(), threading.Event()
    original_copy = pf.copy
    calls = 0
    def blocked_copy(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(5)
        return original_copy(*args)
    pf.copy = blocked_copy
    async def run():
        first = asyncio.create_task(cpu.batched_get_non_blocking('first', ['key']))
        while not entered.is_set(): await asyncio.sleep(.001)
        second = asyncio.create_task(cpu.batched_get_non_blocking('second', ['key']))
        await asyncio.sleep(.02)
        release.set()
        for out in await asyncio.gather(first, second): discard(out)
    try:
        asyncio.run(run())
        assert [r['request_id'] for r in rows] == ['first', 'second']
        assert rows[1]['queued_ns'] < rows[0]['worker_end_ns'] <= rows[1]['worker_start_ns']
        assert inner.total_allocated_size == 0 and sources[0].refs == 1
    finally:
        release.set()
        pf.close()
