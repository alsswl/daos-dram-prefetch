import asyncio
import io
import json
from types import SimpleNamespace

from lmcache_daos.staging_trace import StagingTrace
from staging_pressure import trace_summary


def test_ready_memory_is_released_only_on_real_free():
    class Allocator:
        total_allocated_size = 0
        num_active_allocations = 0

        def allocate(self):
            self.total_allocated_size += 32
            self.num_active_allocations += 1
            return SimpleNamespace(get_size=lambda: 32)

        def free(self, obj):
            self.total_allocated_size -= 32
            self.num_active_allocations -= 1

        def batched_allocate(self):
            return None

        def batched_free(self, objs):
            for obj in objs:
                self.free(obj)

    allocator = Allocator()
    async def get(*args):
        return [allocator.allocate()]
    backend = SimpleNamespace(memory_allocator=SimpleNamespace(allocator=allocator),
                              batched_get_non_blocking=get)
    stream = io.StringIO()
    trace = StagingTrace(backend, stream)
    trace.wrap_allocator()
    trace.wrap_prefetch()
    objects = asyncio.run(backend.batched_get_non_blocking('r1', ['key']))
    trace.retrieve_start('r1')
    assert sum(size for _, size in trace.ready.values()) == 32
    allocator.free(objects[0])
    assert trace.ready == {}
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert events[-1]['used_bytes'] == events[-1]['ready_bytes'] == 0
    summary = trace_summary(events)
    assert summary['prefetches'] == summary['peak_active_prefetches'] == 1
    assert summary['alloc_failures'] == 0


def test_serializer_preserves_result_and_records_wait():
    async def experiment():
        class Serializer:
            async def run(self, coro, chunks):
                await asyncio.sleep(.005)
                return await coro
        tr = StagingTrace(SimpleNamespace(memory_allocator=SimpleNamespace(
            allocator=SimpleNamespace(total_allocated_size=0, num_active_allocations=0))), io.StringIO())
        ser = Serializer()
        tr.wrap_serializer(ser)
        async def work(lookup_id):
            return 17
        assert await ser.run(work('r1'), 1) == 17
        rows = [json.loads(line) for line in tr.stream.getvalue().splitlines()]
        assert rows[1]['request_id'] == 'r1'
        assert rows[1]['wait_ms'] >= 4
        assert rows[-1]['event'] == 'serializer_return'
    asyncio.run(experiment())


def test_serializer_cancel_before_acquisition_closes_coroutines():
    async def experiment():
        queued = asyncio.Event()
        class Serializer:
            async def run(self, coro, chunks):
                queued.set()
                await asyncio.Event().wait()
                return await coro
        tr = StagingTrace(SimpleNamespace(memory_allocator=SimpleNamespace(
            allocator=SimpleNamespace(total_allocated_size=0, num_active_allocations=0))), io.StringIO())
        ser = Serializer()
        tr.wrap_serializer(ser)
        async def work(lookup_id):
            raise AssertionError('must not run')
        coro = work('cancelled')
        task = asyncio.create_task(ser.run(coro, 1))
        await queued.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert coro.cr_frame is None
        assert json.loads(tr.stream.getvalue().splitlines()[-1])['acquired'] is False
    asyncio.run(experiment())
