"""Opt-in lifecycle tracing. No scheduling or memory-budget policy changes.

DAOS_GDS_STAGING_TRACE is an output prefix; each worker appends its PID.
Pool counters are observed on actual allocation/free, not CUDA reserved bytes.
This is diagnostic instrumentation, with nonzero logging overhead.
"""
import functools
import json
import os
import threading
import time


class StagingTrace:
    def __init__(self, backend, stream):
        self.backend = backend
        self.allocator = backend.memory_allocator.allocator
        self.stream = stream
        self.lock = threading.RLock()
        self.ready = {}  # IDs/sizes only: never retain MemoryObj references.
        self.ready_at = {}

    def emit(self, event, **fields):
        with self.lock:
            row = dict(event=event, time_ns=time.time_ns(),
                       monotonic_ns=time.monotonic_ns(), pid=os.getpid(),
                       used_bytes=self.allocator.total_allocated_size,
                       active_objects=self.allocator.num_active_allocations,
                       ready_bytes=sum(size for _, size in self.ready.values()),
                       **fields)
            self.stream.write(json.dumps(row) + "\n")

    def wrap_allocator(self):
        for name in ("allocate", "batched_allocate", "free", "batched_free"):
            original = getattr(self.allocator, name)

            def wrapper(*args, _original=original, _name=name, **kwargs):
                ids = []
                if _name == "free":
                    ids = [id(args[0] if args else kwargs['memory_obj'])]
                elif _name == "batched_free":
                    ids = [id(o) for o in (args[0] if args else kwargs['memory_objs'])]
                result = _original(*args, **kwargs)
                with self.lock:
                    for oid in ids:
                        self.ready.pop(oid, None)
                    self.emit(_name, failed=result is None if 'allocate' in _name else False)
                return result

            setattr(self.allocator, name, wrapper)

    def wrap_prefetch(self):
        original = self.backend.batched_get_non_blocking

        async def prefetch(lookup_id, keys, transfer_spec=None):
            self.emit('prefetch_start', request_id=lookup_id, chunks=len(keys))
            try:
                objects = await original(lookup_id, keys, transfer_spec)
            except BaseException as exc:
                self.emit('prefetch_error', request_id=lookup_id, error=type(exc).__name__)
                raise
            with self.lock:
                self.ready_at[lookup_id] = time.monotonic_ns()
                for obj in objects:
                    self.ready[id(obj)] = (lookup_id, obj.get_size())
                self.emit('prefetch_ready', request_id=lookup_id, chunks=len(objects),
                          requested_chunks=len(keys), bytes=sum(o.get_size() for o in objects))
            return objects

        self.backend.batched_get_non_blocking = prefetch

    def wrap_serializer(self, serializer):
        original = serializer.run

        async def run(coro, num_chunks):
            frame = getattr(coro, 'cr_frame', None)
            rid = frame.f_locals.get('lookup_id') if frame else None
            queued = time.monotonic_ns()
            self.emit('serializer_queued', request_id=rid, chunks=num_chunks)
            entered = False

            async def observed():
                nonlocal entered
                entered = True
                self.emit('serializer_acquired', request_id=rid, chunks=num_chunks,
                          wait_ms=(time.monotonic_ns() - queued) / 1e6)
                return await coro

            wrapped = observed()
            try:
                return await original(wrapped, num_chunks)
            finally:
                # Avoid unawaited-coroutine leaks if cancelled during acquisition.
                if not entered:
                    wrapped.close()
                    coro.close()
                self.emit('serializer_return', request_id=rid, acquired=entered)

        serializer.run = run

    def retrieve_start(self, rid):
        with self.lock:
            ready_at = self.ready_at.pop(rid, None)
            self.emit('retrieve_start', request_id=rid,
                      ready_wait_ms=None if ready_at is None else
                      (time.monotonic_ns() - ready_at) / 1e6)


def install(backend):
    from lmcache.v1.cache_engine import LMCacheEngine
    path = f"{os.environ['DAOS_GDS_STAGING_TRACE']}.{os.getpid()}.jsonl"
    trace = StagingTrace(backend, open(path, 'a', buffering=1))
    trace.wrap_allocator()
    trace.wrap_prefetch()
    # This diagnostic targets one GDS backend per worker process.
    LMCacheEngine._daos_staging_trace = trace
    if not getattr(LMCacheEngine, '_daos_staging_trace_installed', False):
        original = LMCacheEngine.retrieve

        @functools.wraps(original)
        def retrieve(self, *args, **kwargs):
            tr = LMCacheEngine._daos_staging_trace
            rid = self._get_req_id(kwargs)
            tr.retrieve_start(rid)
            try:
                return original(self, *args, **kwargs)
            finally:
                tr.emit('retrieve_return', request_id=rid)

        LMCacheEngine.retrieve = retrieve
        LMCacheEngine._daos_staging_trace_installed = True
    trace.emit('init', capacity_bytes=backend.gpu_buffer_bytes,
               chunk_budget=backend.calculate_chunk_budget(), transport=backend.transport)
    return trace
