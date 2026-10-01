"""Opt-in early CPU readiness with atomic cancel-before-start at retrieve.

No installed LMCache files are edited. Sources remain owned until both the
consumer and any in-flight DMA finish. A running transfer is never cancelled.
"""
import functools
import threading
import time


class DeferredCPUObject:
    def __init__(self, batch, index):
        self.batch, self.index = batch, index
        self._refs = 1
        self._lock = threading.Lock()

    @property
    def source(self):
        return self.batch.sources[self.index]

    def get_size(self):
        return self.source.get_size()

    def get_shapes(self):
        return self.source.get_shapes()

    def get_dtypes(self):
        return self.source.get_dtypes()

    @property
    def is_pinned(self):
        return self.source.is_pinned

    def pin(self):
        return self.source.pin()

    def unpin(self):
        return self.source.unpin()

    def get_ref_count(self):
        with self._lock:
            return self._refs

    def ref_count_up(self):
        with self._lock:
            if self._refs <= 0:
                raise RuntimeError('Cannot retain released deferred CPU object')
            self._refs += 1

    def ref_count_down(self):
        with self._lock:
            if self._refs <= 0:
                raise RuntimeError('Double release of deferred CPU object')
            self._refs -= 1
            final = self._refs == 0
        if final:
            self.batch.release(self.index)

    def __getattr__(self, name):
        # Fail closed if a new LMCache path accesses data before the explicit
        # retrieve hook. Resolve the WHOLE CPU tier once, never chunk by chunk.
        self.batch.resolve()
        return getattr(self.batch.selected[self.index], name)


class DeferredBatch:
    def __init__(self, prefetch, sources, lookup_id):
        self.prefetch, self.sources, self.lookup_id = prefetch, sources, lookup_id
        self.lock = threading.RLock()
        self.resolve_lock = threading.Lock()
        self.destinations = None
        self.selected = None
        self.finished = False
        self.released, self.cleaned = set(), set()
        self.future = None
        self.queued_ns = time.monotonic_ns()
        with prefetch._deferred_stats_lock:
            prefetch.stats['deferred_pending_batches'] += 1
            prefetch.stats['deferred_submitted'] += 1
        try:
            # Same signature as normal stage, preserving the timing wrapper.
            self.future = prefetch.worker.submit(self.work, sources, lookup_id)
        except BaseException:
            self.finished = True
            for source in sources:
                if source.is_pinned:
                    source.unpin()
            self.released.update(range(len(sources)))
            self._clean()
            raise

    def emit(self, event, **fields):
        trace = getattr(self.prefetch.backend, '_staging_trace', None)
        if trace is not None:
            trace.emit(event, request_id=self.lookup_id, **fields)

    def work(self, sources, lookup_id):
        pf = self.prefetch
        try:
            pf.backend._ensure_cuda_ctx()
            if all(s.raw_data.device.type == 'cpu' for s in sources):
                self.destinations = pf.reserve(sources)
            if self.destinations is None:
                pf.increment_stats(fallback_requests=1)
            else:
                pf.copy(sources, self.destinations)
                pf.increment_stats(staged_requests=1, staged_bytes=sum(s.get_size() for s in sources))
                # Probe's early CPU-ready event is NOT staging-ready. Register
                # GPU readiness only after DMA, and before a possible free.
                trace = getattr(pf.backend, '_staging_trace', None)
                if trace is not None:
                    with trace.lock:
                        for obj in self.destinations:
                            trace.ready[id(obj)] = (lookup_id, obj.get_size())
                            ready = getattr(pf.backend, '_probe_cpu_ready', None)
                            if ready is not None:
                                ready[id(obj)] = lookup_id
                        trace.emit('cpu_deferred_staging_ready', request_id=lookup_id,
                                   chunks=len(sources))
        except BaseException:
            pf.increment_stats(copy_errors=1)
            raise
        finally:
            # copy() synchronizes also on errors before buffers can be freed.
            with self.lock:
                self.finished = True
                self._clean()

    def resolve(self):
        """Called once per CPU tier at retrieve's event-consumption boundary."""
        with self.resolve_lock:
            if self.selected is not None:
                return
            started = time.monotonic_ns()
            if self.prefetch.cancel_queued and self.future.cancel():
                # Future.cancel atomically wins against executor starting work.
                # The executor will skip its queue entry, even if not removed
                # physically from SimpleQueue. No allocation/copy takes place.
                with self.lock:
                    self.finished = True
                    self.selected = self.sources
                decision = 'cancelled_queued'
            else:
                already_done = self.future.done()
                self.future.result()  # Running or early_wait: never race DMA.
                self.selected = self.destinations if self.destinations is not None else self.sources
                decision = ('ready_gpu' if already_done else 'waited_gpu') if self.destinations is not None else 'capacity_cpu'
            self.prefetch.increment_stats(**{'retrieve_' + decision: 1})
            self.emit('cpu_prefetch_retrieve_decision', decision=decision,
                      queued_ns=self.queued_ns, resolve_start_ns=started,
                      resolve_end_ns=time.monotonic_ns(), chunks=len(self.sources),
                      bytes=sum(s.get_size() for s in self.sources))

    def release(self, index):
        with self.lock:
            self.released.add(index)
            # Abort/unused tail: avoid copying if ALL views have been released.
            if len(self.released) == len(self.sources) and not self.finished:
                if self.future.cancel():
                    self.finished = True
                    self.prefetch.increment_stats(deferred_abort_cancelled=1)
            self._clean()

    def _clean(self):
        if not self.finished:
            return
        for i in self.released - self.cleaned:
            if self.destinations is not None:
                self.prefetch.backend._release_memory_obj(self.destinations[i])
            self.sources[i].ref_count_down()
            self.cleaned.add(i)
        if len(self.cleaned) == len(self.sources) and not getattr(self, '_retired', False):
            self._retired = True
            with self.prefetch._deferred_stats_lock:
                self.prefetch.stats['deferred_pending_batches'] -= 1


def resolve_event(engine, lookup_id):
    from lmcache.v1.event_manager import EventType
    future = engine.event_manager.get_event_future(EventType.LOADING, lookup_id)
    # Preserve native missing/incomplete-event error handling.
    if future is None or not future.done():
        return
    seen = set()
    for tier in future.result():
        for _, obj in tier:
            if (isinstance(obj, DeferredCPUObject) and obj.get_ref_count() > 0
                    and id(obj.batch) not in seen):
                seen.add(id(obj.batch))
                obj.batch.resolve()


def install_retrieve_hook():
    from lmcache.v1.cache_engine import LMCacheEngine
    if getattr(LMCacheEngine, '_daos_deferred_cpu_installed', False):
        return
    original = LMCacheEngine._async_process_tokens_internal

    @functools.wraps(original)
    def process(engine, *args, **kwargs):
        # Inside retrieve timing, before hashing/reading the event's MemoryObjs.
        # Existing consumed-event ownership patch remains inside this wrapper.
        if kwargs.get('req_id') is not None:
            resolve_event(engine, kwargs['req_id'])
        return original(engine, *args, **kwargs)

    LMCacheEngine._async_process_tokens_internal = process
    LMCacheEngine._daos_deferred_cpu_installed = True
