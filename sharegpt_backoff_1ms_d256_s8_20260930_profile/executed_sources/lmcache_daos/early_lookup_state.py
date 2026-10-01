"""Ownership/state machine for metadata-first prefetch (no CUDA dependencies)."""
from concurrent.futures import Future
import threading
import time


class EarlyReadView:
    def __init__(self, batch, index):
        self.batch, self.index = batch, index
        self._refs = 1
        self._lock = threading.Lock()

    @property
    def is_pinned(self):
        return self.batch.owns_pin(self.index)

    def unpin(self):
        self.batch.unpin(self.index)

    def ref_count_up(self):
        with self._lock:
            if self._refs <= 0:
                raise RuntimeError('Retain after early-view release')
            self._refs += 1

    def ref_count_down(self):
        with self._lock:
            if self._refs <= 0:
                raise RuntimeError('Double early-view release')
            self._refs -= 1
            final = self._refs == 0
        if final:
            self.batch.release(self.index)

    def get_ref_count(self):
        with self._lock:
            return self._refs

    def __getattr__(self, name):
        # Payload access is only legal after explicit retrieve resolution.
        if self.batch.selected is None:
            raise RuntimeError('Early payload accessed before retrieve resolution')
        return getattr(self.batch.selected[self.index], name)


class EarlyReadBatch:
    """Future.cancel wins only while queued; running DMA is never cancelled.

    sources are owned CPU references/pins, or None for DAOS. load functions
    return owned GPU objects, or None to retain the CPU source on admission miss.
    Consumers and the worker jointly control lifetime via released+finished.
    """
    def __init__(self, rid, tier, count, sources, release_gpu, demand, emit, retired=lambda b: None):
        self.rid, self.tier, self.count = rid, tier, count
        self.sources, self.release_gpu, self.demand = sources, release_gpu, demand
        self.emit, self.retired = emit, retired
        self.future = Future()
        self.work_finished = threading.Event()
        self.lock, self.resolve_lock = threading.RLock(), threading.Lock()
        self.destinations = self.selected = None
        self.finished = self.is_retired = False
        self.demand_running = False
        self.released, self.cleaned, self.unpinned = set(), set(), set()
        self.views = [EarlyReadView(self, i) for i in range(count)]

    def owns_pin(self, index):
        with self.lock:
            return self.sources is not None and index not in self.unpinned

    def unpin(self, index):
        with self.lock:
            if self.sources is not None and index not in self.unpinned:
                self.sources[index].unpin()
                self.unpinned.add(index)

    def claim(self):
        return self.future.set_running_or_notify_cancel()

    def finish(self, objects=None, error=None, publish=True):
        if objects is not None and (len(objects) != self.count or any(o is None for o in objects)):
            for obj in objects:
                if obj is not None:
                    self.release_gpu(obj)
            objects = None
            error = RuntimeError('Announced cache could not be fully loaded; refuse unsafe model execution')
        if self.sources is None and objects is None and error is None:
            error = RuntimeError('DAOS load returned no payload')
        with self.lock:
            self.destinations = objects
            self.finished = True
            self._clean()
            self.work_finished.set()
        if publish:
            if error is None:
                self.future.set_result(None)
            else:
                self.future.set_exception(error)
        elif error is not None:
            raise error

    def resolve(self):
        with self.resolve_lock:
            if self.selected is not None:
                return
            started = time.monotonic_ns()
            # Serialize ownership handoff against abort/release.
            with self.lock:
                if self.released:
                    raise RuntimeError('Resolve after release/abort')
                cancelled = self.future.cancel()
                if cancelled:
                    self.demand_running = True
            if cancelled:
                # A queued prefetch (including serializer wait) cannot start now.
                try:
                    self.finish(self.demand(), publish=False)
                except BaseException:
                    with self.lock:
                        self.finished = True
                        self._clean()
                        self.work_finished.set()
                    raise
                finally:
                    with self.lock:
                        self.demand_running = False
                decision = 'queued_to_demand'
            else:
                ready = self.future.done()
                self.future.result()
                decision = 'ready' if ready else 'wait_running'
            with self.lock:
                if self.released:
                    raise RuntimeError('Request aborted during retrieval')
                self.selected = self.destinations if self.destinations is not None else self.sources
            self.emit('early_retrieve_decision', request_id=self.rid, tier=self.tier,
                      decision=decision, wait_ms=(time.monotonic_ns()-started)/1e6)

    def release(self, index):
        with self.lock:
            if index in self.released:
                raise RuntimeError('Duplicate batch release')
            self.unpin(index)
            self.released.add(index)
            if (len(self.released) == self.count and not self.finished
                    and not self.demand_running and self.future.cancel()):
                self.finished = True
                self.work_finished.set()
            self._clean()

    def abort(self):
        for i in range(self.count):
            with self.lock:
                if i not in self.released:
                    self.release(i)

    def _clean(self):
        if not self.finished:
            return
        for i in self.released - self.cleaned:
            if self.destinations is not None:
                self.release_gpu(self.destinations[i])
            if self.sources is not None:
                self.sources[i].ref_count_down()
            self.cleaned.add(i)
        if len(self.cleaned) == self.count and not self.is_retired:
            self.is_retired = True
            self.retired(self)
