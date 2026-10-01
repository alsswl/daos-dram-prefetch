"""Optional experiment-only CPU-prefetch timing; no extra CUDA synchronization.

Timestamps are monotonic host times. copy_ms includes enqueue + stream wait,
NOT pure DMA time. One summary record is emitted after a worker task finishes.
"""
import threading
import time


def install(prefetch, trace):
    state = threading.local()
    original_submit = prefetch.worker.submit
    original_reserve, original_copy = prefetch.reserve, prefetch.copy

    def reserve(sources):
        row = getattr(state, 'row', None)
        if row is None:
            return original_reserve(sources)
        row['reserve_start_ns'] = time.monotonic_ns()
        try:
            result = original_reserve(sources)
            row['allocation_ok'] = result is not None
            return result
        finally:
            row['reserve_end_ns'] = time.monotonic_ns()

    def copy(sources, destinations):
        row = getattr(state, 'row', None)
        if row is None:
            return original_copy(sources, destinations)
        row['copy_start_ns'] = time.monotonic_ns()
        try:
            return original_copy(sources, destinations)
        finally:
            row['copy_end_ns'] = time.monotonic_ns()

    def submit(fn, sources, lookup_id):
        # Capture before executor submission: includes submission/dispatch cost.
        row = dict(request_id=lookup_id, queued_ns=time.monotonic_ns(),
                   chunks=len(sources), bytes=sum(s.get_size() for s in sources))

        def observed():
            row['worker_start_ns'] = time.monotonic_ns()
            row['worker_thread_id'] = threading.get_ident()
            state.row = row
            try:
                result = fn(sources, lookup_id)
                row['outcome'] = 'staged' if row.get('allocation_ok') else 'fallback'
                return result
            except BaseException as exc:
                row.update(outcome='error', error=type(exc).__name__)
                raise
            finally:
                row['worker_end_ns'] = time.monotonic_ns()
                stream = getattr(prefetch, 'stream', None)
                if stream is not None:
                    row['cuda_stream_id'] = int(stream.cuda_stream)
                del state.row
                trace.emit('cpu_prefetch_timing', **row)

        return original_submit(observed)

    prefetch.reserve, prefetch.copy = reserve, copy
    prefetch.worker.submit = submit
