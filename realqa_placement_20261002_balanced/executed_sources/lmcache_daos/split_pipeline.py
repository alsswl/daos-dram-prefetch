"""Bounded, ordered consumption of concurrently loaded retrieve windows."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import time


def pipeline_windows(spans, limit, depth, load, copy, release, emit,
                     timeout=5.0, initializer=None):
    if limit < 1 or depth < 1 or timeout <= 0:
        raise ValueError('Positive window, depth and timeout required')
    groups = [spans[i:i+limit] for i in range(0, len(spans), limit)]
    pending = deque()
    total = completed = next_group = 0

    def fetch(index, group):
        emit('pipeline_load_start', window=index, chunks=len(group))
        items = []
        deadline = time.monotonic()+timeout
        try:
            while len(items) < len(group):
                remaining = group[len(items):]
                part = load(remaining[0][0], remaining[-1][1])
                actual = [(x[2], x[3]) for x in part]
                if actual != remaining[:len(part)] or len(part)>len(remaining):
                    items.extend(part)
                    raise RuntimeError('Non-contiguous pipeline load')
                if part:
                    items.extend(part)
                    deadline = time.monotonic()+timeout
                elif time.monotonic() >= deadline:
                    raise TimeoutError('Pipelined retrieve made no progress')
                else:
                    time.sleep(.01)
            emit('pipeline_load_done', window=index, chunks=len(items))
            return items
        except BaseException:
            # No GPU scatter was enqueued by this worker. The backend supplies
            # a release callback safe for loader threads as well as the caller.
            if items:
                release(items)
            raise

    executor = ThreadPoolExecutor(max_workers=depth, initializer=initializer,
                                  thread_name_prefix='daos-window-load')
    try:
        while completed < len(spans):
            while next_group < len(groups) and len(pending) < depth:
                index = next_group
                ctx = copy_context()
                pending.append((index, executor.submit(ctx.run, fetch, index, groups[index])))
                next_group += 1
                emit('pipeline_submitted', window=index, inflight=len(pending))
            index, future = pending[0]
            # Keep ownership in pending if result raises/times out, so cleanup
            # drains every completed future and releases its buffers.
            items = future.result(timeout=timeout)
            pending.popleft()
            try:
                nbytes = sum(x[1].get_size() for x in items)
                emit('window_copy_start', window=index, start=items[0][2],
                     end=items[-1][3], chunks=len(items), bytes=nbytes)
                copy(items)
                total += nbytes
                emit('window_copy_done', window=index, chunks=len(items), bytes=nbytes)
            finally:
                release(items)
            completed += len(items)
            emit('window_released', window=index, completed_chunks=completed,
                 total_chunks=len(spans))
    finally:
        for _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        for _, future in pending:
            if not future.cancelled() and future.exception() is None:
                release(future.result())
    return total
