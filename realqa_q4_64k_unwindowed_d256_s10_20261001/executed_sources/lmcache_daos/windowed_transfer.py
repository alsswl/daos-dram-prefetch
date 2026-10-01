"""CPU-only bounded-window driver. Copy must complete before release returns."""
import math
import time


def window_chunks(window_mib, chunk_bytes, pool_bytes):
    if isinstance(window_mib, bool) or not math.isfinite(float(window_mib)) or window_mib < 0:
        raise ValueError('retrieve_window_mib must be finite and >= 0 (0 disables windowing)')
    if window_mib == 0:
        return 0
    size = int(window_mib * 2**20)
    if not 0 < chunk_bytes <= size <= pool_bytes:
        raise ValueError('Read window must fit at least one KV chunk and be <= staging capacity')
    return size // chunk_bytes


def transfer_windows(spans, limit, load, copy, release, emit, timeout=5.0,
                     clock=time.monotonic, sleep=time.sleep):
    """Return copied bytes; never silently report a missing announced prefix.

    load(start,end) returns a contiguous prefix of (key,obj,start,end). A short
    successful prefix is consumed and freed BEFORE retrying its remaining tail.
    release must wait for DMA/secondary readers, and runs even on copy errors.
    """
    if limit < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive window size and bounded positive timeout required')
    position, total, deadline = 0, 0, clock()+timeout
    while position < len(spans):
        group = spans[position:position+limit]
        items = load(group[0][0], group[-1][1])
        if not items:
            if clock() >= deadline:
                raise TimeoutError('Windowed retrieve made no progress: staging busy or cache read failed')
            emit('window_retry', start=group[0][0], end=group[-1][1])
            sleep(.01)
            continue
        try:
            actual = [(v[2],v[3]) for v in items]
            if actual != group[:len(items)] or len(items)>len(group):
                raise RuntimeError('Window loader returned non-contiguous or out-of-window data')
            nbytes = sum(item[1].get_size() for item in items)
            emit('window_copy_start', start=actual[0][0], end=actual[-1][1],
                 chunks=len(items), bytes=nbytes)
            copy(items)
            total += nbytes
            emit('window_copy_done', chunks=len(items), bytes=nbytes)
        finally:
            release(items)
        position += len(items)
        deadline = clock()+timeout
        emit('window_released', completed_chunks=position, total_chunks=len(spans))
    return total
