"""Ordered GPU scatter with dynamically admitted concurrent window loads."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import time


def capacity_pipeline(spans, limit, depth, load, copy, release, emit, budget,
                      timeout=5., initializer=None):
    groups=[spans[i:i+limit] for i in range(0,len(spans),limit)]
    pending=deque();next_group=completed=total=0
    progress=time.monotonic()
    executor=ThreadPoolExecutor(max_workers=depth,initializer=initializer,
                                thread_name_prefix='capacity-window')
    def fetch(index,group,ticket):
        items=[];deadline=time.monotonic()+timeout
        with ticket.activate():
            emit('pipeline_load_start',window=index,chunks=len(group))
            try:
                while len(items)<len(group):
                    rest=group[len(items):]
                    part=load(rest[0][0],rest[-1][1])
                    actual=[(x[2],x[3]) for x in part]
                    items.extend(part)
                    if actual!=rest[:len(part)] or len(part)>len(rest):
                        raise RuntimeError('Non-contiguous capacity pipeline load')
                    if part:deadline=time.monotonic()+timeout
                    elif time.monotonic()>=deadline:raise TimeoutError('Read window made no progress')
                    else:time.sleep(.001)
                emit('pipeline_load_done',window=index,chunks=len(items))
                return items
            except BaseException:
                if items:release(items)
                raise
    try:
        while completed<len(spans):
            while next_group<len(groups) and len(pending)<depth:
                group=groups[next_group]
                ticket=budget.try_reserve('retrieve',len(group)*budget.chunk_bytes)
                if ticket is None:break
                index=next_group;ctx=copy_context()
                try:future=executor.submit(ctx.run,fetch,index,group,ticket)
                except BaseException:
                    ticket.close();raise
                pending.append((index,future,ticket));next_group+=1
                emit('pipeline_submitted',window=index,inflight=len(pending))
            if not pending or not pending[0][1].done():
                if time.monotonic()-progress>=timeout:
                    raise TimeoutError('Capacity pipeline waited too long for space/read completion')
                # Poll admission too: store completions can free capacity while
                # the first read is in flight. Never wait for it to fill slots.
                time.sleep(.001);continue
            index,future,ticket=pending[0]
            items=future.result();pending.popleft()
            try:
                nbytes=sum(x[1].get_size() for x in items)
                emit('window_copy_start',window=index,start=items[0][2],end=items[-1][3],
                     chunks=len(items),bytes=nbytes)
                copy(items);total+=nbytes
                emit('window_copy_done',window=index,chunks=len(items),bytes=nbytes)
            finally:
                try:release(items)
                finally:ticket.close()
            completed+=len(items);progress=time.monotonic()
            emit('window_released',window=index,completed_chunks=completed,total_chunks=len(spans))
    finally:
        for _,future,_ in pending:future.cancel()
        executor.shutdown(wait=True,cancel_futures=True)
        for _,future,ticket in pending:
            try:
                if not future.cancelled() and future.exception() is None:release(future.result())
            finally:ticket.close()
    return total
