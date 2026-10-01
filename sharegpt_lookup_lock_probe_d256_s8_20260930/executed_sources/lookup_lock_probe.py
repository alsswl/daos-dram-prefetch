"""Opt-in timing of the existing async lookup mutex, without changing its policy.

Acquisition duration includes thread scheduling/GIL overhead, not just contention.
JSON writes occur AFTER unlock; this diagnostic is not a speedup benchmark.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

from lookup_ready_return import EXPECTED_SOURCE_SHA256


class TimedLock:
    def __init__(self, raw, emit):
        self.raw, self.emit = raw, emit
        self.local = threading.local()

    def _acquire(self, blocking, timeout, operation, rid):
        was_locked = self.raw.locked()
        start = time.monotonic_ns()
        success = self.raw.acquire(blocking, timeout)
        acquired = time.monotonic_ns()
        if success:
            self.local.record = dict(operation=operation, request_id=rid,
                thread=threading.current_thread().name, tid=threading.get_native_id(),
                requested_ns=start, acquired_ns=acquired, observed_locked_before_acquire=was_locked)
        return success

    def acquire(self, blocking=True, timeout=-1):
        return self._acquire(blocking, timeout, 'explicit_acquire', None)

    def release(self):
        record = self.local.record
        record['release_started_ns'] = time.monotonic_ns()
        self.raw.release()
        record['released_ns'] = time.monotonic_ns()
        self.local.record = None
        record.update(time_ns=time.time_ns(), pid=os.getpid())
        self.emit(record)

    def locked(self):
        return self.raw.locked()

    def __enter__(self):
        frame = sys._getframe(1)
        operation, rid = frame.f_code.co_name, frame.f_locals.get('lookup_id')
        del frame
        self._acquire(True, -1, operation, rid)
        return self

    def __exit__(self, *args):
        self.release()
        return False


def install():
    if os.environ.get('DAOS_LOOKUP_LOCK_PROBE') != '1':
        return
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import LMCacheAsyncLookupClient as cls
    if getattr(cls, '_minji_lock_probe_installed', False):
        return
    if hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError('Async lookup source changed; review lock probe')
    original = cls.__init__

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        # Retain the SAME underlying mutex even if a response thread already exists.
        folder = Path(os.environ['DAOS_GDS_STAGING_TRACE']).parent
        path = folder/f'lookup_lock.{os.getpid()}.jsonl'
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        client_id = id(self)
        def emit(record):
            record['client_id'] = client_id
            payload = (json.dumps(record, separators=(',', ':'))+'\n').encode()
            if os.write(fd, payload) != len(payload):
                raise RuntimeError('Incomplete lock diagnostic record')
        # fd remains valid until process exit: close() can leave a recv thread alive.
        self.lock = TimedLock(self.lock, emit)
        print(f'LOOKUP_LOCK_PROBE active pid={os.getpid()} path={path}', flush=True)

    cls.__init__ = init
    cls._minji_lock_probe_installed = True
