"""Opt-in async lookup result recheck after the ORIGINAL send and backoff.

Does not modify lookup_cache, sleeps, locks, worker aggregation or cleanup.
Installed LMCache stays untouched; disable DAOS_LOOKUP_READY_RETURN to roll back.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path

EXPECTED_SOURCE_SHA256 = '570c495d6955b071f77837b127c887c40e05e9206f5a8f9c6f2c668598235645'


def wrap_lookup(original):
    @functools.wraps(original)
    def lookup(self, token_ids, lookup_id, request_configs=None):
        # Preserve the original serialization, socket send, backoff and errors.
        result = original(self, token_ids, lookup_id, request_configs)
        stats = getattr(self, '_minji_ready_return_stats', None)
        if stats is None:
            stats = self._minji_ready_return_stats = dict(calls=0, ready=0, zero=0, pending=0, aborted=0)
        stats['calls'] += 1
        if result is None:
            with self.lock:
                if lookup_id in self.aborted_lookups:
                    stats['aborted'] += 1
                else:
                    ready = self.reqs_status.get(lookup_id)
                    # This map is populated only after ALL worker replies arrive.
                    # Missing/ongoing is None, not a cache miss (zero).
                    if isinstance(ready, int) and ready >= 0:
                        result = ready
                        stats['ready'] += 1
                        stats['zero'] += int(ready == 0)
                    else:
                        stats['pending'] += 1
        # Low-rate aggregate observations, no per-request disk I/O or new waits.
        if stats['calls'] <= 16 or stats['calls'] % 128 == 0:
            print('LOOKUP_READY_RETURN_STATS '+json.dumps(dict(pid=os.getpid(), **stats)), flush=True)
        return result
    return lookup


def install():
    if os.environ.get('DAOS_LOOKUP_READY_RETURN') != '1':
        return
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import LMCacheAsyncLookupClient as cls
    if getattr(cls, '_minji_ready_return_installed', False):
        return
    source = Path(inspect.getfile(cls))
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError('Async lookup source changed; review ready-return patch before enabling')
    original_close = cls.close
    cls.lookup = wrap_lookup(cls.lookup)

    @functools.wraps(original_close)
    def close(self):
        stats = getattr(self, '_minji_ready_return_stats', None)
        if stats is not None:
            print('LOOKUP_READY_RETURN_FINAL '+json.dumps(dict(pid=os.getpid(), **stats)), flush=True)
        return original_close(self)

    cls.close = close
    cls._minji_ready_return_installed = True
    print(f'LOOKUP_READY_RETURN installed pid={os.getpid()} source_sha256={EXPECTED_SOURCE_SHA256}', flush=True)
