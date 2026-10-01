import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from lookup_lock_probe import TimedLock, install
from sharegpt_lookup_lock_probe import external_request_id


def test_records_caller_and_preserves_exception():
    raw, rows = threading.Lock(), []
    lock = TimedLock(raw, rows.append)
    lookup_id = 'request-1'
    with pytest.raises(ValueError):
        with lock:
            assert raw.locked()
            raise ValueError('test')
    assert not raw.locked()
    row = rows[0]
    assert row['request_id'] == lookup_id
    assert row['operation'] == 'test_records_caller_and_preserves_exception'
    assert row['requested_ns'] <= row['acquired_ns'] <= row['release_started_ns'] <= row['released_ns']


def test_logging_happens_after_unlock():
    raw = threading.Lock()
    def emit(row):
        assert not raw.locked()
    with TimedLock(raw, emit):
        pass


def test_same_raw_lock_controls_both_threads():
    rows, raw = [], threading.Lock()
    lock = TimedLock(raw, rows.append)
    entering = threading.Event()
    def waiter():
        entering.set()
        with lock:
            return True
    with ThreadPoolExecutor(1) as pool:
        with lock:
            future = pool.submit(waiter)
            assert entering.wait(1)
            assert not future.done()
        assert future.result(timeout=1)
    assert len(rows) == 2 and not lock.locked()


def test_nonblocking_failure_preserves_owner_record():
    rows = []
    lock = TimedLock(threading.Lock(), rows.append)
    assert lock.acquire()
    assert not lock.acquire(False)
    lock.release()
    assert len(rows) == 1


def test_disabled(monkeypatch):
    monkeypatch.delenv('DAOS_LOOKUP_LOCK_PROBE', raising=False)
    assert install() is None


def test_native_poll_still_sleeps_under_lock_and_response_is_attributed(monkeypatch):
    from types import SimpleNamespace
    import time
    from lmcache.v1.lookup_client import lmcache_async_lookup_client as native
    c = native.LMCacheAsyncLookupClient.__new__(native.LMCacheAsyncLookupClient)
    rows, sleeps = [], []
    c.lock = TimedLock(threading.Lock(), rows.append)
    c.reqs_status, c.aborted_lookups = {'r': None}, set()
    c.first_lookup_time = {'r': time.time()}
    c.lookup_backoff_time = 0.001
    c.config = SimpleNamespace(lookup_timeout_ms=100000)
    def sleep(value):
        assert c.lock.locked()  # Probe must NOT implement the proposed fix.
        sleeps.append(value)
    monkeypatch.setattr(native.time, 'sleep', sleep)
    assert c.lookup_cache('r') is None
    assert sleeps == [0.001]
    c.running, c.world_size, c.res_for_each_worker = True, 1, {}
    def recv(copy):
        c.running = False
        return native.msgspec.msgpack.encode(native.LookupResponseMsg(lookup_id='r', num_hit_tokens=128))
    c.pull_socket = SimpleNamespace(recv=recv)
    c.process_responses_from_workers()
    assert c.reqs_status['r'] == 128
    assert [(r['operation'], r['request_id']) for r in rows] == [
        ('lookup_cache', 'r'), ('process_responses_from_workers', 'r')]


def test_internal_request_suffix_maps_to_api_request_id():
    assert external_request_id('chatcmpl-a756bcd84bd5e0e5-87a52240') == \
        'chatcmpl-a756bcd84bd5e0e5'
    assert external_request_id('custom-id-with-hyphens') == 'custom-id-with-hyphens'
    assert external_request_id(None) is None
