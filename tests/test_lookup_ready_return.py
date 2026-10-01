from types import SimpleNamespace
import threading
import pytest
from lookup_ready_return import wrap_lookup, install


def client():
    return SimpleNamespace(lock=threading.Lock(), reqs_status={}, aborted_lookups=set())


@pytest.mark.parametrize('ready', [0, 128, 8192])
def test_returns_reply_arriving_during_original_call(ready):
    c, calls = client(), []
    def original(self, tokens, rid, config):
        calls.append((tokens, rid, config))
        with self.lock:
            self.reqs_status[rid] = ready
        return None
    assert wrap_lookup(original)(c, [1,2], 'r', {'x': 3}) == ready
    assert calls == [([1,2], 'r', {'x': 3})]
    assert c.reqs_status['r'] == ready  # No cleanup/consumption here.


@pytest.mark.parametrize('status', ['missing', None, -1])
def test_missing_or_partial_response_remains_pending(status):
    c = client()
    if status != 'missing':
        c.reqs_status['r'] = status
    c.res_for_each_worker = {'r': [128]}  # Partial multi-rank response is not enough.
    assert wrap_lookup(lambda *args: None)(c, [], 'r') is None


def test_abort_never_returns_stale_ready_value():
    c = client()
    c.reqs_status['r'] = 128
    c.aborted_lookups.add('r')
    assert wrap_lookup(lambda *args: None)(c, [], 'r') is None
    assert c.aborted_lookups == {'r'}


def test_original_result_and_exceptions_preserved():
    c = client()
    assert wrap_lookup(lambda *args: 256)(c, [], 'r') == 256
    def fail(*args):
        raise ValueError('socket failure')
    with pytest.raises(ValueError, match='socket failure'):
        wrap_lookup(fail)(c, [], 'r')


def test_response_thread_can_publish_before_recheck():
    c = client()
    def original(self, *args):
        def respond():
            with self.lock:
                self.reqs_status['r'] = 128
        thread = threading.Thread(target=respond)
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
        return None
    assert wrap_lookup(original)(c, [], 'r') == 128


def test_disabled_install_is_noop(monkeypatch):
    monkeypatch.delenv('DAOS_LOOKUP_READY_RETURN', raising=False)
    assert install() is None


def test_installed_client_keeps_send_sleep_and_polling_implementation(monkeypatch):
    from lmcache.v1.lookup_client import lmcache_async_lookup_client as native
    cls = native.LMCacheAsyncLookupClient
    original_poll = cls.lookup_cache
    # Register undo before the opt-in installer changes class attributes.
    monkeypatch.setattr(cls, 'lookup', cls.lookup)
    monkeypatch.setattr(cls, 'close', cls.close)
    monkeypatch.setattr(cls, '_minji_ready_return_installed', False, raising=False)
    monkeypatch.setenv('DAOS_LOOKUP_READY_RETURN', '1')
    sleeps, sent = [], []
    monkeypatch.setattr(native.time, 'sleep', lambda value: sleeps.append(value))
    install()
    wrapper = cls.lookup
    install()
    assert cls.lookup is wrapper and cls.lookup_cache is original_poll
    c = cls.__new__(cls)
    c.lock, c.reqs_status, c.aborted_lookups = threading.Lock(), {'r': None}, set()
    c.world_size, c.lookup_backoff_time = 1, 0.001
    c.token_database = SimpleNamespace(process_tokens=lambda *a, **k: [(0,128,123)])
    def send(buf, copy):
        sent.append((buf, copy))
        c.reqs_status['r'] = 128
    c.push_sockets = [SimpleNamespace(send=send)]
    assert c.lookup([1]*128, 'r') == 128
    assert len(sent) == 1 and sent[0][1] is False
    assert sleeps == [0.001]
