import threading
import time

import pytest

from prepare_sharegpt_scale import history_pairs, prefix_chunks
from sharegpt_scale_experiment import session_requests


def test_prefix_identity_includes_earlier_chunks():
    assert len(prefix_chunks(list(range(256)))) == 2
    assert prefix_chunks(list(range(128))) <= prefix_chunks(list(range(256)))
    assert len(prefix_chunks(list(range(127)))) == 0
    a = [0]*128+[1]*128
    b = [2]*128+[1]*128
    assert not prefix_chunks(a) & prefix_chunks(b)


def test_original_history_and_roles():
    entry = {'conversations': [
        {'from':'human','value':'Q1'}, {'from':'gpt','value':'A1'},
        {'from':'human','value':'Q2'}, {'from':'gpt','value':'A2'}]}
    assert history_pairs(entry,2) == [('Q1','A1'),('Q1\nA1\nQ2','A2')]
    assert history_pairs(entry,3) is None
    entry['conversations'][1]['from'] = 'human'
    assert history_pairs(entry,2) is None


def test_rolling_sessions_obey_dependencies_and_limit():
    records = [dict(index=t*3+s,session=s,turn=t) for t in range(4) for s in range(3)]
    active, finished, peaks, saved = set(), {}, [], []
    lock = threading.Lock()
    def invoke(r, barrier):
        barrier.wait()
        with lock:
            assert r['session'] not in active
            assert r['turn'] == finished.get(r['session'],0)
            active.add(r['session'])
            peaks.append(len(active))
        time.sleep(.002)
        with lock:
            active.remove(r['session'])
            finished[r['session']] = r['turn']+1
        return dict(r,end_ns=time.time_ns())
    session_requests(records,2,invoke,saved.append)
    assert sorted(r['index'] for r in saved) == list(range(12))
    assert max(peaks) == 2
    assert finished == {0:4,1:4,2:4}


def test_invalid_schedule_and_error_stop():
    with pytest.raises(RuntimeError, match='Invalid'):
        session_requests([dict(index=0,session=0,turn=1)],1,lambda r,b:None,lambda r:None)
    calls = []
    def fail(r,b):
        calls.append(r)
        return dict(r,error='test',end_ns=time.time_ns())
    with pytest.raises(RuntimeError, match='Request failed'):
        session_requests([dict(index=i,session=i,turn=0) for i in range(3)],1,fail,lambda r:None)
    assert len(calls) == 1
