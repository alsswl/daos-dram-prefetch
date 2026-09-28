import threading
import time

import pytest

from discovery_rolling_replay import LogHealth, rolling_requests


def test_refills_while_an_initial_request_is_still_running():
    replacement_started=threading.Event()
    lock=threading.Lock(); state=dict(active=0,peak=0); rows=[]
    def invoke(record,barrier):
        barrier.wait(timeout=3)
        with lock:
            state['active']+=1; state['peak']=max(state['peak'],state['active'])
        if record==0:
            assert replacement_started.wait(3), 'Waiting for the whole initial group prevents refill'
        elif record==2: replacement_started.set()
        with lock: state['active']-=1
        return dict(index=record,end_ns=time.time_ns())
    rolling_requests(range(12),2,invoke,rows.append)
    assert sorted(r['index'] for r in rows)==list(range(12))
    assert state['peak']<=2 and state['active']==0
    assert replacement_started.is_set()


def test_failed_request_stops_refilling_and_preserves_inflight():
    admitted=[]; results=[]
    def invoke(record,barrier):
        admitted.append(record); barrier.wait(timeout=3)
        return dict(index=record,error='simulated failure',end_ns=time.time_ns())
    with pytest.raises(RuntimeError,match='no retry'):
        rolling_requests(range(10),2,invoke,results.append)
    assert sorted(admitted)==[0,1]
    assert sorted(r['index'] for r in results)==[0,1]


def test_empty_and_short_inputs():
    results=[]
    def invoke(record,barrier):
        barrier.wait(timeout=3); return dict(index=record,end_ns=time.time_ns())
    rolling_requests([],8,invoke,results.append)
    rolling_requests([7],8,invoke,results.append)
    assert results[0]['index']==7 and len(results)==1
    with pytest.raises(ValueError): rolling_requests([],0,invoke,results.append)


def test_storage_error_split_across_reads_is_detected(tmp_path):
    log=tmp_path/'server.log'; log.write_text('healthy\n')
    health=LogHealth(log); health()
    with log.open('a') as f: f.write('DER_NO')
    health()
    with log.open('a') as f: f.write('SPACE(-1007)')
    with pytest.raises(RuntimeError,match='benchmark invalid'): health()
