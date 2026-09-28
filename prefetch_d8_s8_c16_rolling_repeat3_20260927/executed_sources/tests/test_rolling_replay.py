import threading
import time

import pytest

from discovery_rolling_replay import LogHealth, rolling_requests


@pytest.mark.parametrize('concurrency',[2,8,16])
def test_refills_while_an_initial_request_is_still_running(concurrency):
    replacement_started=threading.Event()
    lock=threading.Lock(); state=dict(active=0,peak=0); rows=[]
    def invoke(record,barrier):
        barrier.wait(timeout=3)
        with lock:
            state['active']+=1; state['peak']=max(state['peak'],state['active'])
        if record==0:
            assert replacement_started.wait(3), 'Waiting for the whole initial group prevents refill'
        elif record==concurrency: replacement_started.set()
        with lock: state['active']-=1
        return dict(index=record,end_ns=time.time_ns())
    rolling_requests(range(concurrency*3),concurrency,invoke,rows.append)
    assert sorted(r['index'] for r in rows)==list(range(concurrency*3))
    assert state['peak']<=concurrency and state['active']==0
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


def test_failed_preflight_never_starts_benchmark(tmp_path,monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    import repeat_prefetch_c8
    import discovery_rolling_replay
    folder=tmp_path/'failed-preflight'
    monkeypatch.setattr(sys,'argv',['repeat_prefetch_c8.py','--arrival-mode','rolling','--output',str(folder)])
    monkeypatch.setattr(repeat_prefetch_c8.subprocess,'run',lambda *args,**kw:
                        SimpleNamespace(stdout='simulated probe',stderr='DER_NOSPACE',returncode=1))
    def forbidden(*args,**kw): raise AssertionError('Benchmark started despite storage failure')
    monkeypatch.setattr(discovery_rolling_replay,'run_case',forbidden)
    with pytest.raises(RuntimeError,match='preflight failed'):
        repeat_prefetch_c8.main()
    assert json.loads((folder/'status.json').read_text())['status']=='failed'
    assert not (folder/'r1_off').exists()
