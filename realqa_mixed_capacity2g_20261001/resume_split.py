"""Resume post-shared cleanup with the DAOS CLI environment loaded."""
import fcntl
from pathlib import Path
import subprocess
import sys
import realqa_mixed_comparison as comparison
qa=comparison.qa
folder=Path(__file__).resolve().parent
with (folder/'runner.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cases=qa.read(folder/'comparison_plan.json')['cases']
    shared,split=(Path(c['root']) for c in cases)
    comparison.check_sources(shared);plan=comparison.check_sources(split)
    assert qa.read(shared/'status.json')['status']=='completed'
    assert qa.read(split/'status.json')['status']=='prepared'
    assert qa.early.idle_gpu()
    qa.dump(folder/'post_shared_cleanup_initial_failure.json',qa.read(folder/'status.json'))
    partial=shared/'comparison_cache_cleanup'
    assert set(p.name for p in partial.iterdir())=={'before_keys.json','target_keys.json'}
    partial.rename(shared/'comparison_cache_cleanup_initial_plan')
    comparison.cleanup_generated(shared)
    comparison.wait_space(split,plan['capacity']['stored_kv_upper_gib'])
    qa.dump(folder/'status.json',dict(status='running',condition='split',index=2,total=2,
        resumed_after='DAOS CLI PATH corrected; shared measurement and all executed sources unchanged'))
    with (split/'benchmark.log').open('x') as log:
        subprocess.run([sys.executable,str(qa.ROOT/'realqa_q4_cxs.py'),'run','--output',str(split)],
                       cwd=qa.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert qa.read(split/'status.json')['status']=='completed'
    qa.dump(folder/'status.json',dict(status='completed',conditions=2,requests=960))
