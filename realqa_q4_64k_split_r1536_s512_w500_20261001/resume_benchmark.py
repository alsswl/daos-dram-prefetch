"""Resume after completed correctness gate; preserve the original runner log."""
import fcntl
import os
from pathlib import Path
import time
import realqa_q4_cxs as qa

root=Path(__file__).resolve().parent
with (root/'launch.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=qa.read(root/'plan.json')
    for name,digest in plan['source_sha256'].items():
        assert qa.base.digest(qa.ROOT/name)==digest, f'Implementation changed: {name}'
    gate=root/'gpu_9gib_check'
    result=qa.read(gate/'result.json')
    assert result['status']=='passed' and result['all_returned_gpu_values_match']
    assert result['returned_tokens']==65536 and result['capacity_failed_chunks']==0
    assert result['split_validation']['passed'] and qa.read(gate/'cleanup.json')['remaining']==0
    qa.dump(root/'pre_resume_status.json',qa.read(root/'status.json'))
    for attempt in range(30):
        try:
            qa.base.storage_guard(root,plan['capacity']['stored_kv_upper_gib'])
            break
        except RuntimeError as exc:
            qa.dump(root/'status.json',dict(status='waiting_for_daos_reclaim',attempt=attempt+1,error=str(exc)))
            print(f'Waiting for deleted gate objects to reclaim: {exc}',flush=True)
            time.sleep(10)
    else:
        raise RuntimeError('Space guard still blocked after reclamation wait')
    assert qa.early.idle_gpu()
    qa.run(root)
