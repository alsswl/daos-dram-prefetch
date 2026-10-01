"""Opt-in diagnostic only: observe V2 CPU/GPU counters after rollback.

Reads GPU scalar state (synchronizes); does not repair or mutate token counters.
Use only for fault diagnosis, not performance measurements.
"""
import functools
import json
import os


def observe_update(runner, output, original, emit):
    states=runner.req_states
    rolled=[]
    reqs=output.scheduled_cached_reqs
    for rid,new in zip(reqs.req_ids,reqs.num_computed_tokens):
        idx=states.req_id_to_index[rid]
        previous=int(states.num_computed_tokens_np[idx])
        if new<previous:
            rolled.append((rid,idx,previous,int(new)))
    result=original(runner,output)
    for rid,idx,previous,new in rolled:
        gpu=int(states.num_computed_tokens.gpu[idx].item())
        query=int(output.num_scheduled_tokens.get(rid,0))
        emit(dict(event='rollback_state',request_id=rid,previous_cpu=previous,
            scheduler_tokens=new,cpu_after=int(states.num_computed_tokens_np[idx]),
            gpu_after=gpu,scheduled_tokens=query,
            intended_seq_end=new+query,gpu_seq_end=gpu+query,
            request_max_seq_len=int(states.max_seq_len[idx])))
    return result


def install():
    if os.environ.get('DAOS_RECOMPUTE_STATE_PROBE')!='1':return
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    original=GPUModelRunner.update_requests
    if getattr(original,'_recompute_state_probe',False):return
    def emit(data):
        print('RECOMPUTE_STATE_PROBE '+json.dumps(dict(pid=os.getpid(),**data)),flush=True)
    @functools.wraps(original)
    def wrapper(self,output):return observe_update(self,output,original,emit)
    wrapper._recompute_state_probe=True
    GPUModelRunner.update_requests=wrapper
    emit(dict(event='installed'))
