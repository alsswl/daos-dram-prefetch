from types import SimpleNamespace as NS
from lmcache_daos.recompute_state_probe import observe_update


def test_probe_observes_does_not_fix_rollback():
    states=NS(req_id_to_index={'r':0},num_computed_tokens_np=[61440],
              num_computed_tokens=NS(gpu=[NS(item=lambda:61454)]),max_seq_len=[61710])
    runner=NS(req_states=states)
    out=NS(scheduled_cached_reqs=NS(req_ids=['r'],num_computed_tokens=[14464]),num_scheduled_tokens={'r':8192})
    def original(r,o):r.req_states.num_computed_tokens_np[0]=14464;return 7
    events=[]
    assert observe_update(runner,out,original,events.append)==7
    assert events[0]['cpu_after']==14464 and events[0]['gpu_after']==61454
    assert events[0]['intended_seq_end']==22656 and events[0]['gpu_seq_end']==69646
    assert states.num_computed_tokens.gpu[0].item()==61454
