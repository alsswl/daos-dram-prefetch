"""Explicit DAOS-only experiment mode; no host KV allocation or copying."""
import threading

def configure_no_dram(cfg):
    cfg.update(local_cpu=False,max_local_cpu_size=0)
    cfg['extra_config'].update({'daosgds.dram_disabled':True,
        'daosgds.async_dram':False,'daosgds.dram_promote_on_read':False,
        'daosgds.dram_prefetch':False})
    return cfg

def validate_no_dram(config):
    ec=config.extra_config or {}
    if (config.local_cpu or config.max_local_cpu_size!=0 or
        any(ec.get('daosgds.'+key,False) for key in
            ('async_dram','dram_promote_on_read','dram_prefetch'))):
        raise ValueError('DRAM-disabled mode requires zero CPU KV capacity and all DRAM paths disabled')

class DisabledDramMirror:
    """Zero counters for existing telemetry; no thread, buffer, or retained ref.

    Keeping this interface also preserves the existing gather-stream barrier
    in gpu_store.put_direct, so disabling DRAM does not change that barrier.
    """
    def __init__(self,cpu):
        self.cpu=cpu;self.lock=threading.Lock()
        self.stats={key:0 for key in ('admitted','copied','copied_bytes','skipped_budget',
            'skipped_present','skipped_closed','skipped_expired','skipped_allocation',
            'errors','peak_pending_bytes','copy_ms','cpu_hit_chunks','read_submit_errors')}
        for origin in ('write','read'):
            self.stats.update({f'{origin}_{key}':0 for key in
                               ('offered','admitted','copied','copied_bytes')})
    def offer(self,*args,**kwargs):return False
    def snapshot(self):
        return dict(self.stats,enabled=False,pending_bytes=0,pending_chunks=0)
    def close(self):pass

def validate_no_dram_events(events):
    samples=[e for e in events if e['event']=='occupancy_sample']
    assert samples
    for e in samples:
        assert e['cpu_hot_bytes']==e['cpu_hot_chunks']==e['cpu_ready_bytes']==0
        m=e['dram_mirror'];assert m['enabled'] is False
        assert all(v==0 for k,v in m.items() if k!='enabled')
    assert not any(e['event']=='tier_lookup' and e['tier']=='dram' and e['hit_chunks'] for e in events)
    total=sum(e['chunks'] for e in events if e['event']=='window_retrieve_done')
    loaded=sum(e['returned_chunks'] for e in events if e['event']=='daos_demand_outcome')
    assert total==loaded and total>0
    return dict(passed=True,cpu_cache_peak_bytes=0,dram_copied_bytes=0,
                retrieved_chunks=total,daos_returned_chunks=loaded,
                daos_returned_gib=loaded*18/1024)
