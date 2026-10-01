def validate_capacity_events(events):
    config=[e for e in events if e['event']=='capacity_pipeline_enabled']
    assert len(config)==1
    cfg=config[0];active=set();copying=set();peak=overlap=0
    for e in events:
        assert e['used_bytes']==e['store_used_bytes']+e['retrieve_used_bytes']
        assert 0<=e['used_bytes']<=cfg['capacity_bytes']
        if not cfg['shared']:
            assert e['store_used_bytes']<=cfg['store_capacity_bytes']
            assert e['retrieve_used_bytes']<=cfg['retrieve_capacity_bytes']
        if e['event'] in ('allocate','batched_allocate'):assert not e.get('failed')
        key=(e.get('request_id'),e.get('window'))
        if e['event']=='pipeline_load_start':
            active.add(key);peak=max(peak,len(active));overlap+=bool(copying)
        elif e['event']=='pipeline_load_done':active.remove(key)
        elif e['event']=='window_copy_start':copying.add(key);overlap+=bool(active)
        elif e['event']=='window_copy_done':copying.remove(key)
    assert not active and not copying
    assert 2<=peak<=cfg['pipeline_depth'] and overlap>0
    return dict(passed=True,shared=cfg['shared'],pipeline_depth=cfg['pipeline_depth'],
        max_concurrent_loads=peak,load_copy_overlap_events=overlap,allocation_failures=0,
        store_peak_gib=max(e['store_used_bytes'] for e in events)/2**30,
        retrieve_peak_gib=max(e['retrieve_used_bytes'] for e in events)/2**30)
