"""Checks based on allocator events and load/copy intervals, not sampling alone."""
def validate_split_events(events):
    enabled=[e for e in events if e['event']=='split_staging_enabled']
    assert len(enabled)==1
    cfg=enabled[0]
    assert cfg['pipeline_depth']==3
    for e in events:
        assert 0<=e['store_used_bytes']<=cfg['store_capacity_bytes']
        assert 0<=e['retrieve_used_bytes']<=cfg['retrieve_capacity_bytes']
        assert e['used_bytes']==e['store_used_bytes']+e['retrieve_used_bytes']
    failures=[e for e in events if e['event'] in ('allocate','batched_allocate') and e.get('failed')]
    assert not failures, 'Split arena allocation failed'
    active=set();peak=0;overlap=0;copies=set()
    for e in events:
        key=(e.get('request_id'),e.get('window'))
        if e['event']=='pipeline_load_start':
            active.add(key);peak=max(peak,len(active))
            if copies:overlap+=1
        elif e['event']=='pipeline_load_done':active.remove(key)
        elif e['event']=='window_copy_start':
            copies.add(key)
            if active:overlap+=1
        elif e['event']=='window_copy_done':copies.remove(key)
    assert not active and not copies
    assert 2<=peak<=cfg['pipeline_depth'], 'No concurrent retrieve loads observed'
    assert overlap>0, 'No overlap between window loading and GPU copy observed'
    return dict(passed=True,store_peak_gib=max(e['store_used_bytes'] for e in events)/2**30,
                retrieve_peak_gib=max(e['retrieve_used_bytes'] for e in events)/2**30,
                max_concurrent_window_loads=peak,load_copy_overlap_events=overlap,
                allocation_failures=0,**cfg)
