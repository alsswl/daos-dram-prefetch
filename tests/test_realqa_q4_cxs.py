"""CPU-only test: preserve upstream CxS scheduling and real answer feedback."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import realqa_q4_cxs as bench
import pytest
import yaml


def test_unwindowed_ten_gib_config_and_validation(tmp_path,monkeypatch):
    from prepare_windowed_config import make_config
    cfg=make_config(0,10,256)
    assert cfg['extra_config']['daosgds.retrieve_window_mib']==0
    assert cfg['extra_config']['daosgds.gpu_buffer_gb']==10
    assert not cfg['enable_async_loading']
    assert not cfg['extra_config']['daosgds.dram_prefetch']
    events=[dict(event='windowed_demand_enabled',window_mib=0,effective_window_bytes=0),
            dict(event='daos_demand_outcome',other_failed_chunks=0,capacity_failed_chunks=0)]
    monkeypatch.setattr(bench.base,'read_events',lambda _:events)
    bench.validate_windowed_run(tmp_path)
    assert bench.read(tmp_path/'window_policy_check.json')['copy_windows']==0
    events.append(dict(event='window_copy_start',bytes=18*2**20))
    with pytest.raises(AssertionError,match='Unwindowed baseline'):
        bench.validate_windowed_run(tmp_path)


def test_64k_server_context_not_hardcoded():
    for maximum in (32768,65536):
        args=bench.server_args(dict(model=bench.MODEL,max_model_len=maximum,max_num_seqs=8))
        assert args.max_model_len==maximum
        assert args.max_num_seqs==8 and args.model==bench.MODEL


@pytest.mark.parametrize('failed_phase',['retrieve','store'])
def test_store_validation_distinguishes_retried_read_from_dropped_write(tmp_path,monkeypatch,failed_phase):
    events=[dict(event='windowed_demand_enabled',window_mib=500,effective_window_bytes=500*2**20),
            dict(event='windowed_store_enabled',window_mib=500,effective_window_bytes=500*2**20),
            dict(event='window_retrieve_start',request_id='r')]
    if failed_phase=='retrieve':events.append(dict(event='allocate',failed=True))
    events += [dict(event='window_copy_start',bytes=18*2**20),
               dict(event='daos_demand_outcome',other_failed_chunks=0,capacity_failed_chunks=int(failed_phase=='retrieve')),
               dict(event='window_retrieve_done',request_id='r'),
               dict(event='store_window_request_start',request_id='s',chunks=1)]
    if failed_phase=='store':events.append(dict(event='allocate',failed=True))
    events += [dict(event='store_window_copy_start',bytes=18*2**20),
               dict(event='store_window_submitted',chunks=1),
               dict(event='store_window_request_done',request_id='s',tokens=128)]
    monkeypatch.setattr(bench.base,'read_events',lambda _:events)
    if failed_phase=='store':
        with pytest.raises(AssertionError,match='allocation failure'):bench.validate_windowed_run(tmp_path)
    else:
        bench.validate_windowed_run(tmp_path)
        data=bench.read(tmp_path/'store_window_policy_check.json')
        assert data['store_allocation_failures']==0 and data['retrieve_allocation_failures']==1


@pytest.mark.parametrize('util',[0.75,0.835])
def test_vllm_memory_budget_propagates(util):
    import compare_e2e
    args=bench.server_args(dict(model=bench.MODEL,max_model_len=65536,
        max_num_seqs=8,gpu_memory_utilization=util))
    cmd=compare_e2e.server_command(args)
    assert float(cmd[cmd.index('--gpu-memory-utilization')+1])==util
    del args.gpu_memory_utilization
    cmd=compare_e2e.server_command(args)
    assert cmd[cmd.index('--gpu-memory-utilization')+1]=='0.75'
    args.gpu_memory_utilization=float('nan')
    with pytest.raises(ValueError):compare_e2e.server_command(args)


def test_prefetch_config_only_changes_two_flags(tmp_path):
    old=bench.three.config('none')
    folder=tmp_path/'c8_s10_none'
    folder.mkdir()
    (folder/'config.yaml').write_text(yaml.safe_dump(old))
    plan=dict(baseline=str(tmp_path))
    new=bench.three.config('both')
    bench.validate_comparison_config(plan,new)
    for tier in ('dram','daos'):
        assert old['extra_config'][f'daosgds.early_{tier}_prefetch'] is False
        assert new['extra_config'][f'daosgds.early_{tier}_prefetch'] is True
    new['max_local_cpu_size']=128
    with pytest.raises(AssertionError,match='Unexpected non-prefetch'):
        bench.validate_comparison_config(plan,new)


@pytest.mark.parametrize('depth,mixed,replay',[(10,False,False),(2,False,False),(2,True,False),(2,True,True)])
def test_upstream_cxs_history_and_counts(tmp_path,depth,mixed,replay):
    root = tmp_path
    case = root/'case'
    case.mkdir()
    (root/'documents').mkdir()
    (root/'documents/book.txt').write_text('Real document text')
    sessions=8*depth;expected=sessions*6
    bench.dump(root/'plan.json',dict(max_model_len=32768,expected_requests=expected,
                                   concurrency=8,session_depth=depth,mixed_schedule=mixed))
    bench.dump(root/'sessions.json',[dict(index=i,file='book.txt') for i in range(sessions)])
    if mixed:
        from prepare_cxs_light import light_schedule
        schedule=light_schedule()
        assert all(len(lane)==12 for lane in schedule['lanes'])
        bench.dump(root/'mixed_schedule.json',schedule)
    up = bench.upstream()
    reference=None
    if replay:
        reference=root/'reference';reference.mkdir()
        messages=[]
        for prompt in [up.FIRST_PROMPT.format('Real document text')]+up.FOLLOWUP_PROMPTS[:5]:
            messages.extend([dict(role='user',content=prompt),dict(role='assistant',content='actual generated answer')])
        for i in range(sessions):bench.dump(reference/f'{i:03d}.json',messages)
    state = dict(active=0,peak=0,calls=0,closed=0)

    class HTTP:
        def __init__(self,**kwargs): pass
        async def get(self,url): return NS(text='test metrics')
        async def aclose(self): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass

    class Usage:
        completion_tokens = 1
        prompt_tokens = 10
        def model_dump(self):
            return dict(completion_tokens=1,prompt_tokens=10,
                        prompt_tokens_details=dict(cached_tokens=8))

    class Stream:
        def __init__(self,rid): self.rid=rid
        def __aiter__(self): return self.generate()
        async def generate(self):
            await asyncio.sleep(0)
            yield NS(id=self.rid,choices=[NS(delta=NS(content='different replay answer' if replay else 'actual generated answer'))],usage=None)
            yield NS(id=self.rid,choices=[],usage=Usage())
        async def close(self):
            state['active']-=1
            state['closed']+=1

    async def create(**kwargs):
        history = kwargs['messages']
        assert len(history)%2==1
        assert all(m['content']=='actual generated answer' for m in history[1::2])
        assert kwargs['temperature']==0 and kwargs['max_tokens']==256
        state['calls']+=1
        state['active']+=1
        state['peak']=max(state['peak'],state['active'])
        return Stream(f'req-{state["calls"]}')

    fake_client = NS(chat=NS(completions=NS(create=create)))
    fake_tok = NS(apply_chat_template=lambda *a,**kw:list(range(10)))
    with patch.object(bench,'upstream',return_value=up), \
         patch.object(bench,'tokenizer',return_value=fake_tok), \
         patch.object(up.openai,'AsyncOpenAI',return_value=fake_client), \
         patch.object(up.httpx,'AsyncClient',HTTP):
        asyncio.run(bench.workload(root,case,lambda:None,replay_histories=reference))
    calls = bench.read(case/'calls.json')
    assert len(calls)==expected and state['closed']==expected
    assert 1 < state['peak'] <= 8 and state['active']==0
    assert len({(c['session_index'],c['turn']) for c in calls})==expected
    assert all(c['cached_tokens']==8 and c['prompt_tokens']==10 for c in calls)
    assert len(list((case/'histories').glob('*.json')))==sessions
    assert all(len(json.loads(p.read_text()))==12 for p in (case/'histories').glob('*.json'))
