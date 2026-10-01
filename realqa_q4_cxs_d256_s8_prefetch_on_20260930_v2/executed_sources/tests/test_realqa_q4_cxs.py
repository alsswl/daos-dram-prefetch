"""CPU-only test: preserve upstream CxS scheduling and real answer feedback."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import realqa_q4_cxs as bench
import pytest
import yaml


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


def test_upstream_cxs_history_and_counts(tmp_path):
    root = tmp_path
    case = root/'case'
    case.mkdir()
    (root/'documents').mkdir()
    (root/'documents/book.txt').write_text('Real document text')
    bench.dump(root/'plan.json',dict(max_model_len=32768,expected_requests=480))
    bench.dump(root/'sessions.json',[dict(index=i,file='book.txt') for i in range(80)])
    up = bench.upstream()
    state = dict(active=0,peak=0,calls=0,closed=0)

    class HTTP:
        def __init__(self,**kwargs): pass
        async def get(self,url): return NS(text='test metrics')
        async def aclose(self): pass

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
            yield NS(id=self.rid,choices=[NS(delta=NS(content='actual generated answer'))],usage=None)
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
        asyncio.run(bench.workload(root,case,lambda:None))
    calls = bench.read(case/'calls.json')
    assert len(calls)==480 and state['closed']==480
    assert 1 < state['peak'] <= 8 and state['active']==0
    assert len({(c['session_index'],c['turn']) for c in calls})==480
    assert all(c['cached_tokens']==8 and c['prompt_tokens']==10 for c in calls)
    assert len(list((case/'histories').glob('*.json')))==80
    assert all(len(json.loads(p.read_text()))==12 for p in (case/'histories').glob('*.json'))
