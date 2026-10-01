from contextlib import contextmanager
import json

from eqbench_longform_pilot import build_messages, load_templates, common_prefix, streaming_call, UPSTREAM


def test_original_templates_and_append_only_history():
    templates=load_templates(UPSTREAM/'data')
    assert set(templates)==set(range(1,14))
    assert 'chapter 1' in templates[6] and 'chapter 8' in templates[13]
    assert all(f'chapter {i-5}' in templates[i] for i in range(7,13))
    story={'writing_prompt':'A story'}; outputs={str(i):f'actual generated answer {i}' for i in range(1,14)}
    for step in range(1,14):
        messages=build_messages(story,templates,outputs,step)
        assert len(messages)==2*step-1
        assert all('{n_chapters}' not in m['content'] and '{writing_prompt}' not in m['content'] for m in messages)
        if step>1:
            assert messages[:-2]==build_messages(story,templates,outputs,step-1)
            assert messages[-2]==dict(role='assistant',content=outputs[str(step-1)])


def test_prefix_length():
    assert common_prefix([], [1,2])==0
    assert common_prefix([1,2],[1,2,3])==2
    assert common_prefix([1,2,4],[1,2,3])==2


def test_four_chapters_finish_at_ninth_call():
    templates=load_templates(UPSTREAM/'data', chapters=4)
    assert list(templates)==list(range(1,10))
    assert 'finish the story with chapter 4' in templates[9]
    outputs={str(i):f'answer {i}' for i in range(1,9)}
    messages=build_messages({'writing_prompt':'test'},templates,outputs,9)
    assert '4 chapters' in messages[0]['content']
    assert len(messages)==17
    assert [m['content'] for m in messages if m['role']=='assistant']==list(outputs.values())


def test_short_run_counts_cap_and_indices(tmp_path, monkeypatch):
    import threading
    import eqbench_longform_pilot as pilot
    templates=load_templates(UPSTREAM/'data', chapters=4)
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):return [1]*len(messages)
    def call(client, messages, seed, max_tokens):
        assert max_tokens==2500
        return dict(output='answer',prompt_tokens=len(messages),completion_tokens=1,
            output_words=1,cached_tokens=0,ttft_ms=1)
    monkeypatch.setattr(pilot,'streaming_call',call)
    pilot.run_story(None,Tokenizer(),'16',{'writing_prompt':'test'},templates,
        tmp_path,threading.Barrier(1),threading.Event(),2500,32768)
    folder=tmp_path/'stories/16'
    assert json.loads((folder/'status.json').read_text())=={'status':'completed','requests':9}
    calls=json.loads((folder/'calls.json').read_text())
    assert [r['index'] for r in calls]==list(range(135,144))
    assert calls[-1]['chapter']==4


def test_actual_context_guard_never_sends_oversized_request(tmp_path, monkeypatch):
    import threading
    import pytest
    import eqbench_longform_pilot as pilot
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):return [1]*31000
    monkeypatch.setattr(pilot,'streaming_call',lambda *a,**kw:pytest.fail('must not send'))
    with pytest.raises(RuntimeError,match='31000\\+2500>32768'):
        pilot.run_story(None,Tokenizer(),'1',{'writing_prompt':'test'},load_templates(UPSTREAM/'data',4),
            tmp_path,threading.Barrier(1),threading.Event(),2500,32768)


def test_stream_does_not_force_output_or_retry():
    messages=[dict(role='user',content='hello')]
    class Response:
        def raise_for_status(self):pass
        def iter_lines(self):
            yield 'data: '+json.dumps(dict(id='test',choices=[dict(delta=dict(content='Short.'),finish_reason='stop')],
                usage=dict(prompt_tokens=20,completion_tokens=2,prompt_tokens_details=dict(cached_tokens=0))))
            yield 'data: [DONE]'
    class Client:
        @contextmanager
        def stream(self,method,path,json):
            assert json['messages']==messages
            assert json['temperature']==.7 and json['min_p']==.1 and json['max_tokens']==4000
            assert 'ignore_eos' not in json and 'min_tokens' not in json and 'stop' not in json
            yield Response()
    row=streaming_call(Client(),messages,101)
    assert 'error' not in row and row['output']=='Short.' and row['upstream_short_response']
