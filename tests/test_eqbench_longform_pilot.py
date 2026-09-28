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
