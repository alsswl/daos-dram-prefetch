import json
from contextlib import contextmanager
from types import SimpleNamespace

from compare_govreport import request


def test_govreport_has_natural_eos_and_standard_cap():
    class Response:
        def raise_for_status(self): pass
        def iter_lines(self):
            yield 'data: ' + json.dumps(dict(id='chatcmpl-test', choices=[
                dict(delta=dict(content='Summary'), finish_reason='stop')], usage=dict(
                    prompt_tokens=42, completion_tokens=3, prompt_tokens_details=dict(cached_tokens=0))))
            yield 'data: [DONE]'
    class Client:
        @contextmanager
        def stream(self, method, url, json):
            assert json['max_tokens'] == 512 and json['stop'] == []
            assert not json.get('ignore_eos') and not json.get('min_tokens')
            assert json['chat_template_kwargs']['enable_thinking'] is False
            yield Response()
    record=dict(index=0, prompt='GovReport', prompt_sha256='test', expected_prompt_tokens=42)
    barrier=SimpleNamespace(wait=lambda timeout: None)
    row=request(Client(), record, barrier)
    assert 'error' not in row and row['completion_tokens'] == 3
    record['expected_prompt_tokens']=41
    assert 'Tokenization mismatch' in request(Client(),record,barrier)['error']
