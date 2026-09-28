"""Comparison driver checks without a DAOS server or GPU."""
import json
from types import SimpleNamespace

import httpx
import pytest
import yaml

import compare_e2e as bench


def test_chunk_boundary_and_percentiles():
    assert bench.expected_hit({'tokens': [0] * 2048}, 2048) == 2047
    assert bench.expected_hit({'tokens': [0] * 2049}, 2048) == 2048
    assert bench.percentile([1, 2, 3], .5) == 2
    assert bench.percentile([4], .95) == 4


def test_configs_are_isolated_and_match(tmp_path):
    source = (bench.ROOT / 'lmcache_config_daosgds_unified.yaml').read_bytes()
    a = SimpleNamespace(repeats=2, chunk_size=2048, pool='pool', container='container',
                        gpu_buffer_gb=8, io_workers=8, meta_workers=8)
    paths = bench.configs(a, tmp_path, 'unique-id')
    namespaces = set()
    for (repeat, mode), path in paths.items():
        cfg = yaml.safe_load(path.read_text())
        ec = cfg['extra_config']
        namespaces.add(ec['daosgds.object_namespace'])
        assert ec['daosgds.transport'] == mode
        assert ec['daosgds.dfs_oclass'] == bench.OC_SX
        assert ec['daosgds.root'].startswith('/minji-compare-unique-id-')
        env = bench.environment(path, mode)
        assert env['LMCACHE_CONFIG_FILE'] == str(path)
        assert env['DAOS_PROBE_CHUNKS'] == env['DAOSGDR_TIMING'] == '0'
    assert len(namespaces) == 4
    assert (bench.ROOT / 'lmcache_config_daosgds_unified.yaml').read_bytes() == source


def stream_client(usage):
    events = [{'choices': [{'text': ''}]}, {'choices': [{'text': 'hello', 'token_ids': [11, 12]}]},
              {'choices': [], 'usage': usage}]
    body = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events)
    body += 'data: [DONE]\n\n'
    return httpx.Client(base_url='http://test', transport=httpx.MockTransport(
        lambda req: httpx.Response(200, text=body)))


def test_stream_latency_and_usage():
    usage = dict(prompt_tokens=3, completion_tokens=2,
                 prompt_tokens_details={'cached_tokens': 0})
    with stream_client(usage) as client:
        row = bench.request(client, {'id': 1, 'tokens': [1, 2, 3]},
                            SimpleNamespace(max_tokens=2))
    assert row['text'] == 'hello'
    assert 0 <= row['ttft_ms'] <= row['e2e_ms']
    assert row['cached_tokens'] == 0


@pytest.mark.parametrize('override', [
    {'prompt_tokens_details': None}, {'prompt_tokens': 4}, {'completion_tokens': 1},
])
def test_reject_unverifiable_or_changed_workload(override):
    usage = dict(prompt_tokens=3, completion_tokens=2,
                 prompt_tokens_details={'cached_tokens': 0})
    usage.update(override)
    with stream_client(usage) as client, pytest.raises(RuntimeError):
        bench.request(client, {'id': 1, 'tokens': [1, 2, 3]},
                      SimpleNamespace(max_tokens=2))


def test_fixed_contexts_are_exact_and_independent(monkeypatch):
    from transformers import AutoTokenizer

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(c) for c in text]

    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda _: Tokenizer())
    monkeypatch.setenv('HF_HOME', '/tmp/test-unused-cache')
    a = SimpleNamespace(model='unused', context_tokens=[513, 768],
                        max_model_len=1024, chunk_size=256)
    prompts, warm, domain = bench.workload(a)
    assert [len(p['tokens']) for p in prompts] == [513, 768]
    assert prompts[0]['tokens'][:256] != prompts[1]['tokens'][:256]
    assert all(warm['tokens'][:256] != p['tokens'][:256] for p in prompts)
    assert domain == 'fixed-token reference comparison'
