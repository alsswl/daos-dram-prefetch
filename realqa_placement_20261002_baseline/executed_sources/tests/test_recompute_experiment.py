import json
import pytest
from recompute_experiment_support import RecomputeHealth
from realqa_q4_cxs import server_args
from compare_e2e import server_command


def test_policy_explicit_and_default():
    p=dict(model='test',max_model_len=65536,max_num_seqs=8)
    for policy in ('fail','recompute'):
        p['kv_load_failure_policy']=policy
        cmd=server_command(server_args(p))
        assert json.loads(cmd[cmd.index('--kv-transfer-config')+1])['kv_load_failure_policy']==policy
    del p['kv_load_failure_policy']
    assert server_args(p).kv_load_failure_policy=='fail'


def test_scheduler_option_is_opt_in():
    a=server_args(dict(model='test',max_model_len=65536,max_num_seqs=8))
    assert '--no-async-scheduling' not in server_command(a)
    a.async_scheduling=False
    assert '--no-async-scheduling' in server_command(a)
    a.async_scheduling=True
    assert '--async-scheduling' in server_command(a)
    a.async_scheduling='false'
    with pytest.raises(ValueError):server_command(a)


def test_allow_only_known_partial_load_errors(tmp_path):
    p=tmp_path/'server.log'
    watcher=RecomputeHealth(p)
    suffix=' (vllm_v1_adapter.py:881:lmcache.integration.vllm.vllm_v1_adapter)\n'
    p.write_text('(EngineCore pid=1) \x1b[31mLMCache ERROR:\x1b[0m Request req1The number of retrieved tokens is less than the expected number of tokens! This should not happen!'+suffix+
        'LMCache ERROR: Num retrieved tokens: 128, num expected tokens: 256'+suffix)
    watcher()
    assert watcher.allowed_lines==2
    watcher()
    assert watcher.allowed_lines==2
    with p.open('a') as f:f.write('LMCache ERROR: Unknown bug\n')
    with pytest.raises(RuntimeError,match='Unexpected'):watcher()


@pytest.mark.parametrize('message',['CUDA out of memory','Double free','DER_NOSPACE',
    'Traceback (most recent call last)','failure_policy=fail','[libdaosgdr] daos_obj_update_gpu rc=-1'])
def test_fatal_errors_not_allowed(tmp_path,message):
    p=tmp_path/'server.log';p.write_text(message+'\n')
    with pytest.raises(RuntimeError):RecomputeHealth(p)()


def test_partial_lines(tmp_path):
    p=tmp_path/'server.log';p.write_text('LMCache ER')
    watcher=RecomputeHealth(p);watcher()
    with p.open('a') as f:f.write('ROR: Unknown bug\n')
    with pytest.raises(RuntimeError):watcher()
