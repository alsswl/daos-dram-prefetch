import hashlib
from pathlib import Path

import pytest
import leval_prefetch as leval
import supervise_eqbench_matrix as supervisor


def documents():
    return [dict(document_id=f'd{i}',text=f'document {i}',questions=[
        dict(question=f'question {q}',prompt_tokens=100) for q in range(4)]) for i in range(16)]


def test_fixed_unique_questions_no_generated_history():
    docs=documents();rows=leval.interleave(docs)
    assert rows==leval.interleave(docs)
    assert len(rows)==64
    assert [r['index'] for r in rows]==list(range(64))
    assert len({r['prompt_sha256'] for r in rows})==64
    for doc in docs:
        subset=[r for r in rows if r['document_id']==doc['document_id']]
        assert [r['question_index'] for r in subset]==list(range(4))
        for row in subset:
            assert row['prompt'].startswith(leval.HEADER+doc['text'])
            assert row['prompt_sha256']==hashlib.sha256(row['prompt'].encode()).hexdigest()
    assert len({r['document_id'] for r in rows[:16]})>1


def test_12_cases_only_prefetch_changes():
    specs=leval.cases();assert len(specs)==12
    for repeat in (1,2,3):
        for c in (8,16):
            pair=[s for s in specs if s['repeat']==repeat and s['concurrency']==c]
            assert {s['prefetch'] for s in pair}=={False,True}
            assert all(s['cpu_gib']==s['staging_gib']==8 and not s['cancel_queued'] for s in pair)
    assert specs[0]['prefetch'] is False and specs[4]['prefetch'] is True
    assert leval.cases_from_plan(dict(workload_kind='leval_document_qa',cases=specs))==specs
    with pytest.raises(AssertionError):leval.cases_from_plan(dict(workload_kind='leval_document_qa',cases=[]))


def test_supervisor_selects_correct_driver():
    driver,script=supervisor.driver_for(Path('/root/discos_minji/leval_prefetch_test'))
    assert driver is leval and script.name=='leval_prefetch.py'
    _,script=supervisor.driver_for(Path('/root/discos_minji/eqbench_matrix_test'))
    assert script.name=='eqbench_longform_matrix.py'
    with pytest.raises(AssertionError):supervisor.driver_for(Path('/root/other'))


def test_chat_template_has_no_assistant_history():
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            assert messages==[dict(role='user',content='hello')]
            assert kwargs['enable_thinking'] is False
            assert kwargs['return_dict'] is False
            return [1,2,3]
    assert leval.tokenize(Tokenizer(),'hello')==[1,2,3]


def test_original_duplicate_questions_are_skipped_in_order():
    row=dict(instructions=['a','b','b','c','d'],outputs=[1,2,2,3,4])
    assert leval.distinct_questions(row)==[(0,'a',1),(1,'b',2),(3,'c',3),(4,'d',4)]
    with pytest.raises(AssertionError):leval.distinct_questions(dict(row,outputs=[1]))


def test_worker_setting_changes_only_worker_count_and_keeps_legacy_default():
    assert leval.worker_count({})==1
    for invalid in (0,3,True,'2'):
        with pytest.raises(AssertionError):leval.worker_count(dict(dram_prefetch_workers=invalid))
    for spec in leval.cases():
        one=leval.config_for(spec,'fixed',dict(dram_prefetch_workers=1))
        two=leval.config_for(spec,'fixed',dict(dram_prefetch_workers=2))
        assert one['extra_config'].pop('daosgds.dram_prefetch_workers')==1
        assert two['extra_config'].pop('daosgds.dram_prefetch_workers')==2
        assert one==two


def test_reuse_completed_run_preserves_inputs(tmp_path,monkeypatch):
    import json
    def save(path,value):path.write_text(json.dumps(value))
    source=tmp_path/'leval_prefetch_original';source.mkdir()
    monkeypatch.setattr(leval,'ROOT',tmp_path)
    monkeypatch.setattr(leval,'snapshot',lambda root:{'new_source':'checksum'})
    monkeypatch.setattr(leval,'report',lambda root:None)
    save(source/'status.json',dict(status='completed'))
    for name in ('requests','documents','selection_audit','sources'):save(source/(name+'.json'),[])
    (source/'source_data').mkdir()
    plan=dict(workload_kind='leval_document_qa',cases=leval.cases(),requests_per_case=72,
        total_requests=864,notes=[],source_revisions=['old'],
        requests_sha256=leval.digest(source/'requests.json'),documents_sha256=leval.digest(source/'documents.json'))
    save(source/'plan.json',plan)
    target=tmp_path/'leval_prefetch_two'
    leval.prepare_from_run(target,source,2)
    assert (target/'requests.json').read_bytes()==(source/'requests.json').read_bytes()
    actual=json.loads((target/'plan.json').read_text())
    assert actual['dram_prefetch_workers']==2 and actual['baseline_dram_prefetch_workers']==1
    assert actual['cases']==plan['cases'] and 'source_revisions' not in actual
    with pytest.raises(FileExistsError):leval.prepare_from_run(target,source,2)
