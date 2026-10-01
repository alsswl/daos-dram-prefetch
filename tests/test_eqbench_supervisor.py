import json
from pathlib import Path
import pytest
import supervise_eqbench_matrix as supervisor


def test_failure_classification_does_not_mask_scientific_failures():
    assert supervisor.classify_failure(-9,'')=='terminated_by_signal'
    assert supervisor.classify_failure(1,'ReadTimeout')=='transport_or_runner_interruption'
    assert supervisor.classify_failure(1,'',True)=='stalled'
    for text in ('CUDA out of memory','AssertionError','Context budget exceeded','TypeError'):
        assert supervisor.classify_failure(-9,text)=='needs_investigation'
    assert supervisor.classify_failure(1,'DER_NOSPACE')=='storage_unavailable'
    assert supervisor.classify_failure(1,'unknown failure')=='needs_investigation'


def test_atomic_status_and_empty_progress(tmp_path):
    supervisor.dump(tmp_path/'status.json',dict(current='case',completed=['ok']))
    s=supervisor.progress(tmp_path)
    assert s['current']=='case' and s['completed_calls']==0
    assert not (tmp_path/'status.json.tmp').exists()


def test_complete_measurement_not_archived(tmp_path,monkeypatch):
    case=tmp_path/'case';case.mkdir()
    supervisor.dump(case/'phase.json',dict(end_ns=123))
    supervisor.dump(case/'final_sample.json',{})
    monkeypatch.setattr(supervisor,'stop_case_server',lambda p:pytest.fail('must not stop'))
    assert supervisor.archive_failed(tmp_path,case,'test') is None
    assert case.exists()


def test_only_exact_namespace_cleanup_receipt_is_accepted(tmp_path):
    supervisor.dump(tmp_path/'identity.json',dict(namespace='one:'))
    f=tmp_path/'interrupted_kv_cleanup'
    supervisor.dump(f/'result.json',dict(preserved_set_unchanged=True,remaining_targets=0))
    supervisor.dump(f/'manifest.json',dict(namespaces=[dict(namespace='other:')]))
    assert supervisor.cleanup_verified(tmp_path) is None
    supervisor.dump(f/'manifest.json',dict(namespaces=[dict(namespace='one:')]))
    assert supervisor.cleanup_verified(tmp_path)==f/'result.json'
