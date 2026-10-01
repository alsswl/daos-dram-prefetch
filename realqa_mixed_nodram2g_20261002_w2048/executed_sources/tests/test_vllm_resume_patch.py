"""CPU-only regression tests against the installed adapter's real classes."""
from types import SimpleNamespace as NS
import pytest

from lmcache.integration.vllm.vllm_v1_adapter import RequestTracker, ReqMeta, LoadSpec
from lmcache_daos.vllm_resume_patch import patch_tracker


def request(prompt_len=62830, full_len=62958):
    return NS(req_id='resume-regression',prompt_token_ids=list(range(prompt_len)),
              prefill_token_ids=list(range(full_len)),
              block_ids=(list(range(4096)),),sampling_params=NS(extra_args=None),
              mm_features=[])


@pytest.fixture
def installed(monkeypatch):
    original=RequestTracker.from_new_request
    monkeypatch.setattr(RequestTracker,'from_new_request',staticmethod(original))
    events=[]
    patch_tracker(RequestTracker,events.append)
    return events


def metadata(tracker, expected):
    return ReqMeta.from_request_tracker(tracker,16,128,LoadSpec(0,expected,True),True,True)


def test_original_reproduces_one_missing_chunk():
    tracker=RequestTracker.from_new_request(None,request(),62849,62848,False)
    assert len(metadata(tracker,62848).token_ids)==62720


def test_v2_resume_restores_generated_history(installed):
    req=request()
    tracker=RequestTracker.from_new_request(None,req,62849,62848,False)
    meta=metadata(tracker,62848)
    assert tracker.prompt_len==62830
    assert tracker.token_ids==req.prefill_token_ids[:62849]
    assert len(meta.token_ids)==len(meta.slot_mapping)==62848
    assert meta.load_spec.lmcache_cached_tokens==62848
    assert tracker.num_lmcache_cached_tokens==62848
    assert tracker.allocated_block_ids==req.block_ids[0]
    assert installed[-1]['tracker_after']==62849
    assert req.prompt_token_ids==list(range(62830))


@pytest.mark.parametrize('v2',[False,True])
def test_new_request_unchanged(installed,v2):
    req=request(61454,61454)
    if not v2:del req.prefill_token_ids
    tracker=RequestTracker.from_new_request(None,req,61454,61440,False)
    assert len(metadata(tracker,61440).token_ids)==61440
    assert not installed


def test_full_hit_recompute_last_token(installed):
    req=request(62830,62848)
    tracker=RequestTracker.from_new_request(None,req,62848,62848,False)
    assert len(metadata(tracker,62848).token_ids)==62848


def test_short_and_mismatched_history_rejected(installed):
    req=request()
    with pytest.raises(RuntimeError,match='shorter'):
        RequestTracker.from_new_request(None,req,64000,62848,False)
    req.prefill_token_ids[0]=-1
    with pytest.raises(RuntimeError,match='does not match'):
        RequestTracker.from_new_request(None,req,62849,62848,False)


def test_install_idempotent(installed):
    first=RequestTracker.from_new_request
    patch_tracker(RequestTracker,lambda e:None)
    assert RequestTracker.from_new_request is first


def test_extension_still_in_original_chunk(installed):
    req=request(61454,61460)
    tracker=RequestTracker.from_new_request(None,req,61460,61440,False)
    assert tracker.prompt_len==61454 and len(tracker.token_ids)==61460
    assert len(metadata(tracker,61440).token_ids)==61440
