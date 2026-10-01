"""Opt-in fix for text-only V2 resumed requests entering scheduled_new_reqs.

Keep site-packages unchanged. Disable DAOS_VLLM_RESUME_TOKEN_FIX to roll back.
vLLM sends the full current history in prefill_token_ids; prompt_token_ids
remains the original prompt. Preserve prompt_len and all other native fields.
"""
import functools
import inspect
import json
import os


def patch_tracker(tracker_class, emit):
    original = tracker_class.from_new_request
    if getattr(original, '_daos_resume_token_fix', False):
        return
    expected = ('lmcache_config', 'new_request', 'num_tokens_to_compute',
                'lmcache_cached_tokens', 'skip_save')
    if tuple(inspect.signature(original).parameters) != expected:
        raise RuntimeError('Unsupported LMCache RequestTracker signature; review resume fix')

    @functools.wraps(original)
    def create(lmcache_config, new_request, num_tokens_to_compute,
               lmcache_cached_tokens, skip_save):
        full = getattr(new_request, 'prefill_token_ids', None)
        prompt = new_request.prompt_token_ids
        tracker = original(lmcache_config, new_request, num_tokens_to_compute,
                           lmcache_cached_tokens, skip_save)
        # Legacy runner and ordinary new requests keep the original behavior.
        if full is None or prompt is None or len(full) <= len(prompt):
            return tracker
        if tracker.mm_hashes or getattr(new_request, 'mm_features', None):
            raise RuntimeError('DAOS resume token fix is validated for text-only requests')
        if list(full[:len(prompt)]) != list(prompt):
            raise RuntimeError('Resumed full token history does not match original prompt')
        needed = max(num_tokens_to_compute, lmcache_cached_tokens)
        if needed > len(full):
            raise RuntimeError('Resumed token history is shorter than scheduled/load range')
        before = len(tracker.token_ids)
        tracker.token_ids = list(full[:needed])
        emit(dict(event='resume_full_history', request_id=new_request.req_id,
                  prompt_tokens=len(prompt), full_tokens=len(full),
                  tracker_before=before, tracker_after=len(tracker.token_ids),
                  expected_cache_tokens=lmcache_cached_tokens))
        return tracker

    create._daos_resume_token_fix = True
    tracker_class.from_new_request = staticmethod(create)


def install():
    if os.environ.get('DAOS_VLLM_RESUME_TOKEN_FIX') != '1':
        return
    from lmcache.integration.vllm.vllm_v1_adapter import RequestTracker
    if getattr(RequestTracker.from_new_request, '_daos_resume_token_fix', False):
        return
    def emit(data):
        print('DAOS_RESUME_TOKEN_FIX ' + json.dumps(dict(pid=os.getpid(), **data)), flush=True)
    patch_tracker(RequestTracker, emit)
    emit(dict(event='installed'))
