"""Strict partial-load allowlist for explicit recompute experiments only.

No model, transport, scheduler or production logging behavior is changed.
Original log bytes are retained. Unknown LMCache errors remain fatal.
"""
import json
import re

ANSI = re.compile(r'\x1b\[[0-9;]*m')
ALLOWED = re.compile(
    r'^LMCache ERROR: (?:Request \S+The number of retrieved tokens is less than the '
    r'expected number of tokens! This should not happen!|'
    r'Num retrieved tokens: \d+, num expected tokens: \d+) '
    r'\(vllm_v1_adapter\.py:\d+:lmcache\.integration\.vllm\.vllm_v1_adapter\)$')
FATAL = ('DER_NOSPACE','rc=-1007','[libdaosgdr] daos_obj_update_gpu rc=',
         'Double free','Double release','negative: -','CUDA error:','CUDA out of memory',
         'Traceback (most recent call last)', 'failure_policy=fail')


class RecomputeHealth:
    def __init__(self, path):
        self.path, self.offset, self.pending = path, 0, ''
        self.allowed_lines = 0

    def __call__(self):
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            text = self.pending + stream.read().decode(errors='replace')
            self.offset = stream.tell()
        clean = ANSI.sub('', text)
        for pattern in FATAL:
            if pattern in clean:
                raise RuntimeError(f'Fatal server/storage error: {pattern}')
        lines = clean.split('\n')
        self.pending = lines.pop()
        for line in lines:
            if 'LMCache ERROR:' not in line:
                continue
            message = line[line.index('LMCache ERROR:'):].strip()
            if not ALLOWED.fullmatch(message):
                raise RuntimeError(f'Unexpected LMCache error: {message}')
            self.allowed_lines += 1


def recovery_report(case):
    """Scheduler-affected tokens are recovery accounting, not a GPU compute timer."""
    from staging_mixed_pressure import read_events
    log = ANSI.sub('', (case/'server.log').read_text())
    assert "kv_load_failure_policy='recompute'" in log
    recovery = [(int(n), int(t)) for n,t in re.findall(
        r'Recovered from KV load failure: (\d+) request\(s\) rescheduled \((\d+) tokens affected\)',log)]
    events = read_events(case)
    outcomes = [e for e in events if e['event']=='daos_demand_outcome']
    assert outcomes and all(e['other_failed_chunks']==0 for e in outcomes)
    calls = json.loads((case/'calls.json').read_text())
    assert len(calls)==480 and all(c['status']=='success' for c in calls)
    missing = re.findall(r'Request (\S+) failed to load (\d+) tokens across (\d+) blocks', log)
    assert not missing or recovery, 'Partial loads occurred but scheduler recovery was not observed'
    affected = sum(t for _,t in recovery)
    return dict(policy='recompute',completed_calls=len(calls),
        recovery_events=len(recovery),rescheduled_request_events=sum(n for n,_ in recovery),
        affected_tokens_for_recompute=affected,
        affected_tokens_over_input_pct=100*affected/sum(c['prompt_tokens'] for c in calls),
        missing_load_reports=[dict(request_id=r,tokens=int(t),blocks=int(b)) for r,t,b in missing],
        capacity_failed_chunks=sum(e['capacity_failed_chunks'] for e in outcomes),
        requested_daos_chunks=sum(e['requested_chunks'] for e in outcomes),
        returned_daos_chunks=sum(e['returned_chunks'] for e in outcomes),
        successful_tail_discarded_chunks=sum(e['successful_tail_discarded_chunks'] for e in outcomes),
        notes=['Affected tokens are scheduler recovery accounting, not measured compute time.',
               'Request events can repeat for one request; not a unique request count.',
               'Lookup hits are availability, not actual successful KV reuse after load failure.'])
