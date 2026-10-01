#!/usr/bin/env python3
"""Queued metadata-first experiment; never changes the running baseline runner."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
import uuid

import yaml
import sharegpt_cold_warm as cw

base, ROOT = cw.base, cw.ROOT
read, dump = base.read, base.dump
EXTRA_SOURCES = ('sharegpt_early_lookup.py', 'report_early_lookup.py',
                 'tests/early_lookup_roundtrip.py', 'tests/test_early_lookup_experiment.py',
                 'tests/test_early_lookup.py')


def prepare(root, baseline, after_unit):
    if root.parent != ROOT or root.exists():
        raise ValueError('Use a new immediate child directory of discos_minji')
    old = read(baseline/'plan.json')
    assert old['cpu_gib'] == 256 and old['warm_repeats'] == 4
    assert not old.get('all_prefetch_off') and not old.get('segmented_pinned')
    assert base.digest(baseline/'requests.json') == old['request_sha256']
    root.mkdir()
    for name in ('requests.json', 'sessions.json', 'capacity_estimate.json', 'params.json',
                 'runtime_versions.json'):
        shutil.copy2(baseline/name, root/name)
    plan = dict(old)
    plan.update(early_lookup=True, baseline=str(baseline), after_unit=after_unit,
                cases=[dict(name='c16_early', prefetch=True, pilot=False, concurrency=16)],
                notes=['Metadata-first notification; DRAM and DAOS speculative prefetch ON.',
                       'Queued work is cancelled on retrieve; in-flight work is awaited.',
                       'Same ShareGPT histories, output caps, rolling C16, DRAM256/staging8.',
                       'Fresh process/DRAM/staging/UUID namespace; cold once then warm four times.',
                       'Warm repeats retain evolving caches; not independent trials.',
                       'No lookup backoff changes, occupancy gate, artificial delay or cache placement.',
                       'Separate GPU byte test and small vLLM smoke test must pass before full run.',
                       'Only exact successful experiment UUID keys are cleaned; failures retain evidence.'])
    files = set(old['source_sha256']) | set(EXTRA_SOURCES)
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'lmcache_daos').glob('*.py'))
    plan['source_sha256'] = {}
    for name in sorted(files):
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        plan['source_sha256'][name] = base.digest(dest)
    dump(root/'plan.json', plan)
    dump(root/'status.json', dict(status='queued', after_unit=after_unit))
    (root/'EXPERIMENT_KO.md').write_text(
        '# 존재 확인 먼저 알림: 후속 비교 실험\n\n'
        '기존 비교 실험의 정상 종료·캐시 정리 후 실행한다. 선행 실패 시 자동으로 진행하지 않는다.\n\n'
        'Qwen3-14B BF16 / DRAM256GiB / staging8GiB / C16 / 청크128. '
        '동일 ShareGPT 421개 대화의 앞4턴, 단계당1,684개 요청. cold1회 + warm4회. '
        'warm 사이 캐시를 유지하고, 새로운 조건은 새 프로세스와 빈 DRAM·staging·UUID namespace로 시작한다.\n\n'
        'DRAM/DAOS 프리페치 모두 ON. 존재 확인 직후 먼저 알린 뒤 전송을 시작한다. '
        'retrieve 시 대기 작업은 취소하고 직접 읽으며, 이미 시작한 전송은 기다린다. '
        '작업자1·기본 lookup backoff·출력상한·도착 정책은 기존과 동일하다. '
        '프리페치 완전 OFF 실험이 아니다.\n\n'
        '실제 GPU 바이트 검증 → 별도 작은 vLLM cold/warm 검증 → 본 실험 순서. '
        '검증용 캐시는 본 실험과 분리한다. 실패하면 로그를 보존하고 중단한다. '
        '결과에는 TTFT, 재사용량, 실제 DRAM/DAOS hit, staging 그래프, '
        '완료 데이터 사용/진행 중 대기/대기 작업 직접 읽기 전환을 기록한다. '
        '준비된 CPU 캐시를 인위적으로 추가하거나 계산을 지연하지 않는다.\n')


def config():
    cfg = base.make_config(True, 'minji-cold-warm-'+uuid.uuid4().hex,
                           cpu_gib=256, staging_gib=8)
    cfg['extra_config'].update({
        'storage_plugin.daosgds.module_path': 'lmcache_daos.early_lookup_backend',
        'storage_plugin.daosgds.class_name': 'EarlyLookupBackend',
        'daosgds.early_lookup': True,
        'daosgds.dram_prefetch_workers': 1,
        'daosgds.dram_prefetch_cancel_queued': False,
        'daosgds.dram_prefetch_early_ready': False,
        'daosgds.dram_prefetch_stop_occupancy_ratio': None})
    return cfg


def unit_state(unit):
    output = subprocess.check_output(['systemctl', '--user', 'show', unit,
        '-p', 'ActiveState', '-p', 'SubState', '-p', 'Result', '-p', 'LoadState'], text=True)
    return dict(line.split('=', 1) for line in output.splitlines() if '=' in line)


def predecessor_done(state, status):
    if state['ActiveState'] in ('active', 'activating', 'deactivating', 'reloading'):
        return False
    if status.get('status') != 'completed' or state.get('Result') not in (None, 'success'):
        raise RuntimeError('Predecessor did not complete successfully; refusing to start')
    return True


def wait_predecessor(root, plan):
    deadline = time.monotonic()+24*3600
    while True:
        previous = read(Path(plan['baseline'])/'status.json')
        state = unit_state(plan['after_unit'])
        if predecessor_done(state, previous):
            dump(root/'predecessor_completion.json', dict(unit=state, status=previous,
                                                         time_ns=time.time_ns()))
            return
        if time.monotonic() > deadline:
            raise TimeoutError('Previous experiment did not finish within 24 hours')
        dump(root/'status.json', dict(status='queued', after_unit=plan['after_unit'],
                                     predecessor=previous, updated_ns=time.time_ns()))
        time.sleep(20)


def idle_gpu():
    return not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip()


def wait_idle(root):
    deadline = time.monotonic()+6*3600
    while not idle_gpu():
        dump(root/'status.json', dict(status='waiting_gpu', updated_ns=time.time_ns()))
        if time.monotonic() > deadline:
            raise TimeoutError('GPU occupied; no unrelated job was stopped')
        time.sleep(20)


def verify_sources(root, plan):
    for name, digest in plan['source_sha256'].items():
        assert base.digest(ROOT/name) == digest, f'Source changed while queued: {name}'
    assert base.digest(root/'requests.json') == plan['request_sha256']


def run_case(root, spec):
    case = root/spec['name']
    case.mkdir(exist_ok=False)
    plan, records = read(root/'plan.json'), read(root/'requests.json')
    cap = plan['capacity']
    cw.wait_space(case, 2*cap['conservative_stored_kv_gib']-cap['unique_input_kv_gib'])
    assert idle_gpu(), 'GPU occupied; no unrelated job will be stopped'
    (case/'config.yaml').write_text(yaml.safe_dump(config(), sort_keys=False))
    args = SimpleNamespace(model=plan['model'], max_model_len=16384, max_num_seqs=16, port=8017)
    try:
        with base.server(args, case/'config.yaml', case) as client:
            initial = base.await_empty(case)
            assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
            dump(case/'initial_sample.json', initial)
            health = base.LogHealth(case/'server.log')
            health()
            final = cw.run_phases(case, records, spec['concurrency'], client, health, initial,
                phases=plan['phases'],
                warm_extra_gib=cap['conservative_stored_kv_gib']-cap['unique_input_kv_gib'])
            dump(case/'final_sample.json', final)
        dump(case/'status.json', dict(status='completed', requests=len(records)*len(plan['phases'])))
    except BaseException as exc:
        dump(case/'status.json', dict(status='failed', error=repr(exc)))
        raise


def cleanup(case):
    with (case/'cleanup.log').open('a') as stream:
        subprocess.run([str(ROOT/'run_vllm.sh'), str(ROOT/'venv/bin/python3'),
            str(ROOT/'discovery_occupancy_gate.py'), '--cleanup-case', str(case)], cwd=ROOT,
            env=dict(os.environ, DAOSGDS_TRANSPORT='object'), stdout=stream,
            stderr=subprocess.STDOUT, timeout=900, check=True)


def gpu_preflight(root):
    dump(root/'status.json', dict(status='gpu_validation', updated_ns=time.time_ns()))
    with (root/'gpu_validation.log').open('x') as log:
        subprocess.run([str(ROOT/'run_vllm.sh'), str(ROOT/'venv/bin/python3'),
            str(ROOT/'tests/early_lookup_roundtrip.py'), '--output', str(root/'gpu_validation')],
            cwd=ROOT, env=dict(os.environ, DAOSGDS_TRANSPORT='object'), stdout=log,
            stderr=subprocess.STDOUT, timeout=180, check=True)
    assert read(root/'gpu_validation/result.json')['result'] == 'PASS'


def smoke(root, plan):
    from report_early_lookup import summarize_case
    # Existing audited cleanup accepts only ROOT / experiment / case.
    folder = root.with_name(root.name+'_validation')
    folder.mkdir()
    dump(root/'validation_location.json', dict(path=str(folder)))
    records = read(root/'requests.json')[:8]  # Two real conversations, four turns each.
    pilot = dict(plan, phases=['cold', 'warm'], requests_per_phase=len(records),
                 requests_per_case=2*len(records), cases=[dict(name='pilot', concurrency=16)])
    dump(folder/'plan.json', pilot)
    dump(folder/'requests.json', records)
    dump(root/'status.json', dict(status='vllm_validation', updated_ns=time.time_ns()))
    spec = pilot['cases'][0]
    run_case(folder, spec)
    summarize_case(folder, spec)
    summaries = read(folder/'pilot/summary.json')
    assert summaries[1]['cached_tokens'] > 0, 'Smoke test did not exercise cache retrieval'
    cleanup(folder/'pilot')
    dump(folder/'status.json', dict(status='completed', result='PASS'))


def run(root):
    from report_early_lookup import summarize_case, report
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(root/'plan.json')
        try:
            wait_predecessor(root, plan)
            wait_idle(root)
            verify_sources(root, plan)
            mem = {s.split(':')[0]: int(s.split()[1])*1024
                   for s in Path('/proc/meminfo').read_text().splitlines()}
            assert mem['MemAvailable'] >= 320*2**30, 'Insufficient host memory'
            os.environ['DAOS_GDS_PREFETCH_TIMING'] = '1'
            gpu_preflight(root)
            smoke(root, plan)
            for spec in plan['cases']:
                dump(root/'status.json', dict(status='running', current=spec['name'],
                                             updated_ns=time.time_ns()))
                run_case(root, spec)
                summarize_case(root, spec)
                report(root)
                cleanup(root/spec['name'])
            dump(root/'status.json', dict(status='completed', updated_ns=time.time_ns()))
        except BaseException as exc:
            dump(root/'status.json', dict(status='failed', error=repr(exc),
                                         updated_ns=time.time_ns()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-from', type=Path)
    parser.add_argument('--after-unit')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.prepare_from:
        assert args.after_unit
        prepare(root, args.prepare_from.resolve(), args.after_unit)
    elif args.report:
        from report_early_lookup import report
        report(root)
    else:
        assert root.parent == ROOT
        run(root)


if __name__ == '__main__':
    main()
