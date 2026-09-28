#!/usr/bin/env python3
"""Read-only aggregation/validation of a completed three-condition experiment.

Prints JSON to stdout. Cold and warm are separate; no reruns or cache mutation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics as stats

import yaml

from compare_e2e import percentile
from dram_cold_compare import CONDITIONS, schedule, validate_cold


def normalize_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop('local_cpu')
    ec = cfg['extra_config']
    for name in ('root', 'object_namespace', 'cpu_prefetch_gpu_gb'):
        ec.pop('daosgds.' + name, None)
    ec.pop('storage_plugin.daosgds.module_path')
    ec.pop('storage_plugin.daosgds.class_name')
    return cfg


def summarize(folder):
    assert json.loads((folder/'status.json').read_text())['status'] == 'completed'
    cases = json.loads((folder/'cases.json').read_text())
    plan = json.loads((folder/'plan.json').read_text())
    expected = list(schedule(plan['args']['repeats']))
    assert [(c['repeat'], c['concurrency'], c['condition']) for c in cases] == expected
    configs, maps, groups, outputs, warnings = [], [], {}, {}, []
    assert len({c['namespace'] for c in cases}) == len(cases)
    for case in cases:
        name = f"r{case['repeat']}_c{case['concurrency']}_{case['condition']}"
        path = folder/name
        cfg = yaml.safe_load((path/'main_profile/lmcache_effective.yaml').read_text())
        dram, gpu = CONDITIONS[case['condition']]
        assert cfg['local_cpu'] is (dram == 'on')
        ec = cfg['extra_config']
        expected_class = 'DaosDramPrefetchBackend' if gpu == 'on' else 'DaosGdsBackend'
        assert ec['storage_plugin.daosgds.class_name'] == expected_class
        expected_module = 'lmcache_daos.' + ('dram_prefetch_backend' if gpu == 'on' else 'gds_backend')
        assert ec['storage_plugin.daosgds.module_path'] == expected_module
        assert ec['daosgds.object_namespace'] == case['namespace'] + ':'
        if gpu == 'on':
            assert ec['daosgds.cpu_prefetch_gpu_gb'] == 5
        configs.append(normalize_config(cfg))
        maps.append(json.loads((path/'main_native_maps.json').read_text()))
        assert len(case['samples']) == 1 + plan['args']['reuse_passes']
        for sample in case['samples']:
            phase = 'cold' if sample['tag'] == 'cold' else 'warm'
            if phase == 'cold':
                validate_cold(sample)
            else:
                assert all(r['cached_tokens'] == 4095 for r in sample['rows'])
                assert sample['daos_prefetch_calls'] == (4 if dram == 'off' else 0)
                assert sample['cpu_gpu_prefetch_calls'] == (4 if gpu == 'on' else 0)
                assert sample['cpu_gpu_fallback_calls'] == 0
            assert len(sample['rows']) == 4
            for row in sample['rows']:
                assert row['prompt_tokens'] == 4096 and row['completion_tokens'] == 64
                outputs.setdefault((phase, case['concurrency'], row['request_id']), set()).add(
                    row['output_token_sha256'])
            log = (path/(sample['tag'] + '_server_slice.log')).read_text()
            entry = dict(repeat=case['repeat'], sample=sample,
                retrieve_ms=list(map(float, re.findall(r'Retrieved .*?cost ([0-9.]+) ms', log))),
                cpu_copy_ms=list(map(float, re.findall(r'CPU staging prefetch\[.*? in ([0-9.]+) ms', log))),
                daos_batch_ms=list(map(float, re.findall(r'DaosGdsBackend prefetch\[.*? in ([0-9.]+) ms', log))))
            groups.setdefault((case['condition'], case['concurrency'], phase), []).append(entry)
        full = (path/'main_server.log').read_text()
        for line in full.splitlines():
            if re.search(r'Traceback|ERROR|Logging error|negative: -|Double free|Double release|GPU buffer full', line):
                warnings.append(dict(case=name, line=line))
    assert all(c == configs[0] for c in configs), 'Unexpected configuration difference'
    assert maps[0] and all(m == maps[0] for m in maps), 'Native library paths differ'
    summaries = []
    for (condition, concurrency, phase), entries in sorted(groups.items()):
        rows = [r for e in entries for r in e['sample']['rows']]
        process_means = []
        for repeat in sorted({e['repeat'] for e in entries}):
            subset = [r for e in entries if e['repeat'] == repeat for r in e['sample']['rows']]
            process_means.append(stats.mean(r['ttft_ms'] for r in subset))
        out = dict(condition=condition, concurrency=concurrency, phase=phase,
            requests=len(rows), processes=len(process_means),
            ttft_mean_ms=stats.mean(r['ttft_ms'] for r in rows),
            ttft_p95_ms=percentile([r['ttft_ms'] for r in rows], .95),
            e2e_mean_ms=stats.mean(r['e2e_ms'] for r in rows),
            per_process_ttft_mean_ms=process_means,
            process_mean_ttft_sd_ms=stats.stdev(process_means) if len(process_means) > 1 else None)
        for key in ('retrieve_ms', 'cpu_copy_ms', 'daos_batch_ms'):
            values = [v for e in entries for v in e[key]]
            out[key + '_mean'] = stats.mean(values) if values else None
        for key in ('daos_prefetch_calls', 'cpu_gpu_prefetch_calls', 'cpu_gpu_fallback_calls'):
            out[key] = sum(e['sample'][key] for e in entries)
        summaries.append(out)
    hashes = json.loads((folder/'source_sha256.json').read_text())
    assert all(hashlib.sha256((folder/'executed_sources'/name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items())
    return dict(summary=summaries, validation=dict(configs_match_except_intended_fields=True,
        native_library_paths=maps[0], source_snapshot_hashes_valid=True,
        all_cold_requests_miss=True, all_warm_requests_full_hit=True,
        output_tokens_match_within_phase_concurrency_input=all(len(v) == 1 for v in outputs.values()),
        differing_output_groups=[str(k) for k, v in outputs.items() if len(v) != 1],
        log_findings=warnings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, help='New JSON file; never overwrite an existing result')
    args = parser.parse_args()
    result = summarize(args.folder)
    result['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open('x') as output:
            output.write(rendered + '\n')
    print(rendered)
