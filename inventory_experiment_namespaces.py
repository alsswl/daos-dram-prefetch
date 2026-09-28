#!/usr/bin/env python3
"""Read local experiment configs; write a review inventory. NO DAOS calls/deletes."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import yaml

ROOT=Path(__file__).resolve().parent
PRIORITY={
    'discovery_capacity_matrix_20260927',
    'discovery_capacity_matrix_off_20260927',
    'prefetch_d8_s8_c8_repeat3_20260927',
    'discovery_replay_s8_d4_c16_20260927',
    'discovery_fixed_replay_20260926',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=subprocess.check_output(['rg','-l','daosgds.object_namespace','-g','*.yaml','-g','*.yml',
        '-g','!venv/**','-g','!**/executed_sources/**','-g','!**/agents/**',
        '-g','!discoverybench/**','-g','!lmcache-daos-repo/**','.'],cwd=ROOT,text=True).splitlines()
    entries={}; dfs=[]
    for name in sorted(paths):
        path=ROOT/name; rel=path.relative_to(ROOT); cfg=yaml.safe_load(path.read_text())
        extra=cfg.get('extra_config',{}); ns=extra.get('daosgds.object_namespace')
        pool=extra.get('daosgds.pool'); container=extra.get('daosgds.container'); mode=extra.get('daosgds.transport')
        top_level=len(rel.parts)==1
        if mode!='object' and not top_level:
            dfs.append(dict(config=str(rel),namespace_field=ns,root=extra.get('daosgds.root'),
                            reason='DFS config; namespace field does not prove object data exists'))
            continue
        identity=(pool,container,ns)
        row=entries.setdefault(identity,dict(pool=pool,container=container,namespace=ns,sources=[],
            experiments=[],run_statuses=[],execution_logs_found=False,has_default_reference=False,
            live_keys_checked=False,reclaimable_bytes=None))
        row['sources'].append(str(rel))
        row['has_default_reference'] |= top_level
        if not top_level:
            experiment=rel.parts[0]
            if experiment not in row['experiments']: row['experiments'].append(experiment)
            status_path=next((p for p in (path.parent/'status.json',ROOT/experiment/'status.json') if p.exists()),None)
            if status_path:
                value=json.loads(status_path.read_text()).get('status','unknown')
                row['run_statuses'].append(dict(source=str(status_path.relative_to(ROOT)),status=value))
            row['execution_logs_found'] |= bool(list(path.parent.glob('*.log')) or
                list((ROOT/experiment).glob('*.log')) or list((ROOT/experiment).glob('*/*.log')))
    rows=[]
    for row in entries.values():
        if row['has_default_reference']:
            category='KEEP_DEFAULT'
        elif any('dryrun' in n for n in row['experiments']):
            category='EXCLUDE_DRYRUN'
        elif not row['execution_logs_found']:
            category='REVIEW_NO_LOG'
        elif set(row['experiments'])<=PRIORITY and row['pool']=='discospool' and row['container']=='kvcache':
            category='PRIORITY_REVIEW'
        else:
            category='OLDER_REVIEW'
        row['category']=category
        rows.append(row)
    rows.sort(key=lambda r:(r['category'],r['experiments'],r['namespace']))
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    payload=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope=str(ROOT),
        deletion_authorized=False,daos_calls_made=False,
        warning='Local config/log inventory only. Not an actual live key/byte inventory. Review before deleting.',
        category_counts=dict(Counter(r['category'] for r in rows)),namespaces=rows,dfs_configs_excluded=dfs)
    (out/'inventory.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    with (out/'inventory.csv').open('w') as stream:
        writer=csv.writer(stream); writer.writerow(['category','pool','container','namespace','experiments','configs','execution_logs_found'])
        for r in rows:
            writer.writerow([r['category'],r['pool'],r['container'],r['namespace'],
                ';'.join(r['experiments']),';'.join(r['sources']),r['execution_logs_found']])
    primary=[r for r in rows if r['category']=='PRIORITY_REVIEW']
    (out/'priority_review.json').write_text(json.dumps(dict(deletion_authorized=False,
        live_keys_checked=False,entries=primary),indent=2,ensure_ascii=False)+'\n')
    text=['# DAOS 실험 namespace 정리 후보 — 삭제하지 않음','',
        '설정과 실행 로그의 존재를 기준으로 만든 검토 목록이다. DAOS 키 열거·삭제는 하지 않았다. '
        '현재 서버에 남아 있는 키 수, 실제 저장량, 회수 가능한 물리 용량, 다른 프로세스의 사용 여부는 미확인이다.',
        '', '## 우선 검토: 최근 재생 실험', '',
        f'`discospool/kvcache`의 최근 실험 전용 namespace **{len(primary)}개**. 결과 로그·그래프를 보존한 채 '
        'KV 데이터만 정리하는 후보다. 삭제하면 해당 namespace의 warm/restart 재사용은 불가능해지고 다시 fill해야 한다. '
        '새 namespace로 시작하는 다음 rolling 실험에는 과거 캐시가 필요하지 않다.', '',
        '| 실험 | namespace 수 |','|---|---:|']
    counts=Counter(e for r in primary for e in r['experiments'])
    text += [f'| {e} | {n} |' for e,n in sorted(counts.items())]
    text += ['', '### 정확한 namespace와 설정 근거','',
             '| 번호 | 실행 설정 | namespace 전체 문자열 |','|---:|---|---|']
    for i,r in enumerate(primary,1):
        text.append(f'| {i} | [{r["sources"][0]}]({ROOT/r["sources"][0]}) | `{r["namespace"]}` |')
    text += ['', '## 기본 설정 — 우선 정리에서 제외','']
    text += [f'- `{r["namespace"]}`: '+', '.join(r['sources']) for r in rows if r['category']=='KEEP_DEFAULT']
    text += ['', '## 그 밖의 과거 실험 — 별도 검토','',
             '실행 실패·부분 저장·중복 설정이 있을 수 있다. 로그가 있다는 사실만으로 현재 데이터 잔존이나 삭제 안전성을 보장하지 않는다.', '',
             '| 구분 | 실험 | namespace |','|---|---|---|']
    for r in rows:
        if r['category'] not in ('PRIORITY_REVIEW','KEEP_DEFAULT'):
            text.append(f'| {r["category"]} | {", ".join(r["experiments"])} | `{r["namespace"]}` |')
    text += ['', '## 안전 범위','',
        '- 이 목록은 삭제 승인이나 자동 삭제 스크립트가 아니다. broad prefix `minji-` 전체 삭제 금지.',
        '- 삭제 승인 후에도 pool/container/OID와 위의 정확한 namespace를 대조해 dkey 목록을 먼저 확인해야 한다.',
        '- 풀·컨테이너 전체 및 서버 NVMe 파일을 직접 삭제하지 않는다.',
        '- DFS 구성에 적힌 object_namespace는 실제 object 저장을 의미하지 않는다. DFS 설정은 별도 제외 목록으로 inventory.json에 보존했다.',
        '- dry-run과 실행 로그가 없는 설정은 우선 후보에서 제외했다. 실행 소스 스냅샷 안의 기본 설정도 중복 집계하지 않았다.',
        '- 로컬 로그·CSV·그래프·소스는 지우지 않는다. KV 제거는 백업이 없다면 되돌릴 수 없고 다시 추론/저장해야 한다.',
        '- 예상 확보 용량은 미산정이다. 논리 키 삭제 후 물리 공간 반영까지도 별도로 확인해야 한다.', '',
        '[전체 JSON](inventory.json) · [CSV](inventory.csv) · [우선 후보 JSON](priority_review.json)', '']
    (out/'NAMESPACE_REVIEW_KO.md').write_text('\n'.join(text))
    print(json.dumps(dict(output=str(out),counts=payload['category_counts'],priority_groups=counts,
                          dfs_config_count=len(dfs)),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
