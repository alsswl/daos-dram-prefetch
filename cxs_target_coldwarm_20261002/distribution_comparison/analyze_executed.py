"""Read-only counterfactual baseline vs recorded balanced CXS target placement.

No DAOS connection, GPU allocation, or benchmark rerun. The baseline is computed
for the SAME logical keys, namespace, and saved object layout as the measured arm.
"""
import argparse
from collections import Counter
import ctypes as C
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import statistics


def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def describe(values):
    mean=statistics.mean(values)
    return dict(total=sum(values),mean=mean,min=min(values),max=max(values),
                max_min_ratio=max(values)/min(values),
                population_stddev=statistics.pstdev(values),
                coefficient_of_variation_pct=100*statistics.pstdev(values)/mean,
                max_above_mean_pct=100*(max(values)/mean-1))


def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    plan=read(source/'plan.json');manifest=read(source/'placement.json')
    validated=read(source/'placement_validation.json')
    assert plan['target_count']==16 and validated['passed']
    assert read(source/'status.json')['status']=='completed'
    assert read(source/'owned_object_cleanup.json')['rc']==0
    db=sqlite3.connect(f'file:{source}/placement.positions.sqlite?mode=ro',uri=True)
    pairs=db.execute('SELECT key,idx FROM positions WHERE key LIKE ? ORDER BY key',
                     (plan['experiment_namespace']+'%',)).fetchall();db.close()
    assert len(pairs)==validated['validated_keys']==read(source/'serving/warm/final_sample.json')['daos_puts']
    lib=C.CDLL(manifest['library'])
    lib.placement_predict.argtypes=[C.c_char_p,C.c_uint];lib.placement_predict.restype=C.c_uint
    layout={int(k):tuple(v) for k,v in manifest['layout'].items()}
    n=len(layout);assert n==16 and len(set(layout.values()))==16
    # Validate predictor against the descriptor's keys whose actual shards were
    # verified through DAOS during the original experiment.
    for shard,key in enumerate(manifest['addressing']['placement_keys']):
        assert lib.placement_predict(key.encode(),n)==shard
    records=[]
    for full,index in pairs:
        baseline=lib.placement_predict(full.encode(),n)
        balanced=manifest['target_spread_shards'][index%n]
        # Reconstruct the exact salted dkey from the archived implementation.
        for salt in range(100000):
            dk=(full+f'|target-spread/{salt:06d}').encode()
            if lib.placement_predict(dk,n)==balanced:break
        else:raise RuntimeError('Salt search exhausted')
        records.append(dict(full_key=full,absolute_chunk=index,baseline_shard=baseline,
                            balanced_shard=balanced,baseline_target=layout[baseline],
                            balanced_target=layout[balanced],balanced_salt=salt))
    actual=Counter(r['balanced_shard'] for r in records)
    assert actual==Counter({int(k):v for k,v in validated['counts_by_shard'].items()})
    targets=sorted(layout.values())
    sections={}
    for name,rows in [('all_stored',records),('initial_480_chunks_per_document',[r for r in records if r['absolute_chunk']<480]),
                      ('conversation_suffix',[r for r in records if r['absolute_chunk']>=480])]:
        counts={mode:Counter(tuple(r[mode+'_target']) for r in rows) for mode in ['baseline','balanced']}
        table=[dict(rank=rank,target=target,baseline_chunks=counts['baseline'][rank,target],
                    balanced_chunks=counts['balanced'][rank,target]) for rank,target in targets]
        sections[name]=dict(chunks=len(rows),targets=table)
        if name!='conversation_suffix':
            sections[name]['statistics']={mode:describe([c[mode+'_chunks'] for c in table]) for mode in ['baseline','balanced']}
    assert sections['initial_480_chunks_per_document']['chunks']==3840
    assert all(row['balanced_chunks']==240 for row in sections['initial_480_chunks_per_document']['targets'])
    result=dict(passed=True,source=str(source),model=plan['model'],sessions=8,turns=6,phases=['cold','warm'],
                chunk_bytes=18*2**20,namespace=plan['experiment_namespace'],oid=manifest['oid'],
                sections=sections,
                interpretation=['Balanced counts reproduce the prior actual DAOS shard validation for all 4056 stored keys.',
                                'Baseline is a deterministic counterfactual: hash the same full keys without the routing suffix.',
                                'Same namespace, key set and object layout; baseline was not a separate serving run.',
                                'These are unique stored chunk counts after cold+warm, not fetch counts or runtime concurrency.',
                                'All 3840 initial body chunks distribute exactly 240 per target with index-modulo-16 placement.',
                                'Conversation suffix branches occupy indices 480..491, so total balanced counts need not be equal.',
                                'Good aggregate hash balance does not imply balanced concurrent/window-level demand.'],
                source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                               for name in ['plan.json','placement.json','placement_validation.json']},
                native_library_sha256=hashlib.sha256(Path(manifest['library']).read_bytes()).hexdigest())
    save(out/'results.json',result);save(out/'key_mapping.json',records)
    for name in ['plan.json','placement.json','placement_validation.json']:shutil.copy2(source/name,out/('source_'+name))
    shutil.copy2(__file__,out/'analyze_executed.py')
    with (out/'distribution.csv').open('w') as f:
        f.write('scope,engine_rank,target_index,baseline_chunks,balanced_chunks\n')
        for scope,section in sections.items():
            for r in section['targets']:f.write(f"{scope},{r['rank']},{r['target']},{r['baseline_chunks']},{r['balanced_chunks']}\n")
    lines=['Saved light CXS: target distribution comparison',
           'Baseline = calculated original dkey placement; balanced = reconstructed and matched against recorded DAOS verification.',
           'Same 4056 unique KV keys, namespace and object layout from the 16-target cold/warm run.',
           'Counts describe stored chunks, not concurrent I/O. No serving rerun or DAOS mutation.', '',
           'engine:target | original hash | balanced (actual prior validation)']
    for r in sections['all_stored']['targets']:lines.append(f"{r['rank']}:{r['target']} | {r['baseline_chunks']} | {r['balanced_chunks']}")
    lines+=['','Statistics:']
    for mode,s in sections['all_stored']['statistics'].items():lines.append(mode+': '+json.dumps(s))
    lines+=['']+result['interpretation']
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,1,figsize=(12,7.4),layout='constrained')
    x=np.arange(16);width=.38
    for ax,scope,title in zip(axes,['all_stored','initial_480_chunks_per_document'],
                             ['All stored chunks after cold + warm (4056 unique keys)',
                              'Initial 480 chunks per document (8 documents, 3840 unique keys)']):
        table=sections[scope]['targets']
        for offset,mode,color,label in [(-width/2,'baseline','#777777','Original hash (calculated)'),
                                        (width/2,'balanced','#0072b2','Balanced (verified recorded placement)')]:
            bars=ax.bar(x+offset,[r[mode+'_chunks'] for r in table],width,color=color,label=label)
            ax.bar_label(bars,fontsize=7,padding=2)
        ax.axhline(sections[scope]['chunks']/16,color='black',ls='--',lw=.8,label='Equal-share mean')
        ax.set_xticks(x,[f'{rank}:{target}' for rank,target in targets]);ax.set_ylim(0,340)
        ax.set_ylabel('Unique stored KV chunks');ax.set_xlabel('Engine rank : target index')
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8,ncol=3)
    fig.suptitle('Light CXS: same KV keys and object layout, original vs index-balanced placement',fontsize=12)
    for suffix in ['png','svg']:fig.savefig(out/f'distribution.{suffix}',dpi=180)
    print(json.dumps(sections['all_stored'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.source.resolve(),a.out.resolve())
