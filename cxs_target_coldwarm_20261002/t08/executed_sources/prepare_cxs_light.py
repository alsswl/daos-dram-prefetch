"""Prepare a smaller real CXS workload; does not launch a server or touch DAOS."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'realqa_mixed_nodram2g_20261002_w500'


def light_schedule(seed=20261001):
    lanes=[]
    for lane in range(8):
        rng=random.Random(seed+lane);turns=[0,0];rows=[]
        for position in range(12):
            if position==0: slot=0
            elif position==3: slot=1
            else:
                eligible=[s for s in range(2) if turns[s]<6 and (s==0 or position>3)]
                slot=rng.choice(eligible)
            rows.append(dict(lane=lane,position=position,session_index=lane*2+slot,
                             turn=turns[slot],new_session=turns[slot]==0))
            turns[slot]+=1
        assert turns==[6,6]
        lanes.append(rows)
    return dict(seed=seed,start_positions=[0,3],lanes=lanes,requests=96,
                concurrency=8,sessions=16,turns_per_session=6)


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    original=json.loads((SOURCE/'sessions.json').read_text())
    selected=[]
    for lane in range(8):
        for slot in range(2):
            old=next(s for s in original if s['group']==lane and s['slot']==slot)
            selected.append(dict(old,source_session_index=old['index'],index=lane*2+slot,group=lane,slot=slot))
    filenames={s['file'] for s in selected}
    docs=[d for d in json.loads((SOURCE/'documents.json').read_text()) if d['file'] in filenames]
    (output/'documents').mkdir()
    for d in docs:
        shutil.copy2(SOURCE/'documents'/d['file'],output/'documents'/d['file'])
        assert hashlib.sha256((output/'documents'/d['file']).read_bytes()).hexdigest()==d['sha256']
    kv_bytes=36*2*8*128*2
    initial_gib=sum(d['initial_prompt_tokens'] for d in docs)*kv_bytes/2**30
    profile=dict(model='Qwen/Qwen3-4B-Instruct-2507',source_dataset=str(SOURCE),
                 concurrency=8,session_depth=2,sessions=16,num_rounds=5,turns_per_session=6,
                 expected_requests=96,answer_len=256,document_tokens=61440,max_model_len=65536,
                 max_num_seqs=8,chunk_tokens=128,cpu_gib=0,dram_disabled=True,
                 staging_gib=2,store_window_mib=500,retrieve_window_mib=500,
                 gpu_memory_utilization=.835,split_store_gib=0,capacity_pipeline=True,mixed_schedule=True,
                 capacity=dict(unique_selected_books=len(docs),initial_unique_kv_upper_gib=initial_gib,
                               stored_kv_upper_gib=initial_gib+16*6*(256+128)*kv_bytes/2**30),
                 notes=['Workload inputs only; comparison conditions, fresh object manifests and source snapshots must be added before execution.',
                        'First two original document assignments from each of eight lanes. Document contents unchanged.',
                        'Each lane starts sessions at positions 0 and 3; other requests are seeded follow-ups.',
                        'Keep all six turns and real generated-answer feedback; no shortening of documents or output budget.',
                        'Reduced sessions change reuse/interleaving relative to the 480-request run; compare conditions within this new workload.'])
    for name,data in [('sessions.json',selected),('documents.json',docs),('mixed_schedule.json',light_schedule())]:
        (output/name).write_text(json.dumps(data,indent=2)+'\n')
    profile['input_sha256']={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                             for name in ['sessions.json','documents.json','mixed_schedule.json']}
    (output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    (output/'status.json').write_text(json.dumps(dict(status='workload_prepared',server_started=False))+'\n')
    shutil.copy2(__file__,output/'prepare_cxs_light.py')
    return profile


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().output),indent=2))
