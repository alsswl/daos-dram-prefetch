"""Create a frozen isolated-OID descriptor; no production data is accessed."""
import argparse
import ctypes as C
import json
from pathlib import Path
import re
import subprocess
import uuid

from lmcache_daos.placement_backend import load_library,checked
from experiments.chunk_placement.addressing import Addressing


def prepare(path,mode):
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    library=Path(__file__).resolve().parent/'lmcache_transport.so'
    lib=load_library(library)
    lib.placement_predict.argtypes=[C.c_char_p,C.c_uint];lib.placement_predict.restype=C.c_uint
    ctx,hi,lo=C.c_void_p(),C.c_uint64(),C.c_uint64()
    nonce=(uuid.uuid4().int & ((1<<63)-1)) | (1<<62)
    checked(lib.placement_open(b'discospool',b'kvcache',nonce,C.byref(ctx),C.byref(hi),C.byref(lo)))
    try:
        oid=f'{hi.value}.{lo.value}'
        text=subprocess.check_output(['daos','object','query','discospool','kvcache',oid],text=True)
        layout={int(g):[int(r),int(t)] for g,r,t in re.findall(r'grp: (\d+)\s+replica 0 (\d+):(\d+)',text)}
        n=len(layout)
        assert n==16 and sorted(layout)==list(range(n)) and len({tuple(v) for v in layout.values()})==n
        keys=[None]*n
        for salt in range(100000):
            key=f'placement-v1/{salt}'
            shard=lib.placement_predict(key.encode(),n)
            if keys[shard] is None:keys[shard]=key
            if all(keys):break
        assert all(keys)
        for i,key in enumerate(keys):
            actual=C.c_uint();checked(lib.placement_shard(ctx,key.encode(),C.byref(actual)))
            assert actual.value==i
        data=dict(version=1,pool='discospool',container='kvcache',nonce=nonce,oid=oid,
                  library=str(library),layout=layout,addressing=Addressing(mode,tuple(keys)).to_dict(),
                  isolated_experiment_object=True)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,indent=2)+'\n')
        return data
    finally:checked(lib.placement_close(ctx,0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['baseline','balanced'],required=True);a=p.parse_args()
    print(json.dumps(prepare(a.output,a.mode)))
