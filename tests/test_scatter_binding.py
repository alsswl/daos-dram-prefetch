import ctypes as C
import pytest

from lmcache_daos.scatter_binding import ScatterObjectStore
from lmcache_daos.scatter_plan import Segment
from lmcache_daos.object_binding import DaosObjectError


class Library:
    def daosgdr_init(self,*a): return 1
    def daosgdr_fini(self,*a): pass
    def daosgdr_getv_device(self,ctx,key,ptrs,lens,offsets,n,device):
        self.call = (key,[(ptrs[i],lens[i],offsets[i]) for i in range(n.value)],device.value)
        return getattr(self,"rc",0)
    def daosgdr_putv_device(self,ctx,key,ptrs,lens,offsets,n,meta,length,device):
        self.metadata = bytes(meta.raw[:length.value])
        return self.daosgdr_getv_device(ctx,key,ptrs,lens,offsets,n,device)


def test_scatter_abi_and_metadata():
    lib=Library(); store=ScatterObjectStore("p","c",library=lib)
    segments=[Segment(2**34,32768,0),Segment(2**36,65536,32768)]
    store.putv("k",segments,b"meta",device_id=2)
    assert lib.call == (b"k",[(s.pointer,s.length,s.offset) for s in segments],2)
    assert lib.metadata == b"meta"
    # Sparse SOURCE extents are permitted only on fetch.
    sparse=[Segment(2**34,32768,32768),Segment(2**36,65536,131072)]
    store.getv("k",sparse)
    with pytest.raises(ValueError,match="complete chunk"):
        store.putv("k",sparse,b"meta")
    lib.rc=-1234
    with pytest.raises(DaosObjectError) as error:
        store.getv("k",segments)
    assert error.value.rc == -1234
    store.close()
    with pytest.raises(RuntimeError,match="closed"):
        store.getv("k",segments)


@pytest.mark.parametrize("segments", [[],[Segment(0,10,0)], [Segment(10,0,0)],
    [Segment(2**64-1,10,0)], [Segment(10,10,0),Segment(30,10,9)]])
def test_invalid_vectors_never_reach_library(segments):
    lib=Library();store=ScatterObjectStore("p","c",library=lib)
    with pytest.raises(ValueError):store.getv("k",segments)
    assert not hasattr(lib,"call")
