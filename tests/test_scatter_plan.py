import ctypes as C
import random

import pytest

from lmcache_daos.scatter_plan import plan_segments


@pytest.mark.parametrize("layout", ["NB_TWO", "TWO_NB"])
def test_random_scatter_matches_independent_token_reference(layout):
    rng = random.Random(23)
    for _ in range(100):
        layers, nb, bs, hidden, size = 3, 12, 4, 6, 2
        n = rng.randint(1, 25)
        skip = rng.randrange(n+1)
        slots = rng.sample(range(nb*bs), n)
        backing = [(C.c_ubyte * (nb*2*bs*hidden*size))() for _ in range(layers)]
        bases = [C.addressof(b) for b in backing]
        source = bytes(rng.randrange(256) for _ in range(2*layers*n*hidden*size))
        expected = [bytearray(bytes(b)) for b in backing]
        row = hidden*size
        for kv in range(2):
            for l in range(layers):
                for t in range(skip, n):
                    b,u = divmod(slots[t],bs)
                    dest = (((b*2+kv)*bs+u) if layout == "NB_TWO" else
                            ((kv*nb+b)*bs+u))*row
                    start = ((kv*layers+l)*n+t)*row
                    expected[l][dest:dest+row] = source[start:start+row]
        segments = plan_segments(bases,nb,bs,hidden,size,slots,skip=skip,layout=layout)
        for s in segments:
            C.memmove(s.pointer, source[s.offset:s.offset+s.length], s.length)
        assert [bytes(b) for b in backing] == [bytes(b) for b in expected]


def test_page_coalescing_and_qwen_geometry():
    slots = [b*16+t for b in [91,17,203,5,33,80,12,100] for t in range(16)]
    segments = plan_segments([2**32+l*2**24 for l in range(36)],256,16,1024,2,slots)
    assert len(segments) == 576
    assert sum(s.length for s in segments) == 18*2**20
    assert all(s.length == 32768 for s in segments)


@pytest.mark.parametrize("slots,skip", [([-1,2],0),([999],0),([1,1],0),([1,1],1)])
def test_invalid_slots(slots, skip):
    with pytest.raises(ValueError):
        plan_segments([4096],8,4,8,2,slots,skip=skip)


def test_skipped_prefix_may_have_invalid_slots():
    assert plan_segments([4096],8,4,8,2,[-1,2,3],skip=1)


def test_iov_limit_is_explicit():
    with pytest.raises(ValueError,match="SGL too large"):
        plan_segments([4096],8,4,8,2,[0,4,8],max_iovs=2)
