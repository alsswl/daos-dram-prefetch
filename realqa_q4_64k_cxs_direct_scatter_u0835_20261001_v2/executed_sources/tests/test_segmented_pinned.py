import ctypes
import mmap
import sys
from types import SimpleNamespace

import pytest
import segmented_pinned as allocator


def fake_runtime(monkeypatch, fail_at=None):
    calls, unregistered = [], []
    def register(ptr, size, flags):
        calls.append((ptr, size, flags))
        return 2 if len(calls) == fail_at else 0
    lib = SimpleNamespace(cudaHostRegister=register,
        cudaHostUnregister=lambda p: unregistered.append(p) or 0,
        cudaGetErrorString=lambda code: b'test failure')
    class Tensor:
        def __init__(self, region):
            self.region = region
        def data_ptr(self):
            return ctypes.addressof(ctypes.c_char.from_buffer(self.region))
        def is_pinned(self):
            return True
    torch = SimpleNamespace(cuda=SimpleNamespace(init=lambda: None, synchronize=lambda: None),
                            uint8='uint8', frombuffer=lambda r, dtype: Tensor(r))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(allocator, 'runtime', lambda: lib)
    return calls, unregistered


def test_contiguous_registration_and_release(monkeypatch):
    calls, frees = fake_runtime(monkeypatch)
    page = mmap.PAGESIZE
    buf = allocator.allocate(5*page, 2*page)
    ptr = buf.data_ptr()
    assert calls == [(ptr, 2*page, 0), (ptr+2*page, 2*page, 0), (ptr+4*page, page, 0)]
    assert ptr in allocator._live
    allocator.free(buf)
    assert frees == [ptr+4*page, ptr+2*page, ptr]
    assert ptr not in allocator._live
    buf.region.close()


def test_registration_failure_cleans_successful_regions(monkeypatch):
    calls, frees = fake_runtime(monkeypatch, fail_at=3)
    before = dict(allocator._live)
    with pytest.raises(RuntimeError, match='CUDA 2'):
        allocator.allocate(4*mmap.PAGESIZE, mmap.PAGESIZE)
    assert frees == [calls[1][0], calls[0][0]]
    assert allocator._live == before


def test_plugin_is_opt_in(monkeypatch):
    monkeypatch.delenv('DAOS_SEGMENTED_PINNED', raising=False)
    was_installed = allocator._installed
    allocator.install()
    assert allocator._installed == was_installed


def test_all_512gib_kv_chunks_fit_one_registration():
    c = allocator.CHUNK_BYTES
    for start in range(0, (512*2**30//c)*c, c):
        allocator.validate_chunk(start, c)
    with pytest.raises(AssertionError):
        allocator.validate_chunk(allocator.SEGMENT_BYTES-c//2, c)
    with pytest.raises(AssertionError):
        allocator.validate_chunk(0, c*2)
