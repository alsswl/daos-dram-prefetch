"""Opt-in contiguous host buffer with multiple bounded CUDA registrations.

LMCache still sees one tensor/allocator and the same LRU. No pageable fallback.
Only a requested 512GiB pool is replaced, never GPU staging or smaller buffers.
"""
import ctypes
import mmap
import os
import threading
import time

_live = {}
_lock = threading.Lock()
_installed = False
CHUNK_BYTES = 20*2**20
SEGMENT_BYTES = 400*CHUNK_BYTES  # 7.8125GiB, exact multiple of this experiment's KV chunk.


def validate_chunk(start, size):
    assert size == CHUNK_BYTES and start % CHUNK_BYTES == 0, 'Expected aligned 20MiB KV objects'
    assert start % SEGMENT_BYTES + size <= SEGMENT_BYTES, 'KV object crosses registration'


def runtime():
    lib = ctypes.CDLL('/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13')
    lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    lib.cudaHostRegister.restype = ctypes.c_int
    lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    lib.cudaHostUnregister.restype = ctypes.c_int
    lib.cudaGetErrorString.argtypes = [ctypes.c_int]
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    return lib


def check(lib, code, action):
    if code:
        raise RuntimeError(f'{action}: CUDA {code}: {lib.cudaGetErrorString(code).decode()}')


def allocate(size, segment_bytes=SEGMENT_BYTES):
    import torch
    assert size > 0 and size % mmap.PAGESIZE == 0
    assert segment_bytes > 0 and segment_bytes % mmap.PAGESIZE == 0
    lib = runtime()
    torch.cuda.init()
    region = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE)
    ptr = ctypes.addressof(ctypes.c_char.from_buffer(region))
    registered = []
    started = time.monotonic()
    try:
        for offset in range(0, size, segment_bytes):
            n = min(segment_bytes, size-offset)
            check(lib, lib.cudaHostRegister(ptr+offset, n, 0), 'cudaHostRegister')
            registered.append(ptr+offset)
            print(f'SEGMENTED_PINNED {min(offset+n,size)/2**30:.1f}/{size/2**30:.1f}GiB '
                  f'registered ({time.monotonic()-started:.1f}s)', flush=True)
        buffer = torch.frombuffer(region, dtype=torch.uint8)
        assert buffer.is_pinned()
        with _lock:
            _live[ptr] = (region, registered, lib)
        return buffer
    except BaseException:
        for address in reversed(registered):
            lib.cudaHostUnregister(address)
        region.close()
        raise


def free(buffer):
    import torch
    torch.cuda.synchronize()
    ptr = buffer.data_ptr()
    with _lock:
        region, registered, lib = _live[ptr]
    for address in reversed(registered):
        check(lib, lib.cudaHostUnregister(address), 'cudaHostUnregister')
    # torch.frombuffer keeps region alive until all tensor references are gone.
    # Drop our ownership after unregistering; do not close an exported buffer.
    with _lock:
        del _live[ptr]


def install():
    global _installed
    if os.environ.get('DAOS_SEGMENTED_PINNED') != '1' or _installed:
        return
    from lmcache.v1 import memory_management as mm
    from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
    original_alloc, original_free = mm._allocate_cpu_memory, mm._free_cpu_memory

    def alloc(size, numa_mapping=None, shm_name=None, use_hugepages=False):
        if size != 512*2**30:
            return original_alloc(size, numa_mapping, shm_name, use_hugepages)
        assert numa_mapping is None and shm_name is None and not use_hugepages
        return allocate(size)

    def release(buffer, size=None, numa_mapping=None, shm_name=None, use_hugepages=False):
        with _lock:
            ours = buffer.data_ptr() in _live
        if ours:
            free(buffer)
        else:
            original_free(buffer, size, numa_mapping, shm_name, use_hugepages)

    mm._allocate_cpu_memory, mm._free_cpu_memory = alloc, release
    original_slice = TensorMemoryAllocator._get_buffer_slice

    def checked_slice(self, start, size):
        with _lock:
            ours = self.buffer.data_ptr() in _live
        if ours:
            # A CUDA DMA may not span separately registered host regions.
            # Reject incompatible future models/chunk sizes, never silently copy.
            validate_chunk(start, size)
        return original_slice(self, start, size)

    TensorMemoryAllocator._get_buffer_slice = checked_slice
    _installed = True
    print(f'SEGMENTED_PINNED installed in PID {os.getpid()} (512GiB pool only)', flush=True)
