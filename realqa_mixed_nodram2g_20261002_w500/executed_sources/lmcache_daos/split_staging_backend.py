"""Opt-in physical store/read arena split and three-window demand pipeline."""
import contextvars
import threading
import torch
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from .gds_backend import _cfg
from .windowed_store_backend import WindowedStoreBackend
from .split_pipeline import pipeline_windows

_allocation_pool = contextvars.ContextVar('daos_split_allocation_pool', default='retrieve')


class SplitArena:
    def __init__(self, tensor, store_bytes, lock):
        self.lock = lock
        self.parts = {'store': TensorMemoryAllocator(tensor[:store_bytes], align_bytes=4096),
                      'retrieve': TensorMemoryAllocator(tensor[store_bytes:], align_bytes=4096)}
        self.boundary = tensor.data_ptr()+store_bytes

    @property
    def total_allocated_size(self):
        return sum(p.total_allocated_size for p in self.parts.values())

    @property
    def num_active_allocations(self):
        return sum(p.num_active_allocations for p in self.parts.values())

    def pool_usage(self):
        return {f'{name}_used_bytes': part.total_allocated_size for name,part in self.parts.items()}

    def allocate(self, *args, **kwargs):
        with self.lock:
            obj = self.parts[_allocation_pool.get()].allocate(*args, **kwargs)
            if obj is not None:
                obj.parent_allocator = self
            return obj

    def batched_allocate(self, shapes, dtypes, batch_size, fmt, allocator_type=None):
        with self.lock:
            objs = self.parts[_allocation_pool.get()].batched_allocate(
                shapes, dtypes, batch_size, fmt, allocator_type)
            if objs:
                for obj in objs:
                    obj.parent_allocator = self
            return objs

    def free(self, obj, *args, **kwargs):
        with self.lock:
            name = 'store' if obj.tensor.data_ptr()<self.boundary else 'retrieve'
            self.parts[name].free(obj, *args, **kwargs)

    def batched_free(self, objects, *args, **kwargs):
        for obj in objects:
            self.free(obj)

    def memcheck(self):
        return all(p.memcheck() for p in self.parts.values())


class SplitGPUMemoryAllocator(GPUMemoryAllocator):
    def __init__(self, size, store_bytes, device):
        self.tensor = torch.empty(size, dtype=torch.uint8, device=device)
        self.device_mem_lock = threading.RLock()
        self.allocator = SplitArena(self.tensor, store_bytes, self.device_mem_lock)


class SplitStagingBackend(WindowedStoreBackend):
    def initialize_allocator(self, config, metadata=None):
        self.store_capacity_bytes = int(float(_cfg(config, 'store_staging_gib', .5))*2**30)
        self.retrieve_capacity_bytes = self.gpu_buffer_bytes-self.store_capacity_bytes
        if not 0 < self.store_capacity_bytes < self.gpu_buffer_bytes:
            raise ValueError('Both split arenas must have positive capacity')
        return SplitGPUMemoryAllocator(self.gpu_buffer_bytes, self.store_capacity_bytes, self.dst_device)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.store_lock = threading.RLock()
        window_bytes = self.window_chunk_count*self.store_chunk_bytes
        self.pipeline_depth = self.retrieve_capacity_bytes//window_bytes
        if (self.store_window_chunks*self.store_chunk_bytes > self.store_capacity_bytes
                or self.pipeline_depth < 2):
            raise ValueError('Store window must fit its arena; retrieve needs >=2 windows')
        self._staging_trace.emit('split_staging_enabled',
            store_capacity_bytes=self.store_capacity_bytes,
            retrieve_capacity_bytes=self.retrieve_capacity_bytes,
            pipeline_depth=self.pipeline_depth, effective_window_bytes=window_bytes)

    def allocate(self, *args, **kwargs):
        # StorageManager allocates store payloads here. DAOS get allocates
        # directly through memory_allocator and therefore selects retrieve.
        token = _allocation_pool.set('store')
        try:
            return super().allocate(*args, **kwargs)
        finally:
            _allocation_pool.reset(token)

    def store_free_bytes(self):
        return self.store_capacity_bytes-self.memory_allocator.allocator.parts['store'].total_allocated_size

    def transfer_retrieve(self, spans, load, copy, release, emit):
        return pipeline_windows(spans, self.window_chunk_count, self.pipeline_depth,
            load, copy, release, emit, timeout=self.window_timeout_s,
            initializer=self._ensure_cuda_ctx)
