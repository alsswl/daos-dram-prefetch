"""Identical demand pipeline for a shared arena or physical 1:3 split.

Reserve not-yet-allocated bytes before submitting each window, convert the
reservation to physical occupancy chunk by chunk, release on last owner.
"""
import contextvars
from contextlib import contextmanager
import threading
import time
import torch
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from .gds_backend import _cfg
from .windowed_store_backend import WindowedStoreBackend
from .capacity_pipeline import capacity_pipeline

_ticket=contextvars.ContextVar('capacity_staging_ticket',default=None)

class Ticket:
    def __init__(self,budget,pool,size):
        self.budget,self.pool,self.remaining=budget,pool,size
        self.closed=False
    @contextmanager
    def activate(self):
        token=_ticket.set(self)
        try:yield self
        finally:_ticket.reset(token)
    def close(self):
        with self.budget.lock:
            if not self.closed:
                self.budget.pending[self.pool]-=self.remaining
                self.remaining=0;self.closed=True

class BudgetArena:
    def __init__(self,tensor,store_bytes,lock,chunk_bytes):
        self.lock=lock;self.chunk_bytes=chunk_bytes;self.split=store_bytes>0
        self.capacities=({'store':store_bytes,'retrieve':tensor.numel()-store_bytes}
                         if self.split else {'shared':tensor.numel()})
        buffers=({'store':tensor[:store_bytes],'retrieve':tensor[store_bytes:]}
                 if self.split else {'shared':tensor})
        self.parts={k:TensorMemoryAllocator(v,align_bytes=4096) for k,v in buffers.items()}
        self.used={'store':0,'retrieve':0};self.pending={'store':0,'retrieve':0};self.owners={}
    @property
    def total_allocated_size(self):return sum(self.used.values())
    @property
    def num_active_allocations(self):return len(self.owners)
    def pool_usage(self):
        return {f'{k}_used_bytes':v for k,v in self.used.items()}
    def try_reserve(self,pool,size):
        with self.lock:
            occupied=(self.used[pool]+self.pending[pool] if self.split
                      else sum(self.used.values())+sum(self.pending.values()))
            capacity=self.capacities[pool if self.split else 'shared']
            if occupied+size>capacity:return None
            self.pending[pool]+=size
            return Ticket(self,pool,size)
    def allocate(self,*args,**kwargs):
        ticket=_ticket.get()
        if ticket is None or ticket.budget is not self or ticket.closed:
            raise RuntimeError('GPU staging allocation requires a live admission ticket')
        with self.lock:
            pool=ticket.pool;part=self.parts[pool if self.split else 'shared']
            obj=part.allocate(*args,**kwargs)
            if obj is None:return None
            size=(obj.get_size()+4095)//4096*4096
            if size>ticket.remaining:
                part.free(obj);raise RuntimeError('Allocation exceeded window reservation')
            ticket.remaining-=size;self.pending[pool]-=size;self.used[pool]+=size
            self.owners[id(obj)]=(pool,part,size);obj.parent_allocator=self
            return obj
    def batched_allocate(self,*args,**kwargs):
        raise RuntimeError('This full-chunk experiment uses individual reserved allocations')
    def free(self,obj,*args,**kwargs):
        with self.lock:
            pool,part,size=self.owners.pop(id(obj))
            part.free(obj,*args,**kwargs);self.used[pool]-=size
    def batched_free(self,objects,*args,**kwargs):
        for obj in objects:self.free(obj)
    def memcheck(self):return all(p.memcheck() for p in self.parts.values())

class BudgetGPUAllocator(GPUMemoryAllocator):
    def __init__(self,size,store_bytes,device,chunk_bytes):
        self.tensor=torch.empty(size,dtype=torch.uint8,device=device)
        self.device_mem_lock=threading.RLock()
        self.allocator=BudgetArena(self.tensor,store_bytes,self.device_mem_lock,chunk_bytes)

class CapacityPipelineBackend(WindowedStoreBackend):
    propagate_read_context=True
    def initialize_allocator(self,config,metadata=None):
        store_bytes=int(float(_cfg(config,'store_staging_gib',0))*2**30)
        if not 0<=store_bytes<self.gpu_buffer_bytes:raise ValueError('Invalid store split')
        self.split_store_bytes=store_bytes
        return BudgetGPUAllocator(self.gpu_buffer_bytes,store_bytes,self.dst_device,self.store_chunk_bytes)
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.store_lock=threading.RLock()
        self.budget=self.memory_allocator.allocator
        capacity=self.gpu_buffer_bytes-self.split_store_bytes
        self.pipeline_depth=capacity//(self.window_chunk_count*self.store_chunk_bytes)
        if self.pipeline_depth<2:raise ValueError('At least two read windows required')
        if self.split_store_bytes and self.store_window_chunks*self.store_chunk_bytes>self.split_store_bytes:
            raise ValueError('Store window exceeds its physical arena')
        self._staging_trace.emit('capacity_pipeline_enabled',shared=not bool(self.split_store_bytes),
            capacity_bytes=self.gpu_buffer_bytes,store_capacity_bytes=self.split_store_bytes,
            retrieve_capacity_bytes=capacity,pipeline_depth=self.pipeline_depth,
            effective_window_bytes=self.window_chunk_count*self.store_chunk_bytes)
    def reserve_store_space(self,required,emit):
        started=time.monotonic()
        while True:
            ticket=self.budget.try_reserve('store',required)
            if ticket is not None:
                elapsed=time.monotonic()-started
                if elapsed>=.001:emit('store_window_wait',required_bytes=required,wait_ms=elapsed*1000)
                return ticket,_ticket.set(ticket)
            if time.monotonic()-started>=self.store_window_timeout_s:
                raise TimeoutError('Store admission timed out')
            time.sleep(.001)
    def finish_store_admission(self,admission):
        ticket,token=admission
        _ticket.reset(token);ticket.close()
    def transfer_retrieve(self,spans,load,copy,release,emit):
        return capacity_pipeline(spans,self.window_chunk_count,self.pipeline_depth,
            load,copy,release,emit,self.budget,self.window_timeout_s,self._ensure_cuda_ctx)
