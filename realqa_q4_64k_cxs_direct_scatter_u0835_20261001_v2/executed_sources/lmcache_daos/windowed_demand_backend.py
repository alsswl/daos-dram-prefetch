"""Opt-in non-layerwise, single-rank synchronous windowed DAOS retrieve.

Set retrieve_window_mib=0 for the unchanged DemandReadBackend baseline. This
first version serializes read/copy/release; it is NOT an overlap pipeline.
No installed LMCache file is edited. Async metadata/payload mode is rejected.
"""
import functools
import math
import os
import threading
import time

from .demand_read_backend import DemandReadBackend
from .gds_backend import _cfg
from .windowed_transfer import transfer_windows, window_chunks


def install_window_hook():
    from lmcache.v1.cache_engine import LMCacheEngine
    if getattr(LMCacheEngine, '_daos_windowed_installed', False):
        return
    original = LMCacheEngine._process_tokens_internal

    @functools.wraps(original)
    def process(engine, tokens, mask, ret_mask, **kwargs):
        selected = getattr(engine.storage_manager, '_daos_gpu_store', None)
        backend = selected[1] if selected else None
        if not isinstance(backend, WindowedDemandBackend) or not backend.window_chunk_count:
            return original(engine,tokens,mask,ret_mask,**kwargs)
        if engine.async_loading or engine.save_only_first_rank or engine.remove_after_retrieve:
            raise ValueError('Windowed v1 requires sync, single-rank, remove_after_retrieve=false')
        connector = engine.gpu_connector
        # This connector has a synchronous batched_to_gpu and a public load stream.
        from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
        if type(connector) is not VLLMPagedMemGPUConnectorV2:
            raise ValueError('Windowed v1 validated only for VLLMPagedMemGPUConnectorV2')
        rid = engine._get_req_id(kwargs)
        trace = backend._staging_trace
        emit = lambda name, **data: trace.emit(name,request_id=rid,**data)
        infos = list(engine.token_database.process_tokens(tokens=tokens,mask=mask,
            request_configs=kwargs.get('request_configs')))
        spans = [(s,e) for s,e,_ in infos]
        if not spans:
            return [],0
        import torch

        def load(start,end):
            # Prefix tokens remain present for stable chained key hashes. The mask
            # selects ONLY the current window, with absolute model slot offsets.
            submask = torch.zeros(end,dtype=torch.bool,device='cpu')
            submask[start:end] = True
            temporary = torch.zeros(end,dtype=torch.bool,device='cpu')
            items,_ = original(engine,tokens[:end],submask,temporary,**kwargs)
            return items

        def copy(items):
            _,objects,starts,ends = zip(*items)
            connector.batched_to_gpu(list(objects),list(starts),list(ends),**kwargs)
            # Native V2 synchronizes internally; mark valid only after it returns.
            for _,_,start,end in items:
                ret_mask[start:end] = True

        def release(items):
            # Also synchronize on exception, before releasing storage backing DMA.
            connector.load_stream.synchronize()
            gpu=[]
            for _,obj,_,_ in items:
                if obj.tensor is not None and obj.tensor.is_cuda:
                    gpu.append(obj)
                if obj.is_pinned:
                    obj.unpin()
                backend._release_memory_obj(obj)
            # D2H promotion may own another ref. Account for its entire lifetime,
            # rather than pretending space is free when the GDS read completes.
            deadline=time.monotonic()+backend.window_timeout_s
            while any(obj.get_ref_count()>0 for obj in gpu):
                if time.monotonic()>=deadline:
                    raise TimeoutError('Window copy completed but a DRAM mirror still holds its GPU buffers')
                time.sleep(.001)

        # Serialize retrieve windows across callers of this backend. Store writes
        # still share the physical allocator; transient pressure gets bounded retry.
        with backend.window_lock:
            emit('window_retrieve_start',window_mib=backend.window_mib,
                 window_chunks=backend.window_chunk_count,total_chunks=len(spans))
            total=transfer_windows(spans,backend.window_chunk_count,load,copy,release,
                                   emit,timeout=backend.window_timeout_s)
            emit('window_retrieve_done',bytes=total,chunks=len(spans))
        # Already copied and released: outer native retrieve must NOT scatter or
        # free them again. Its total-token mask and request-level metrics remain.
        return [],total

    LMCacheEngine._process_tokens_internal = process
    LMCacheEngine._daos_windowed_installed = True


class WindowedDemandBackend(DemandReadBackend):
    def __init__(self,config,dst_device='cuda',metadata=None,local_cpu_backend=None,loop=None):
        if not os.environ.get('DAOS_GDS_STAGING_TRACE'):
            raise ValueError('Set DAOS_GDS_STAGING_TRACE to a writable log prefix for this experimental backend')
        if metadata is None or metadata.world_size != 1 or metadata.use_mla:
            raise ValueError('Windowed v1 requires single-rank, non-MLA metadata')
        window=_cfg(config,'retrieve_window_mib',512)
        timeout=_cfg(config,'retrieve_window_timeout_s',5.0)
        if isinstance(timeout,bool) or not math.isfinite(float(timeout)) or timeout<=0:
            raise ValueError('retrieve_window_timeout_s must be finite and > 0')
        chunk_bytes=sum(math.prod(s)*d.itemsize for s,d in zip(metadata.get_shapes(),metadata.get_dtypes()))
        # GPU allocator rounds every object to 4KiB.
        chunk_bytes=(chunk_bytes+4095)//4096*4096
        limit=window_chunks(window,chunk_bytes,int(_cfg(config,'gpu_buffer_gb',8)*2**30))
        self.window_mib,self.window_chunk_count,self.window_timeout_s=window,limit,float(timeout)
        self.window_lock=threading.RLock()
        super().__init__(config,dst_device,metadata,local_cpu_backend,loop)
        install_window_hook()
        self._staging_trace.emit('windowed_demand_enabled',window_mib=window,
            effective_window_bytes=limit*chunk_bytes,chunk_bytes=chunk_bytes,
            timeout_s=self.window_timeout_s,pipeline=False)
