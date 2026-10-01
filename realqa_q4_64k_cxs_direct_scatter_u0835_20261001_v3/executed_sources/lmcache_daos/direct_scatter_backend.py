"""Experimental synchronous DAOS <-> final vLLM KV pages, no payload staging.

Use direct_scatter_launch before vLLM initializes LMCache. New array format,
private namespace, one GPU, no DRAM cache/promotion/prefetch. Writes block until
DAOS has consumed source pages: unlike the staging baseline, stores are NOT
asynchronous. Never fall back to a payload allocator or a scatter kernel.
"""
import concurrent.futures as futures
import functools
import json
import os
from pathlib import Path
import threading
import time

import torch
from lmcache.logging import init_logger

from .gds_backend import DaosGdsBackend, _cfg
from .object_binding import DaosObjectError
from .scatter_binding import ScatterObjectStore
from .scatter_plan import plan_segments

logger = init_logger(__name__)
SCHEMA = "daos-direct-scatter-array-v1"


class NoPayloadAllocator:
    """Framework bookkeeping only; any accidental payload allocation is a bug."""
    align_bytes = 4096
    def allocate(self, *args, **kwargs):
        raise RuntimeError("Payload staging is forbidden in direct-scatter mode")
    batched_allocate = allocate
    free = allocate
    batched_free = allocate
    def memcheck(self):
        return True
    def close(self):
        pass


def direct_enabled(config):
    return _cfg(config, "direct_scatter", False) is True


def validate_config(config, metadata):
    if not direct_enabled(config):
        raise ValueError("Set daosgds.direct_scatter=true")
    for name in ("enable_async_loading", "use_layerwise", "use_gpu_connector_v3",
                 "local_cpu", "local_disk", "enable_blending", "enable_pd",
                 "enable_p2p", "enable_controller", "enable_kv_events"):
        if getattr(config, name, False):
            raise ValueError(f"Direct scatter requires {name}=false")
    if getattr(config, "remote_url", None):
        raise ValueError("Direct scatter does not support a remote cache tier")
    for name in ("async_dram", "dram_prefetch", "dram_promote_on_read"):
        if _cfg(config, name, False):
            raise ValueError(f"Direct scatter requires daosgds.{name}=false")
    if float(_cfg(config, "gpu_buffer_gb", 0)) != 0:
        raise ValueError("Direct scatter requires gpu_buffer_gb=0")
    if metadata is None or metadata.world_size != 1 or metadata.use_mla:
        raise ValueError("Direct scatter requires single-rank, non-MLA metadata")
    if metadata.kv_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Direct scatter v1 supports FP16/BF16 KV only")
    if os.environ.get("DAOSGDS_TRANSPORT", _cfg(config, "transport", "object")) != "object":
        raise ValueError("Direct scatter requires object transport")
    if not str(_cfg(config, "object_namespace", "")).strip():
        raise ValueError("A private object_namespace is required")


def find_backend(engine):
    manager = engine.storage_manager
    if manager is None:
        return None
    selected = [b for b in manager.storage_backends.values()
                if isinstance(b, DirectScatterBackend)]
    if not selected:
        if direct_enabled(engine.config):
            raise RuntimeError("Direct-scatter backend failed to initialize; refusing fallback")
        return None
    if len(selected) != 1 or any(
        name != "LocalCPUBackend" and b is not selected[0]
        for name, b in manager.storage_backends.items()
    ):
        raise ValueError("Direct scatter requires exactly one storage plugin")
    return selected[0]


def install_engine_hooks():
    from lmcache.v1.cache_engine import LMCacheEngine
    if getattr(LMCacheEngine, "_daos_direct_scatter_installed", False):
        return
    original_process = LMCacheEngine._process_tokens_internal
    original_store = LMCacheEngine.store

    @functools.wraps(original_process)
    def process(engine, tokens, mask, ret_mask, **kwargs):
        backend = find_backend(engine)
        if backend is None:
            return original_process(engine, tokens, mask, ret_mask, **kwargs)
        return backend.retrieve_into(engine, tokens, mask, ret_mask, **kwargs)

    @functools.wraps(original_store)
    def store(engine, tokens=None, hashes=None, offsets=None, mask=None, **kwargs):
        backend = find_backend(engine)
        if backend is None:
            return original_store(engine, tokens, hashes, offsets, mask, **kwargs)
        if (not engine.is_healthy() or engine._is_passive() or engine.is_frozen()
                or not backend.store_enabled):
            return None
        if tokens is None:
            raise ValueError("Direct scatter v1 requires token-based store")
        return backend.store_from(engine, tokens, mask, **kwargs)

    LMCacheEngine._process_tokens_internal = process
    LMCacheEngine.store = store
    LMCacheEngine._daos_direct_scatter_installed = True


class DirectScatterBackend(DaosGdsBackend):
    def __init__(self, config, dst_device="cuda", metadata=None,
                 local_cpu_backend=None, loop=None):
        # Deliberately do NOT call DaosGdsBackend.__init__: it reserves a GPU
        # pool and installs MemoryObj/prefetch routing that this backend bypasses.
        validate_config(config, metadata)
        import lmcache.v1.gpu_connector as connectors
        if not getattr(connectors, "_daos_direct_launch", False):
            raise RuntimeError("Use python -m lmcache_daos.direct_scatter_launch")
        self.config, self.metadata, self.loop = config, metadata, loop
        self.local_cpu_backend = local_cpu_backend
        dev = torch.device(dst_device)
        if dev.type != "cuda":
            raise ValueError("Direct scatter needs a CUDA worker")
        self.device_id = dev.index if dev.index is not None else torch.cuda.current_device()
        self.dst_device = f"cuda:{self.device_id}"
        self.transport, self.store_path = "object", "direct_scatter"
        # Automatic format suffix makes it impossible to read/overwrite baseline
        # SINGLE-value dkeys even if the user reuses a base experiment namespace.
        self.object_namespace = str(_cfg(config, "object_namespace")) + "scatter-v1:"
        self.gpu_buffer_bytes = 0
        self.memory_allocator = NoPayloadAllocator()
        self.store_enabled = bool(_cfg(config, "store", True))
        self.io_workers = int(_cfg(config, "io_workers", 8))
        self.meta_workers = int(_cfg(config, "meta_workers", 8))
        if min(self.io_workers, self.meta_workers) < 1:
            raise ValueError("Worker counts must be positive")
        self._tls = threading.local()
        self._operation_lock = threading.RLock()
        self._known_lock, self._put_lock = threading.Lock(), threading.Lock()
        self._known, self._put_tasks = set(), set()
        self._dfs = None
        self._staging_trace = None
        self.stats = {"put": 0, "get": 0, "put_bytes": 0, "get_bytes": 0,
                      "miss": 0, "errors": 0, "staging_bytes": 0, "iov_count": 0}
        library = _cfg(config, "object_library", str(
            Path(__file__).resolve().parent.parent / "libdaosgdr_scatter.so"))
        self._ensure_cuda_ctx()
        self._object = ScatterObjectStore(_cfg(config, "pool"), _cfg(config, "container"),
                                          library_path=library)
        self._pool = futures.ThreadPoolExecutor(self.io_workers, initializer=self._ensure_cuda_ctx)
        self._meta_pool = futures.ThreadPoolExecutor(self.meta_workers, initializer=self._ensure_cuda_ctx)
        install_engine_hooks()
        logger.info("DirectScatterBackend enabled: GPU staging=0, synchronous SGL store/retrieve, namespace=%s",
                    self.object_namespace)

    def calculate_chunk_budget(self):
        return self.io_workers

    def get_blocking(self, key):
        raise RuntimeError("Direct scatter requires final KV destinations; MemoryObj get is forbidden")

    def batched_submit_put_task(self, *args, **kwargs):
        raise RuntimeError("Direct scatter requires source KV pages; MemoryObj put is forbidden")

    async def batched_get_non_blocking(self, *args, **kwargs):
        raise RuntimeError("Direct scatter does not support prefetch")

    def _meta(self, n):
        l, _, _, h, d = self.metadata.kv_shape
        return {"schema": SCHEMA, "layers": l, "tokens": n, "hidden": h*d,
                "dtype": str(self.metadata.kv_dtype), "itemsize": 2,
                "length": 2*l*n*h*d*2}

    def contains(self, key, pin=False):
        self._ensure_cuda_ctx()
        try:
            raw = self._object.stat(self._object_key(key))
            if raw is None:
                return False
            meta = json.loads(raw)
            n = meta.get("tokens")
            return (type(n) is int and 0 < n <= self.config.chunk_size
                    and meta == self._meta(n))
        except (DaosObjectError, ValueError, TypeError, AttributeError):
            return False

    def _prepare(self, engine, kwargs):
        from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
        if (engine.async_loading or engine.save_only_first_rank or engine.remove_after_retrieve
                or getattr(engine, "kv_events_enabled", False)):
            raise ValueError("Unsupported direct-scatter engine configuration")
        connector = engine.gpu_connector
        if type(connector) is not VLLMPagedMemGPUConnectorV2:
            raise ValueError("Direct scatter v1 requires VLLMPagedMemGPUConnectorV2")
        if connector.gpu_buffer is not None:
            raise RuntimeError("Start with python -m lmcache_daos.direct_scatter_launch to disable connector staging")
        connector.initialize_kvcaches_ptr(**kwargs)
        tensors = connector.kvcaches
        if not tensors or len(tensors) != self.metadata.kv_shape[0]:
            raise ValueError("KV layer count mismatch")
        # Use the installed detector, including its ambiguity/layout hints.
        connector._initialize_pointers(tensors)
        name = connector.engine_kv_format.name
        layouts = {"NL_X_NB_TWO_BS_NH_HS": "NB_TWO", "NL_X_TWO_NB_BS_NH_HS": "TWO_NB"}
        if name not in layouts:
            raise ValueError(f"Unsupported physical KV format: {name}")
        layout = layouts[name]
        geometry = None
        allocations = []
        for t in tensors:
            if not t.is_cuda or t.device.index != self.device_id or t.dtype != self.metadata.kv_dtype:
                raise ValueError("KV device/dtype mismatch")
            if not t.is_contiguous() or t.ndim != 5:
                raise ValueError("Direct scatter requires contiguous 5D layer tensors")
            shape = tuple(t.shape)
            nb, two, bs, nh, hs = shape if layout == "NB_TWO" else (shape[1],shape[0],*shape[2:])
            if two != 2 or (nh, hs) != tuple(self.metadata.kv_shape[-2:]):
                raise ValueError("KV shape mismatch")
            current = (nb, bs, nh*hs, t.element_size())
            if geometry is not None and geometry != current:
                raise ValueError("Mixed layer geometry is not supported")
            geometry = current
            allocations.append((t.data_ptr(), t.data_ptr() + t.numel()*t.element_size()))
        ranges = sorted(allocations)
        if any(a[1] > b[0] for a,b in zip(ranges, ranges[1:])):
            raise ValueError("Aliased layer allocations")
        slot_tensor = kwargs["slot_mapping"]
        if isinstance(slot_tensor, torch.Tensor):
            if slot_tensor.ndim != 1 or slot_tensor.dtype != torch.int64:
                raise ValueError("slot_mapping must be a 1D int64 tensor")
            slots = slot_tensor.detach().cpu().tolist()
        else:
            slots = list(slot_tensor)
        self._ensure_cuda_ctx()
        # Conservative first version: prior CUDA accesses finish before NIC
        # accesses. Keep tensors referenced until all synchronous I/O completes.
        torch.cuda.synchronize(self.device_id)
        return tensors, [a[0] for a in allocations], geometry, slots, layout

    def _transfer(self, key, segments, meta, write):
        self._ensure_cuda_ctx()
        object_key = self._object_key(key)
        if write:
            self._object.putv(object_key, segments, json.dumps(meta, sort_keys=True).encode(), self.device_id)
        else:
            raw = self._object.stat(object_key)
            if raw is None or json.loads(raw) != meta:
                raise ValueError("Missing or incompatible direct-scatter metadata")
            if segments:
                self._object.getv(object_key, segments, self.device_id)
        return sum(s.length for s in segments)

    def _batch(self, jobs, write):
        pending = []
        try:
            for key, segments, meta in jobs:
                pending.append(self._pool.submit(self._transfer, key, segments, meta, write))
        finally:
            # Even a submit failure / interruption must not release source or
            # destination pages while any worker can still perform RDMA.
            futures.wait(pending)
        results = []
        for f in pending:
            try:
                results.append((f.result(), None))
            except Exception as e:
                results.append((0, e))
        return results

    def _operate(self, engine, infos, ret_mask, kwargs, write):
        total, completed = 0, 0
        started = time.perf_counter()
        with self._operation_lock:
            tensors, bases, geometry, slots, layout = self._prepare(engine, kwargs)
            # Validate every token that will be accessed before submitting ANY
            # I/O, including cross-chunk aliasing and cached-prefix aliases.
            skip_to = 0 if write else int(kwargs.get("vllm_cached_tokens", 0))
            selected = [t for start,end,_ in infos for t in range(max(start,skip_to),end)]
            if any(t >= len(slots) for t in selected):
                raise ValueError("slot_mapping is shorter than the requested tokens")
            active = [slots[t] for t in selected]
            if (any(type(s) is not int or not 0 <= s < geometry[0]*geometry[1] for s in active)
                    or len(set(active)) != len(active)):
                raise ValueError("Invalid or aliased active slots")
            if not write and set(active).intersection(s for s in slots[:skip_to] if s >= 0):
                raise ValueError("Read would overwrite a cached prefix")
            try:
                for at in range(0, len(infos), self.io_workers):
                    group = infos[at:at+self.io_workers]
                    jobs = []
                    for start,end,key in group:
                        if not 0 <= start < end <= len(slots):
                            raise ValueError("Invalid chunk span")
                        skip = min(end-start, max(0,skip_to-start))
                        segments = plan_segments(bases, *geometry, slots[start:end], skip=skip, layout=layout)
                        jobs.append((key,segments,self._meta(end-start)))
                    results = self._batch(jobs, write)
                    failed = False
                    for (start,end,key), (_,segments,_), (size,error) in zip(group,jobs,results):
                        if error is not None:
                            self.stats["errors"] += 1
                            logger.error("Direct scatter %s failed for %s: %s", "store" if write else "fetch", key, error)
                            failed = True
                        if failed:
                            continue
                        first = start if write else max(start, min(end, skip_to))
                        if ret_mask is not None:
                            # Report ONLY tokens actually fetched. Including the
                            # skipped vLLM prefix could hide a short read from the
                            # adapter's num_retrieved < num_expected check.
                            ret_mask[first:end] = True
                        total += size
                        completed += end-first
                        op = "put" if write else "get"
                        self.stats[op] += 1
                        self.stats[op+"_bytes"] += size
                        self.stats["iov_count"] += len(segments)
                    if failed:
                        if write:
                            raise RuntimeError("Direct scatter store failed; all submitted I/O drained")
                        self.stats["miss"] += 1
                        break
            finally:
                # CPU-initiated CUDA barrier after RDMA completion; no attention
                # launch, page reuse, or valid return before this point.
                torch.cuda.synchronize(self.device_id)
                # 'tensors' intentionally stays strongly referenced until here.
        logger.info("[req_id=%s] Direct scatter %s: tokens=%d bytes=%d cost_ms=%.3f staging_bytes=0",
                    kwargs.get("req_id", "unspecified"), "store" if write else "fetch",
                    completed, total, (time.perf_counter()-started)*1000)
        return total, completed

    def retrieve_into(self, engine, tokens, mask, ret_mask, **kwargs):
        infos = list(engine.token_database.process_tokens(tokens=tokens, mask=mask,
                     request_configs=kwargs.get("request_configs")))
        if not infos:
            return [], 0
        total, _ = self._operate(engine, infos, ret_mask, kwargs, False)
        # Native retrieve records metrics but has no MemoryObjs to scatter/free.
        return [], total

    def store_from(self, engine, tokens, mask, **kwargs):
        required = int(mask.sum()) if mask is not None else len(tokens)
        stats = engine.stats_monitor.on_store_request(required)
        completed = 0
        try:
            with stats.profile_process_tokens():
                infos = list(engine.token_database.process_tokens(tokens=tokens, mask=mask,
                             request_configs=kwargs.get("request_configs")))
            if infos:
                with stats.profile_put():
                    _, completed = self._operate(engine, infos, None, kwargs, True)
        finally:
            engine.stats_monitor.on_store_finished(stats, completed)

    def __str__(self):
        return "DirectScatterBackend"
