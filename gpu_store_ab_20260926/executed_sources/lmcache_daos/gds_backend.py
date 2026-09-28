"""Unified LMCache DAOS GPU-direct storage plugin.

``daosgds.transport`` selects either DFS GPU I/O (``dfs_read_gpu`` /
``dfs_write_gpu``) or DFS-bypassing object I/O (``daos_obj_fetch_gpu`` /
``daos_obj_update_gpu``). Both paths transfer KV payloads directly between
DAOS and GPU staging memory, bypassing host staging.

How it plugs in (LMCache 0.5.2, no upstream change)::

    storage_plugins: ["daosgds"]
    local_cpu: false                 # keep the CPU backend allocator-only
    extra_config:
      storage_plugin.daosgds.module_path: lmcache_daos.gds_backend
      storage_plugin.daosgds.class_name: DaosGdsBackend
      daosgds.pool: discospool
      daosgds.container: kvcache     # healthy POSIX container; usable by both paths
      daosgds.transport: dfs         # or object
      daosgds.object_namespace: minji-v2:
      daosgds.gpu_buffer_gb: 6       # GPU staging pool for get/put objects
      daosgds.io_workers: 16
    enable_async_loading: False

What the storage manager does with it (``storage_manager.py``):
- **store (default host_staged)**: the engine allocates host objects in LocalCPUBackend (the fixed
  allocator backend); because ``get_allocator_backend()`` here returns *this*
  backend, the manager copies each object into our GPU allocator
  (``allocate_and_copy_objects``, D2D-free H2D on its copy stream) and calls
  ``batched_submit_put_task`` with the GPU copies. We write them with
  ``dfs_write_gpu``. With ``local_cpu: false`` nothing else keeps the host copy.
- **store (opt-in gpu_direct)**: ``daosgds.store_path: gpu_direct`` selects
  our GPU allocator for engine extraction and submits those same GPU objects
  directly, skipping disabled CPU storage. Requires ``local_cpu: false``.
  Keys and metadata still use host memory; KV payload does not.
- **retrieve**: ``batched_get_blocking`` allocates GPU objects from our pool,
  ``dfs_read_gpu`` lands the payload in them, and the GPU connector's
  ``to_gpu`` kernel scatters from device memory into the paged KV cache. The
  host DRAM is not on the data path.

DFS format is v2 (``serde_v2``): [4 KiB header page][payload at 4096]. Object
mode stores equivalent JSON metadata in the ``meta`` akey and GPU bytes in the
``kv`` akey. Both let the reader size the GPU allocation before fetching the
payload.

Runtime requirements (gpudirect/README.md, "GDS over ofi+verbs"): GPU-direct
DAOS client bundle (``LMCACHE_DAOS_LIBDIR``), CUDA-enabled libfabric with the
verbs dmabuf patch, ``D_MEM_DEVICE=1``, nvidia open kernel module.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future
from typing import Any, Callable, List, Optional, Sequence, Union

import torch

from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.utils import CacheEngineKey

from . import serde_v2 as v2
from .dfs_binding import DaosError, DfsSys, warm_up_all_targets
from .object_binding import DaosObjectError, DaosObjectStore

logger = init_logger(__name__)


_LAST_BACKEND: Optional["DaosGdsBackend"] = None


def _install_multi_prefetch_serializer() -> None:
    """LMCache 0.5.2 hard-codes ``AsyncSingleSerializer`` (an asyncio.Lock: one
    request's prefetch at a time) although ``AsyncMultiSerializer`` -- a
    chunk-budget weighted semaphore -- sits next to it with no config switch.
    Prefetches that cannot overlap cap the aggregate at one request's read
    bandwidth (Part B: 21-26 GB/s vs the 36 GB/s the MP server reaches).

    This module is imported by the plugin launcher *inside*
    ``StorageManager.__init__`` before the serializer is constructed, so
    rebinding the name in that module makes the manager build the multi
    serializer over our GPU pool's chunk budget. Off with
    ``DAOS_GDS_MULTI_PREFETCH=0``."""
    if os.environ.get("DAOS_GDS_MULTI_PREFETCH", "1") != "1":
        return
    try:
        from lmcache.v1.storage_backend import storage_manager as smm
    except Exception:  # pragma: no cover
        return
    if getattr(smm, "_daos_gds_multi_patched", False):
        return
    multi = smm.AsyncMultiSerializer

    def factory(loop):
        be = _LAST_BACKEND
        if be is None:
            return smm.__dict__["_daos_gds_single"](loop)
        ser = multi(be, loop)
        if getattr(be, "_staging_trace", None) is not None:
            be._staging_trace.wrap_serializer(ser)
        logger.info("DaosGdsBackend: prefetch serializer = AsyncMultiSerializer (chunk budget %d, concurrent cap %d)",
                    ser.chunk_budget, ser.chunk_budget // 2)
        return ser

    smm._daos_gds_single = smm.AsyncSingleSerializer
    smm.AsyncSingleSerializer = factory
    smm._daos_gds_multi_patched = True


def _install_consumed_prefetch_event_patch() -> None:
    """Remove a completed async-loading event once retrieve consumes it.

    LMCache 0.5.2 keeps the completed future in ``EventManager`` after
    ``_async_process_tokens_internal`` has handed its MemoryObjs to retrieve.
    Retrieve releases those objects, and the later normal ``lookup_unpin``
    cleanup releases the same future contents a second time.  Direct GPU
    staging exposes this as a negative refcount warning on every cache hit.

    Popping only after successful consumption preserves the abort path: an
    event that was prefetched but never consumed remains available to
    ``cleanup_memory_objs``.
    """
    try:
        from lmcache.v1 import cache_engine as cem
        from lmcache.v1.event_manager import EventStatus, EventType
    except Exception:  # pragma: no cover
        return
    cls = cem.LMCacheEngine
    if getattr(cls, "_daos_gds_prefetch_event_patched", False):
        return
    original = cls._async_process_tokens_internal

    @functools.wraps(original)
    def consume_once(engine, *args, **kwargs):
        result = original(engine, *args, **kwargs)
        lookup_id = kwargs.get("req_id")
        if (
            lookup_id is not None
            and engine.event_manager.get_event_status(EventType.LOADING, lookup_id)
            == EventStatus.DONE
        ):
            engine.event_manager.pop_event(EventType.LOADING, lookup_id)
        return result

    cls._async_process_tokens_internal = consume_once
    cls._daos_gds_prefetch_event_patched = True
    logger.info("DaosGdsBackend: installed consumed-prefetch event ownership fix")


def _cfg(config: LMCacheEngineConfig, key: str, default=None):
    ec = config.extra_config or {}
    return ec.get(f"daosgds.{key}", default)


def _meta_pack(length: int, shapes, dtypes, fmt: MemoryFormat) -> bytes:
    """Self-contained object metadata (JSON, ~150 B). LMCache's RemoteMetadata
    codec needs process-global initialisation by the RemoteBackend, so a
    storage plugin cannot rely on it."""
    return json.dumps({
        "length": int(length),
        "shapes": [list(int(x) for x in s) for s in shapes],
        "dtypes": [str(d).replace("torch.", "") for d in dtypes],
        "fmt": int(fmt.value),
    }, separators=(",", ":")).encode()


def _meta_unpack(meta: bytes):
    d = json.loads(meta.decode())
    shapes = [torch.Size(s) for s in d["shapes"]]
    dtypes = [getattr(torch, n) for n in d["dtypes"]]
    return int(d["length"]), shapes, dtypes, MemoryFormat(int(d["fmt"]))


class DaosGdsBackend(AllocatorBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        dst_device: str = "cuda",
        metadata: Optional[LMCacheMetadata] = None,
        local_cpu_backend=None,
        loop=None,
    ):
        assert dst_device.startswith("cuda"), "DaosGdsBackend needs a CUDA dst_device"
        super().__init__(dst_device=dst_device)
        self.config = config
        self.metadata = metadata
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        from .gpu_store import validate_config, install as install_gpu_store
        self.store_path = str(_cfg(config, "store_path", "host_staged")).lower()
        validate_config(config, self.store_path)
        if self.store_path == "gpu_direct":
            install_gpu_store()
        dev = torch.device(dst_device)
        self.device_id = dev.index if dev.index is not None else torch.cuda.current_device()
        self.dst_device = f"cuda:{self.device_id}"

        pool = _cfg(config, "pool")
        cont = _cfg(config, "container")
        if not pool or not cont:
            raise ValueError("daosgds.pool and daosgds.container are required")
        self.transport = str(
            os.environ.get("DAOSGDS_TRANSPORT", _cfg(config, "transport", "dfs"))
        ).strip().lower()
        if self.transport not in {"dfs", "object"}:
            raise ValueError("daosgds.transport must be 'dfs' or 'object'")
        self.root = str(_cfg(config, "root", v2.V2_PREFIX)).rstrip("/") or v2.V2_PREFIX
        # Object mode has no directory tree.  Prefixing its dkeys provides the
        # equivalent of a private DFS root and prevents this checkout from
        # colliding with another client using the shim's fixed DAOS object.
        self.object_namespace = str(_cfg(config, "object_namespace", ""))
        self.dfs_oclass = int(_cfg(config, "dfs_oclass", 0))
        self.io_workers = int(_cfg(config, "io_workers", 16))
        self.meta_workers = int(_cfg(config, "meta_workers", self.io_workers))
        if self.io_workers <= 0 or self.meta_workers <= 0:
            raise ValueError("daosgds.io_workers and meta_workers must be positive")
        self.gpu_buffer_bytes = int(float(_cfg(config, "gpu_buffer_gb", 6)) * (1 << 30))
        self.store_enabled = bool(_cfg(config, "store", True))

        self._tls = threading.local()
        self._dfs = None
        self._object = None
        if self.transport == "dfs":
            self._dfs = DfsSys(pool=pool, cont=cont, sys=_cfg(config, "sys"))
            self._dfs.mkdir_p(self.root)
        else:
            if _cfg(config, "sys"):
                logger.warning(
                    "daosgds.sys is ignored by object transport; libdaosgdr "
                    "uses the DAOS client environment"
                )
            self._object = DaosObjectStore(
                pool=pool,
                container=cont,
                library_path=_cfg(config, "object_library"),
            )
        # Every pool thread binds the CUDA context up front (initializer): the
        # DAOS client encodes another thread's fetch RPC -- and registers its
        # GPU bulk buffer -- from whichever thread happens to drive progress,
        # so a thread that only ever did stat() can hit the registration path.
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.io_workers, thread_name_prefix="daosgds-io",
            initializer=self._ensure_cuda_ctx)
        # Lookup/stat work is isolated from bulk reads and writes.  Besides
        # avoiding head-of-line blocking, this prevents nested-executor
        # deadlock: the old async contains path occupied an I/O worker while
        # waiting for per-key contains jobs submitted back to that same pool.
        self._meta_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.meta_workers,
            thread_name_prefix="daosgds-meta",
            initializer=self._ensure_cuda_ctx,
        )
        # The target-spreading probe is a DFS/file-layout operation.  Object
        # mode uses one DAOS_OT_MULTI_HASHED object whose dkeys are distributed
        # by DAOS and therefore deliberately skips this DFS-specific warm-up.
        if self.transport == "dfs":
            try:
                n = int(os.environ.get("DAOS_PROBE_CHUNKS", "64"))
                if n > 0:
                    assert self._dfs is not None
                    r = warm_up_all_targets(
                        self._dfs,
                        f"{self.root}/.daos-probe.{os.getpid()}",
                        n,
                        self._pool,
                    )
                    logger.info(
                        "DaosGdsBackend warm-up: %d targets in %.1f ms (total %.1f ms)",
                        r["n"], r["connect_ms"], r["total_ms"],
                    )
            except Exception as e:  # pragma: no cover
                logger.warning("DaosGdsBackend warm-up failed: %s", e)

        self.memory_allocator = self.initialize_allocator(config, metadata)
        self._known: set = set()            # keys seen on DAOS (stat or our own put)
        self._known_lock = threading.Lock()
        self._put_lock = threading.Lock()
        self._put_tasks: set = set()
        self.stats = {"put": 0, "put_bytes": 0, "get": 0, "get_bytes": 0, "miss": 0,
                      "alloc_fail": 0, "get_ms": 0.0, "put_ms": 0.0}
        logger.info("DaosGdsBackend store_path=%s", self.store_path)
        logger.info(
            "DaosGdsBackend: transport=%s pool=%s cont=%s root=%s "
            "object_namespace=%s device=%s "
            "gpu_buffer=%.1f GiB io_workers=%d meta_workers=%d",
            self.transport, pool, cont, self.root, self.object_namespace,
            self.dst_device,
            self.gpu_buffer_bytes / (1 << 30), self.io_workers, self.meta_workers,
        )
        global _LAST_BACKEND
        _LAST_BACKEND = self
        _install_multi_prefetch_serializer()
        _install_consumed_prefetch_event_patch()
        self._staging_trace = None
        if os.environ.get("DAOS_GDS_STAGING_TRACE"):
            from .staging_trace import install
            self._staging_trace = install(self)

    # -- allocator backend ----------------------------------------------------
    def initialize_allocator(self, config, metadata=None) -> MemoryAllocatorInterface:
        return GPUMemoryAllocator(self.gpu_buffer_bytes, device=self.dst_device, align_bytes=4096)

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        return self.memory_allocator

    def get_allocator_backend(self):
        return self

    def allocate(self, shapes, dtypes, fmt: MemoryFormat = MemoryFormat.KV_2LTD,
                 eviction: bool = True, busy_loop: bool = True) -> Optional[MemoryObj]:
        obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
        if obj is None:
            self.stats["alloc_fail"] += 1
        return obj

    def batched_allocate(self, shapes, dtypes, batch_size: int,
                         fmt: MemoryFormat = MemoryFormat.KV_2LTD,
                         eviction: bool = True, busy_loop: bool = True) -> Optional[list]:
        objs = self.memory_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)
        if objs is None:
            self.stats["alloc_fail"] += 1
        return objs

    def calculate_chunk_budget(self) -> int:
        md = self.metadata
        try:
            n = 1
            for d in md.kv_shape:
                n *= int(d)
            chunk_bytes = n * torch.tensor([], dtype=md.kv_dtype).element_size()
            return max(1, self.gpu_buffer_bytes // chunk_bytes)
        except Exception:
            return 16

    _cu = None          # libcuda handle (process-wide)
    _cu_ctx = None      # retained primary context for device_id

    def _ensure_cuda_ctx(self) -> None:
        """libfabric's CUDA HMEM path calls cuMemGetAddressRange / dma-buf export
        on the *calling* thread; a pool thread that never touched CUDA has no
        current driver context and fails with CUDA_ERROR_INVALID_CONTEXT (seen
        as DER_HG_FATAL on the bulk, and it killed the engine). torch's runtime
        calls do not reliably bind the driver context on a fresh thread (worked
        with 16 workers, failed with 24+), so bind it explicitly with the driver
        API: retain the device's primary context once, cuCtxSetCurrent per thread."""
        if getattr(self._tls, "ctx", False):
            return
        import ctypes
        cls = type(self)
        if cls._cu is None:
            cu = ctypes.CDLL("libcuda.so.1")
            rc = cu.cuInit(0)
            if rc != 0:
                raise RuntimeError(f"cuInit failed: {rc}")
            ctx = ctypes.c_void_p()
            rc = cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), ctypes.c_int(self.device_id))
            if rc != 0:
                raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {rc}")
            cls._cu, cls._cu_ctx = cu, ctx
        rc = cls._cu.cuCtxSetCurrent(cls._cu_ctx)
        if rc != 0:
            raise RuntimeError(f"cuCtxSetCurrent failed: {rc}")
        self._tls.ctx = True

    # -- paths ------------------------------------------------------------------
    def _path(self, key: CacheEngineKey) -> str:
        p = v2.key_to_path(key)               # "/v2/<sha>"
        if self.root != v2.V2_PREFIX:
            p = self.root + p[len(v2.V2_PREFIX):]
        return p

    def _object_key(self, key: CacheEngineKey) -> str:
        """Preserve the DFS-bypass prototype's dkey mapping.

        The object transport stores all entries under one fixed multi-hashed
        DAOS object, so the full reversible LMCache key is the dkey; it does
        not need the DFS backend's SHA/path mapping.
        """
        return f"{self.object_namespace}{key.to_string()}"

    # -- lookups ----------------------------------------------------------------
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        self._ensure_cuda_ctx()       # stat() may progress another thread's GPU bulk registration
        with self._known_lock:
            if key in self._known:
                return True
        try:
            if self.transport == "dfs":
                assert self._dfs is not None
                present = self._dfs.stat_size(self._path(key)) is not None
            else:
                assert self._object is not None
                present = self._object.stat(self._object_key(key)) is not None
        except (DaosError, DaosObjectError, OSError, ValueError):
            present = False
        if present:
            with self._known_lock:
                self._known.add(key)
        return present

    def batched_contains(self, keys: List[CacheEngineKey], pin: bool = False) -> int:
        """Number of leading keys present (LMCache's prefix-hit contract: the
        storage manager slices ``keys[:n]`` with the return value)."""
        hits = list(self._meta_pool.map(lambda k: self.contains(k, pin), keys))
        n = 0
        for h in hits:
            if not h:
                break
            n += 1
        return n

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._put_lock:
            return key in self._put_tasks

    # -- store ------------------------------------------------------------------
    def batched_submit_put_task(self, keys: Sequence[CacheEngineKey], memory_objs: List[MemoryObj],
                                transfer_spec: Any = None,
                                on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None
                                ) -> Union[List[Future], None]:
        if not self.store_enabled:
            return None
        futs = []
        for key, obj in zip(keys, memory_objs):
            with self._put_lock:
                if key in self._put_tasks:
                    continue
                self._put_tasks.add(key)
            obj.ref_count_up()                # the manager drops its ref right after submit
            try:
                futs.append(self._pool.submit(self._put_one, key, obj, on_complete_callback))
            except BaseException:
                obj.ref_count_down()
                with self._put_lock:
                    self._put_tasks.discard(key)
                raise
        return futs

    def _put_one(self, key: CacheEngineKey, obj: MemoryObj, cb) -> None:
        t0 = time.perf_counter()
        try:
            self._ensure_cuda_ctx()
            if self.contains(key):
                return
            tensor = obj.tensor
            assert tensor is not None and tensor.is_cuda, "put object must be a GPU MemoryObj"
            n = obj.get_size()
            meta = _meta_pack(n, obj.get_shapes(), obj.get_dtypes(), obj.metadata.fmt)
            # The manager's copy stream is synchronized before submit, but an
            # explicit device sync keeps both transports safe when the backend
            # is driven directly by a test or another plugin caller.
            torch.cuda.synchronize(self.device_id)
            if self.transport == "dfs":
                assert self._dfs is not None
                final = self._path(key)
                tmp = v2.temp_path(final, uuid.uuid4().hex[:8])
                h = self._dfs.open_rdwr_create(tmp, oclass=getattr(self, "dfs_oclass", 0))
                try:
                    page = v2.pack_header(meta, n)
                    import ctypes
                    buf = ctypes.create_string_buffer(page, len(page))
                    self._dfs.write_obj_from(h, 0, len(page), buf)
                    wrote = self._dfs.write_gpu_from(
                        h, v2.payload_offset(), n, tensor.data_ptr(), self.device_id
                    )
                finally:
                    self._dfs.close_obj(h)
                if wrote != n:
                    raise IOError(f"short GPU write {wrote}/{n}")
                parent = self._dfs.lookup(self.root)
                try:
                    self._dfs.move(
                        parent, tmp.rsplit("/", 1)[1],
                        parent, final.rsplit("/", 1)[1],
                    )
                finally:
                    self._dfs.release(parent)
            else:
                assert self._object is not None
                self._object.put(
                    self._object_key(key), tensor.data_ptr(), n, meta, self.device_id
                )
            with self._known_lock:
                self._known.add(key)
            self.stats["put"] += 1
            self.stats["put_bytes"] += n
            self.stats["put_ms"] += (time.perf_counter() - t0) * 1e3
            if cb is not None:
                try:
                    cb(key)
                except Exception as e:  # pragma: no cover
                    logger.warning("put callback failed: %s", e)
        except Exception as e:
            logger.exception("DaosGdsBackend put %s failed: %r", key.to_string(), e)
        finally:
            self._release_memory_obj(obj)
            with self._put_lock:
                self._put_tasks.discard(key)

    # -- retrieve ---------------------------------------------------------------
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        t0 = time.perf_counter()
        self._ensure_cuda_ctx()
        if self.transport == "object":
            return self._get_object_blocking(key, t0)

        assert self._dfs is not None
        path = self._path(key)
        try:
            h = self._dfs.open_rdonly(path)
        except DaosError as e:
            if getattr(e, "rc", None) == 2:
                self.stats["miss"] += 1
                return None
            raise
        obj = None
        try:
            try:
                hdr = v2.parse_header(self._dfs.read_obj(h, 0, v2.HEADER_SIZE))
            except v2.BadHeader as e:
                logger.warning("DaosGdsBackend: %s -> miss (%s)", path, e)
                self.stats["miss"] += 1
                return None
            _length, shapes, dtypes, fmt = _meta_unpack(hdr.meta)
            obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
            if obj is None or obj.tensor is None:
                self.stats["alloc_fail"] += 1
                logger.warning("DaosGdsBackend: GPU buffer full, %s served as miss", path)
                if obj is not None:
                    self._release_memory_obj(obj)
                return None
            n = hdr.payload_len
            if n != obj.get_size():
                logger.warning("DaosGdsBackend: %s payload %d != object %d -> miss", path, n, obj.get_size())
                self._release_memory_obj(obj)
                return None
            got = self._dfs.read_gpu_into(h, v2.payload_offset(), n, obj.tensor.data_ptr(), self.device_id)
            if got != n:
                logger.warning("DaosGdsBackend: short GPU read %d/%d on %s -> miss", got, n, path)
                self._release_memory_obj(obj)
                return None
            self.stats["get"] += 1
            self.stats["get_bytes"] += n
            self.stats["get_ms"] += (time.perf_counter() - t0) * 1e3
            with self._known_lock:
                self._known.add(key)
            return obj
        except Exception as e:
            logger.exception("DaosGdsBackend get %s failed: %r", path, e)
            if obj is not None:
                self._release_memory_obj(obj)
            return None
        finally:
            self._dfs.close_obj(h)

    def _get_object_blocking(
        self, key: CacheEngineKey, started: float
    ) -> Optional[MemoryObj]:
        """Object-API equivalent of the DFS get path.

        Metadata is fetched through the host ``meta`` akey first so the GPU
        staging allocation can be sized, then the ``kv`` akey is fetched
        directly into that allocation with ``daos_obj_fetch_gpu``.
        """
        assert self._object is not None
        obj = None
        object_key = self._object_key(key)
        try:
            meta = self._object.stat(object_key)
            if meta is None:
                self.stats["miss"] += 1
                return None
            _length, shapes, dtypes, fmt = _meta_unpack(meta)
            obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
            if obj is None or obj.tensor is None:
                self.stats["alloc_fail"] += 1
                logger.warning(
                    "DaosGdsBackend: GPU buffer full, object key %s served as miss",
                    object_key,
                )
                if obj is not None:
                    self._release_memory_obj(obj)
                return None
            n = obj.get_size()
            if n != _length:
                logger.warning(
                    "DaosGdsBackend: object metadata payload %d != allocation %d -> miss",
                    _length, n,
                )
                self._release_memory_obj(obj)
                return None
            self._object.get(object_key, obj.tensor.data_ptr(), n, self.device_id)
            self.stats["get"] += 1
            self.stats["get_bytes"] += n
            self.stats["get_ms"] += (time.perf_counter() - started) * 1e3
            with self._known_lock:
                self._known.add(key)
            return obj
        except (DaosObjectError, OSError, ValueError) as e:
            logger.warning("DaosGdsBackend object get %s failed: %s", object_key, e)
            if obj is not None:
                self._release_memory_obj(obj)
            return None

    def _release_memory_obj(self, obj: MemoryObj) -> None:
        """Release a GPU object while serializing access to its arena."""
        lock = getattr(self.memory_allocator, "device_mem_lock", None)
        if lock is None:
            obj.ref_count_down()
            return
        with lock:
            obj.ref_count_down()

    def batched_get_blocking(self, keys: List[CacheEngineKey]) -> List[Optional[MemoryObj]]:
        t0 = time.perf_counter()
        res = list(self._pool.map(self.get_blocking, keys))
        # LMCache consumes hits as a prefix: after the first miss, later objects
        # would be unusable, so release them and report None from there on.
        cut = next((i for i, o in enumerate(res) if o is None), None)
        if cut is not None:
            for o in res[cut + 1:]:
                if o is not None:
                    self._release_memory_obj(o)
            res = res[:cut] + [None] * (len(res) - cut)
        nb = sum(o.get_size() for o in res if o is not None)
        dt = time.perf_counter() - t0
        if nb:
            logger.info("DaosGdsBackend batched_get: %d/%d objects, %.1f MiB in %.1f ms (%.2f GB/s, GPU-direct)",
                        sum(o is not None for o in res), len(keys), nb / 2**20, dt * 1e3, nb / dt / 1e9)
        return res

    def get_non_blocking(self, key: CacheEngineKey, location: Optional[str] = None) -> Optional[Future]:
        return None

    # -- async loading (enable_async_loading: True) ----------------------------
    # The storage manager runs these on its event loop at *lookup* time, before
    # the request is scheduled, and hands the objects to retrieve() through the
    # event manager (not through LocalCPUBackend). That is what lets request
    # N+1's DAOS->GPU reads overlap request N's prefill -- the overlap the MP
    # server gets from its own load tasks, without a separate process.
    async def batched_async_contains(self, lookup_id: str, keys: List[CacheEngineKey],
                                     pin: bool = False) -> int:
        loop = asyncio.get_running_loop()
        hits = await asyncio.gather(
            *(loop.run_in_executor(self._meta_pool, self.contains, k, pin) for k in keys)
        )
        n = 0
        for hit in hits:
            if not hit:
                break
            n += 1
        return n

    async def batched_get_non_blocking(self, lookup_id: str, keys: List[CacheEngineKey],
                                       transfer_spec: Any = None) -> List[MemoryObj]:
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        res = await asyncio.gather(*(loop.run_in_executor(self._pool, self.get_blocking, k) for k in keys))
        # prefix semantics: stop at the first miss, release anything after it
        out: List[MemoryObj] = []
        failed = False
        for o in res:
            if o is None or failed:
                failed = True
                if o is not None:
                    self._release_memory_obj(o)
                continue
            out.append(o)
        nb = sum(o.get_size() for o in out)
        dt = time.perf_counter() - t0
        if nb:
            logger.info("DaosGdsBackend prefetch[%s]: %d/%d objects, %.1f MiB in %.1f ms (%.2f GB/s, GPU-direct)",
                        lookup_id[-8:], len(out), len(keys), nb / 2**20, dt * 1e3, nb / dt / 1e9)
        return out

    # -- misc -------------------------------------------------------------------
    def pin(self, key: CacheEngineKey) -> bool:
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        with self._known_lock:
            self._known.discard(key)
        try:
            if self.transport == "dfs":
                assert self._dfs is not None
                return self._dfs.remove(self._path(key))
            assert self._object is not None
            return self._object.remove(self._object_key(key))
        except (DaosError, DaosObjectError, OSError, ValueError):
            return False

    def batched_remove(self, keys: List[CacheEngineKey], force: bool = True) -> int:
        return sum(1 for k in keys if self.remove(k, force))

    def touch_cache(self) -> None:
        pass

    def cancel_request(self, req_id: str) -> None:
        pass

    def close(self) -> None:
        logger.info(
            "DaosGdsBackend transport=%s stats: %s", self.transport, self.stats
        )
        self._pool.shutdown(wait=True)
        self._meta_pool.shutdown(wait=True)
        try:
            if self._dfs is not None:
                self._dfs.close()
            if self._object is not None:
                self._object.close()
        except Exception:  # pragma: no cover
            pass

    def __str__(self) -> str:
        return "DaosGdsBackend"
