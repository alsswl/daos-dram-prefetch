"""
daosgdr_backend.py

DaosGdrBackend -- an LMCache v1 storage backend that stores/loads KV cache
tensors directly between GPU memory and DAOS via GPUDirect RDMA
(daos_obj_update_gpu/daos_obj_fetch_gpu), bypassing dfuse/POSIX entirely.

This is a standalone module OUTSIDE the LMCache package (nothing under
lmcache/ is modified -- only imported from). It is loaded at runtime by
LMCache's own storage_plugin_launcher (lmcache/v1/storage_backend/__init__.py,
:40-108) via importlib, driven by config.storage_plugins +
config.extra_config["storage_plugin.<name>.module_path"/".class_name"].
See daosgdr_lmcache_config.yaml alongside this file for the config shape,
and run_with_daosgdr.sh for the environment (PYTHONPATH, LD_LIBRARY_PATH,
D_GPU_DIRECT) this module and libdaosgdr.so need at runtime.

Chain of verification this design rests on (each step in this same working
directory): gdr_test.c (raw daos_obj_update_gpu/fetch_gpu round-trip) ->
mixed_test.c (mem_attrs[i]<->sgls[i] index mapping) -> libdaosgdr.c (the C
shim, ~/libdaosgdr.so) -> shim_test.c (shim's public API end-to-end,
including the "not found" and kv-size-query rc findings) -> daosgdr_poc.py
(ctypes binding + pack_meta/unpack_meta pattern, reused below almost
verbatim, now with MemoryFormat added).

Key architectural decisions, and the exact LMCache source lines they were
verified against in this session:

  - class DaosGdrBackend(StoragePluginInterface, AllocatorBackendInterface):
    multiple inheritance. StoragePluginInterface.__init__
    (abstract_backend.py:452) calls super().__init__(dst_device=...)
    cooperatively, so the MRO (DaosGdrBackend -> StoragePluginInterface ->
    AllocatorBackendInterface -> StorageBackendInterface -> object) reaches
    StorageBackendInterface.__init__ correctly even though
    AllocatorBackendInterface itself defines no __init__. No isinstance
    check anywhere in storage_plugin_launcher or CreateStorageBackends
    gates this choice (verified: only two isinstance(...,
    AllocatorBackendInterface) checks exist in the whole package,
    storage_manager.py:324 and :1159, and neither excludes a
    plugin-loaded backend).

  - get_allocator_backend() returns self. storage_manager.py's batched put
    path (:405-429) calls backend.get_allocator_backend() for every
    registered backend and, if that allocator hasn't already produced an
    object for this batch, allocates a fresh one via
    allocator_backend.allocate(...) and GPU-copies the source tensor into
    it on self.internal_copy_stream, then calls
    stream.synchronize() (:116-117 in allocate_and_copy_objects) --
    *before* batched_submit_put_task is ever called. This means the GPU
    tensor batched_submit_put_task() receives already lives in a buffer
    this backend's own allocator owns, and the stream that wrote it has
    already been synchronized by the framework. No extra
    torch.cuda.synchronize() is added in this file's put path for that
    reason (it would be redundant with what storage_manager already did).

  - initialize_allocator() returns a plain GPUMemoryAllocator
    (memory_allocators/gpu_memory_allocator.py:69-88), not a
    CuFileMemoryAllocator subclass. GdsBackend subclasses
    CuFileMemoryAllocator only to call cuFileBufRegister() for cuFile's
    own buffer-registration requirement (memory_allocators/
    cu_file_memory_allocator.py:16-28); DAOS's GPUDirect path has no such
    registration call in gdr_test.c (a bare cudaMalloc'd pointer round-
    tripped successfully), so no equivalent subclass is needed here.

Concurrency (v2 -- put is asynchronous, get is not):

  - batched_submit_put_task() no longer blocks. Each key's DAOS write is
    handed to a ThreadPoolExecutor and the call returns as soon as the
    tasks are queued. Same shape as GdsBackend, which hops to a thread via
    asyncio.to_thread(self._save_gds) (gds_backend.py:661); the difference
    is that this backend has no asyncio loop of its own to ride on, so it
    drives a plain ThreadPoolExecutor directly.
    Motivation, measured on this machine with DAOSGDR_TIMING=1: 98% of a
    put's wall time is inside daos_obj_update_gpu itself (3MB -> 2.08ms,
    12MB -> 4.31ms), while every Python-side segment combined (pack_meta +
    ctypes marshalling + ref_count/lock bookkeeping) is 0.04-0.07ms and C
    struct assembly rounds to 0.000ms. There is nothing left to optimize
    in this file's own code -- only the DAOS call can be moved off vLLM's
    critical path.

  - All worker threads share the SINGLE ctx (hence the single
    daos_handle_t oh) created in __init__. Deliberate, and measured before
    being relied on (~/thread_safety_test.c): with one shared ctx, 4
    threads x 100 iters (400 ops) and 8 threads x 100 iters (800 ops) of
    put -> stat -> get -> verify completed with zero rc failures, zero
    meta mismatches and zero data-verification failures, and throughput
    scaled 190.6 -> 360.2 ops/s from 4 to 8 threads (1.89x, near-linear),
    i.e. DAOS is not serializing internally on the shared handle. Do NOT
    "fix" this by giving each worker its own ctx -- that would multiply
    pool/container connections for no measured benefit.
    (The public headers say nothing about daos_obj_* thread safety:
    daos_obj.h:592-615 and the update/fetch doc blocks are silent. The one
    documented multi-thread restriction, daos_api.h:58-60, is about
    transaction handles, and this backend always passes DAOS_TX_NONE.
    Hence the empirical test.)

  - CUDA context in worker threads: NOT bound explicitly here. If DAOS's
    RDMA registration of the GPU buffer ever fails from a worker thread
    with CUDA_ERROR_INVALID_CONTEXT or DER_HG_FATAL, the cause is that the
    worker thread has no current CUDA *driver* context -- a torch
    runtime-API call from that thread does not reliably establish one for
    a third-party library like libdaos. The fix would be a
    ThreadPoolExecutor(initializer=...) that, via ctypes on libcuda.so,
    calls cuDevicePrimaryCtxRetain(&pctx, dev) once for the device and
    cuCtxSetCurrent(pctx) in each worker thread. Deliberately not added
    pre-emptively: untested code on a path that may never be needed.

  - Releasing a MemoryObj from a worker thread needs the allocator's lock,
    which LMCache's own ref_count_down() path does not take. See
    _release_memory_obj() -- this is the one correctness change v2 needed
    beyond "call the same code on another thread".

  - get_blocking() stays synchronous -- StorageBackendInterface.get_blocking
    (abstract_backend.py:119) is a synchronous API and its caller
    (storage_manager.py:441-459) needs the MemoryObj as a return value, so
    there is nothing to overlap it with here.
"""

# Standard
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, Union
import ctypes
import functools
import os
import struct
import threading
import time

# Third Party
import torch

# First Party (import-only -- nothing under lmcache/ is modified)
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StoragePluginInterface,
)

logger = init_logger(__name__)


# =============================================================================
# SECTION: timing instrumentation (measurement only, gated by DAOSGDR_TIMING=1)
#
# Added to chase down why the DaosGdrBackend put path was observed running
# ~2x slower than daosgdr_poc.py's raw ctypes calls despite going through
# the same libdaosgdr.so -- this instruments _submit_put_task_sync and
# get_blocking below to find which segment (pack_meta, create_string_buffer,
# the ctypes call itself, ref_count/lock bookkeeping, ...) actually accounts
# for the difference. No optimization here, only measurement.
#
# Read once at import time (not a function call re-checked on every put/get)
# so the steady-state cost of having this instrumented at all, when
# DAOSGDR_TIMING is unset, is a single `if False:`-equivalent boolean check
# per segment -- no os.environ.get(), no time.perf_counter() calls.
# =============================================================================

_TIMING_ENABLED = os.environ.get("DAOSGDR_TIMING") == "1"


class _Timer:
    """Collects (label, perf_counter()) marks and logs the deltas between
    consecutive marks as one line when done. Only ever instantiated when
    _TIMING_ENABLED is True -- callers guard `_Timer() if _TIMING_ENABLED
    else None` and every other use behind `if timer is not None`, so this
    class's own overhead (attribute access, list append) never happens on
    the normal (disabled) path.

    Also stashes the last segment breakdown as a plain dict on the owning
    backend instance (self._last_put_timing_ms / self._last_get_timing_ms)
    so a measurement script can read exact numbers back directly instead of
    parsing logger output -- this is a Python-side convenience only, not a
    change to any function signature (the rule against out-params was about
    libdaosgdr.c's C API, which is unchanged here).

    Since v2 (async put), a put's timer is CONSTRUCTED IN THE SUBMITTING
    THREAD and its first mark is taken in the worker thread, so the first
    segment ("queued") is exactly the time the task spent waiting for a free
    worker -- the number that tells you whether the pool is too small.
    "total_ms" is therefore submit-to-completion latency, not the DAOS call
    duration; compare it against "submit_ms" (measured separately by the
    caller) to see how much of the put actually left the critical path.
    Because several workers finish concurrently, _last_put_timing_ms is only
    "the most recently completed put"; use _put_timings (a bounded deque of
    every recorded breakdown) for aggregate statistics.
    """

    __slots__ = ("marks",)

    def __init__(self):
        self.marks = [("start", time.perf_counter())]

    def mark(self, label: str) -> None:
        self.marks.append((label, time.perf_counter()))

    def as_dict(self) -> dict:
        d = {}
        for (l0, t0), (l1, t1) in zip(self.marks, self.marks[1:]):
            d[f"{l1}_ms"] = (t1 - t0) * 1000.0
        d["total_ms"] = (self.marks[-1][1] - self.marks[0][1]) * 1000.0
        return d

    def log(self, logger_, prefix: str) -> dict:
        d = self.as_dict()
        segs = " ".join(f"{k}={v:.3f}" for k, v in d.items())
        logger_.info("[DAOSGDR_TIMING] %s: %s", prefix, segs)
        return d


# =============================================================================
# SECTION: ctypes binding to libdaosgdr.so
#
# Kept in its own clearly-marked section so it can be lifted into a separate
# module (e.g. `daosgdr/_ctypes_binding.py`) later without touching the
# backend logic below. Loaded lazily (not at import time) and cached, so
# importing this module doesn't require libdaosgdr.so to be resolvable in
# environments that merely import it without instantiating the backend
# (e.g. the scheduler role, or a plain `import` in a test).
# =============================================================================

_DEFAULT_LIB_PATH = os.path.expanduser("~/libdaosgdr.so")


@functools.lru_cache(maxsize=1)
def _get_lib() -> ctypes.CDLL:
    """Load libdaosgdr.so and bind its full API with explicit argtypes/restype.

    The library path is never hardcoded as the only option: it defaults to
    ~/libdaosgdr.so but can be overridden via the DAOSGDR_LIB environment
    variable, so a future packaged install can point this at a
    site-packages-relative or system path without editing this file.
    """
    lib_path = os.environ.get("DAOSGDR_LIB", _DEFAULT_LIB_PATH)
    lib = ctypes.CDLL(lib_path)

    lib.daosgdr_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.daosgdr_init.restype = ctypes.c_void_p

    lib.daosgdr_put.argtypes = [
        ctypes.c_void_p,  # ctx
        ctypes.c_char_p,  # key
        ctypes.c_void_p,  # gpu_ptr
        ctypes.c_size_t,  # size
        ctypes.c_void_p,  # meta
        ctypes.c_size_t,  # meta_len
    ]
    lib.daosgdr_put.restype = ctypes.c_int

    lib.daosgdr_stat.argtypes = [
        ctypes.c_void_p,  # ctx
        ctypes.c_char_p,  # key
        ctypes.c_void_p,  # meta (out buffer)
        ctypes.POINTER(ctypes.c_size_t),  # meta_len (in: capacity, out: actual)
    ]
    lib.daosgdr_stat.restype = ctypes.c_int

    lib.daosgdr_get.argtypes = [
        ctypes.c_void_p,  # ctx
        ctypes.c_char_p,  # key
        ctypes.c_void_p,  # gpu_ptr
        ctypes.c_size_t,  # size
    ]
    lib.daosgdr_get.restype = ctypes.c_int

    lib.daosgdr_remove.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.daosgdr_remove.restype = ctypes.c_int

    lib.daosgdr_fini.argtypes = [ctypes.c_void_p]
    lib.daosgdr_fini.restype = None

    logger.info("Loaded libdaosgdr from %s", lib_path)
    return lib


# =============================================================================
# SECTION: meta (opaque metadata) encode/decode
#
# The shim (libdaosgdr.c) never interprets the "meta" akey's bytes -- it is
# pure opaque payload as far as DAOS/the C shim are concerned (see
# libdaosgdr.c's daosgdr_stat comment: a data-less DAOS query for the "kv"
# akey's size was tried and failed with -DER_REC2BIG, so the kv byte count
# has to live here instead). This is the same pack_meta/unpack_meta shape
# proven in daosgdr_poc.py, extended with `fmt` (MemoryFormat) because
# get_blocking() below must pass fmt through to self.allocate(shape, dtype,
# fmt) to reconstruct the right kind of MemoryObj.
# =============================================================================

_META_MAGIC = 0x444D4C32  # "2LMD" -- bumped from daosgdr_poc.py's 0x...31 since the wire format changed (fmt byte added)
_META_HEADER = struct.Struct("<IIQBB")  # magic(u32), dtype_code(u32), nbytes(u64), ndims(u8), fmt_code(u8)
_META_MAX_DIMS = 8
META_BUF_CAP = _META_HEADER.size + _META_MAX_DIMS * 8 + 32

_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float64: 3,
    torch.uint8: 4,
    torch.int8: 5,
    torch.int16: 6,
    torch.int32: 7,
    torch.int64: 8,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


def pack_meta(tensor: torch.Tensor, fmt: MemoryFormat) -> bytes:
    """Encode shape/dtype/fmt/nbytes for `tensor` into an opaque byte blob.

    Requires a contiguous tensor -- see daosgdr_poc.py's pack_meta for why
    (data_ptr() + nbytes is only a valid description of the tensor's bytes
    for a contiguous layout; a non-contiguous view would silently RDMA the
    wrong bytes with no error from DAOS).
    """
    if not tensor.is_contiguous():
        raise ValueError(
            "pack_meta: tensor must be contiguous -- data_ptr()+nbytes is "
            "only meaningful for a contiguous layout"
        )
    if tensor.dtype not in _DTYPE_TO_CODE:
        raise ValueError(f"pack_meta: unsupported dtype {tensor.dtype}")
    if len(tensor.shape) > _META_MAX_DIMS:
        raise ValueError(
            f"pack_meta: {len(tensor.shape)} dims exceeds _META_MAX_DIMS={_META_MAX_DIMS}"
        )

    dtype_code = _DTYPE_TO_CODE[tensor.dtype]
    nbytes = tensor.numel() * tensor.element_size()
    shape = tuple(tensor.shape)

    header = _META_HEADER.pack(_META_MAGIC, dtype_code, nbytes, len(shape), fmt.value)
    shape_bytes = struct.pack(f"<{len(shape)}Q", *shape)
    return header + shape_bytes


def unpack_meta(data: bytes):
    """Decode pack_meta()'s output back into (shape, dtype, fmt, nbytes)."""
    if len(data) < _META_HEADER.size:
        raise ValueError(f"unpack_meta: buffer too short ({len(data)} bytes)")

    magic, dtype_code, nbytes, ndims, fmt_code = _META_HEADER.unpack_from(data, 0)
    if magic != _META_MAGIC:
        raise ValueError(f"unpack_meta: bad magic {magic:#x}, corrupt/foreign meta blob")
    if dtype_code not in _CODE_TO_DTYPE:
        raise ValueError(f"unpack_meta: unknown dtype code {dtype_code}")

    offset = _META_HEADER.size
    needed = offset + ndims * 8
    if len(data) < needed:
        raise ValueError(
            f"unpack_meta: buffer too short for {ndims} shape dims "
            f"(need {needed}, got {len(data)})"
        )
    shape = struct.unpack_from(f"<{ndims}Q", data, offset)

    return torch.Size(shape), _CODE_TO_DTYPE[dtype_code], MemoryFormat(fmt_code), nbytes


# =============================================================================
# SECTION: DaosGdrBackend
# =============================================================================


class DaosGdrBackend(StoragePluginInterface, AllocatorBackendInterface):
    """LMCache storage backend: KV cache tensors round-trip GPU<->DAOS
    directly via GPUDirect RDMA, with no dfuse/POSIX and no host bounce
    buffer.

    Loaded as a plugin -- see module docstring and
    daosgdr_lmcache_config.yaml for how config.storage_plugins /
    config.extra_config wire this class in.
    """

    def __init__(
        self,
        dst_device: str = torch_device_type,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheMetadata] = None,
        local_cpu_backend: Optional[Any] = None,
        loop: Optional[Any] = None,
    ):
        assert dst_device.startswith("cuda"), (
            f"DaosGdrBackend requires a CUDA device, got {dst_device!r}. "
            "This backend performs GPUDirect RDMA and has no CPU fallback "
            "path (mirrors GdsBackend's own constraint, gds_backend.py:245)."
        )
        # StoragePluginInterface.__init__ (abstract_backend.py:429-456) sets
        # self.config/self.metadata/self.local_cpu_backend/self.loop and
        # cooperatively chains via super().__init__(dst_device=dst_device)
        # to StorageBackendInterface.__init__ (sets self.dst_device) through
        # AllocatorBackendInterface in the MRO -- see module docstring.
        super().__init__(
            dst_device=dst_device,
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
        )

        if config is None:
            raise ValueError("DaosGdrBackend requires a config")

        pool = config.get_extra_config_value("daos_gdr_pool")
        cont = config.get_extra_config_value("daos_gdr_cont")
        if not pool or not cont:
            raise ValueError(
                "DaosGdrBackend requires 'daos_gdr_pool' and 'daos_gdr_cont' "
                "to be set under extra_config (see daosgdr_lmcache_config.yaml)"
            )

        self._lib = _get_lib()
        self._ctx = self._lib.daosgdr_init(pool.encode("utf-8"), cont.encode("utf-8"))
        if not self._ctx:
            raise RuntimeError(
                f"daosgdr_init(pool={pool!r}, cont={cont!r}) failed -- see "
                "the [libdaosgdr] stderr lines above for the underlying DAOS rc"
            )
        logger.info("DaosGdrBackend connected: pool=%s cont=%s", pool, cont)

        # Same generic (not DAOS-specific) retry knobs GdsBackend reads,
        # gds_backend.py:345-350 -- reused here rather than inventing
        # daos_gdr-prefixed duplicates for the same concept.
        extra_config = config.extra_config or {}
        self.max_alloc_attempts = extra_config.get("max_alloc_attempts", 10)
        self.alloc_attempt_delay_secs = extra_config.get(
            "allocation_attempt_delay_secs", 0.1
        )

        self.memory_allocator = self.initialize_allocator(config, metadata)

        self._put_lock = threading.Lock()
        self._put_tasks: set = set()
        self._closed = False

        # Async put (v2). Worker count is configurable because the right
        # value is a memory tradeoff, not a throughput one: every in-flight
        # put pins a GPU buffer from self.memory_allocator for the whole
        # duration of its DAOS write, so N workers means up to N chunks of
        # the (small, A400 = 4GB) GPU arena unavailable for reuse.
        # ~/thread_safety_test.c showed near-linear scaling to 8 threads
        # (190.6 -> 360.2 ops/s), so 8+ is safe from DAOS's side; the
        # default stays conservative at 4 for the buffer-pressure reason.
        #
        # NOTE (CUDA driver context): no `initializer=` is passed. See the
        # module docstring for the exact cuDevicePrimaryCtxRetain /
        # cuCtxSetCurrent recipe to add here IF a worker thread ever fails
        # with CUDA_ERROR_INVALID_CONTEXT or DER_HG_FATAL inside
        # daos_obj_update_gpu. Not added pre-emptively.
        self._io_workers = int(
            config.get_extra_config_value("daos_gdr_io_workers", 4)
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._io_workers,
            thread_name_prefix="daosgdr-io",
        )
        logger.info(
            "DaosGdrBackend put I/O pool: %d worker thread(s), sharing one "
            "DAOS ctx/object handle",
            self._io_workers,
        )

        # Timing bookkeeping is written from worker threads, so it needs its
        # own lock (self._put_lock guards _put_tasks/_closed and is taken on
        # the critical path -- keep the two uncontended by each other).
        self._timing_lock = threading.Lock()
        self._last_put_timing_ms: Optional[dict] = None
        self._last_get_timing_ms: Optional[dict] = None
        # Bounded: with DAOSGDR_TIMING=1 a long vLLM run would otherwise
        # accumulate one dict per put forever.
        self._put_timings: deque = deque(maxlen=8192)

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def _key_bytes(key: CacheEngineKey) -> bytes:
        return key.to_string().encode("utf-8")

    # -------------------------------------------------------------------
    # AllocatorBackendInterface
    # -------------------------------------------------------------------

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> GPUMemoryAllocator:
        size_mb = config.get_extra_config_value("daos_gdr_buffer_size_mb", 128)
        return GPUMemoryAllocator(int(size_mb) * 1024**2, self.dst_device)

    def get_memory_allocator(self) -> GPUMemoryAllocator:
        return self.memory_allocator

    def allocate(
        self,
        shapes,
        dtypes,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """Mirrors GdsBackend.allocate (gds_backend.py:1128-1173): retry loop
        around the underlying GPUMemoryAllocator, since this backend has no
        eviction policy of its own."""
        if eviction:
            logger.warning("DaosGdrBackend does not support eviction")

        max_attempts = self.max_alloc_attempts if busy_loop else 1
        num_attempts = 0
        while True:
            memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
            if memory_obj is not None:
                return memory_obj
            num_attempts += 1
            if num_attempts < max_attempts:
                if self.alloc_attempt_delay_secs > 0:
                    time.sleep(self.alloc_attempt_delay_secs)
            else:
                break

        logger.warning(
            "DaosGdrBackend allocation failed after %d attempt(s)", num_attempts
        )
        return None

    def batched_allocate(
        self,
        shapes,
        dtypes,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list]:
        if eviction:
            logger.warning("DaosGdrBackend does not support eviction")

        max_attempts = self.max_alloc_attempts if busy_loop else 1
        num_attempts = 0
        while True:
            memory_objs = self.memory_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
            if memory_objs is not None:
                return memory_objs
            num_attempts += 1
            if num_attempts < max_attempts:
                if self.alloc_attempt_delay_secs > 0:
                    time.sleep(self.alloc_attempt_delay_secs)
            else:
                break

        logger.warning(
            "DaosGdrBackend batched allocation failed after %d attempt(s)",
            num_attempts,
        )
        return None

    def get_allocator_backend(self) -> "DaosGdrBackend":
        # Returning self is what makes storage_manager.py's batched put
        # path (:405-429) allocate a GPU buffer from *our* allocator and
        # copy+sync the source tensor into it before batched_submit_put_task
        # is ever called -- see module docstring.
        return self

    # -------------------------------------------------------------------
    # StorageBackendInterface
    # -------------------------------------------------------------------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # `pin` is accepted but ignored, matching GdsBackend.contains
        # (gds_backend.py:520-528, "# TODO: implement pin() semantics").
        #
        # This answers "is it in DAOS *now*", deliberately not "is it in
        # DAOS or on its way there": since v2 a put stays in flight after
        # batched_submit_put_task returns, so a key can be queued and not
        # yet visible here. That is the framework's own split --
        # exists_in_put_tasks() is the separate question, and it now
        # reports the real in-flight set (see _do_put's finally).
        meta_buf = ctypes.create_string_buffer(META_BUF_CAP)
        meta_len = ctypes.c_size_t(META_BUF_CAP)
        rc = self._lib.daosgdr_stat(
            self._ctx, self._key_bytes(key), meta_buf, ctypes.byref(meta_len)
        )
        if rc != 0:
            logger.error(
                "daosgdr_stat failed in contains() for key %s: rc=%d",
                key.to_string(),
                rc,
            )
            return False
        return meta_len.value > 0

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._put_lock:
            return key in self._put_tasks

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List[Future], None]:
        """v2: asynchronous. Every key's DAOS write is queued on
        self._executor and this method returns as soon as the last one is
        queued -- it never waits for a daos_obj_update_gpu to finish. Same
        external shape (List[Future], one per key, in order) as
        GdsBackend.batched_submit_put_task (gds_backend.py:615-634).

        Note that storage_manager.py DISCARDS the returned list
        (storage_manager.py:429 calls this for its side effect only) and
        then immediately ref_count_down()s every MemoryObj it passed in
        (:430-432). That is precisely why the ref_count_up() below happens
        in THIS (the calling) thread rather than in the worker: by the time
        a worker actually touches the GPU buffer, the storage manager has
        already dropped its own reference, and only our count is keeping
        the buffer alive. The matching ref_count_down() is in _do_put's
        finally, i.e. after the RDMA has completed -- the ref now spans the
        whole in-flight window instead of just the inline call.

        `transfer_spec` is accepted but unused, matching GdsBackend (it
        never references transfer_spec in its body either).
        """
        futures: List[Future] = []
        for key, memory_obj in zip(keys, objs, strict=False):
            futures.append(
                self._submit_put_task(key, memory_obj, on_complete_callback)
            )
        return futures

    def _submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> Future:
        """Queue one key's put. Runs on the caller's (critical-path) thread,
        so it does as little as possible: a set membership check, a
        ref_count_up, and executor.submit.
        """
        assert memory_obj.tensor is not None

        # Constructed here, in the submitting thread, so that the worker's
        # first mark measures queue delay. See _Timer's docstring.
        timer = _Timer() if _TIMING_ENABLED else None

        with self._put_lock:
            if self._closed:
                future: Future = Future()
                future.set_exception(
                    RuntimeError(
                        "DaosGdrBackend.batched_submit_put_task called after "
                        "close(); the DAOS ctx is gone or going away"
                    )
                )
                return future
            if key in self._put_tasks:
                # A put for this key is already in flight. Skipping the
                # duplicate is the same choice local_disk_backend.py:333-336
                # makes ("skip repeated save"); it matters more here than in
                # the v1 synchronous version, where the window in which a key
                # could be "in progress" was only ever the inline call.
                # Resolved-True is returned (rather than None) so the returned
                # list stays index-aligned with `keys`.
                logger.debug("Put task for %s is already in progress.", key)
                dup: Future = Future()
                dup.set_result(True)
                return dup
            self._put_tasks.add(key)

        memory_obj.ref_count_up()

        try:
            return self._executor.submit(
                self._do_put, key, memory_obj, on_complete_callback, timer
            )
        except RuntimeError as e:
            # close() won the race between the self._closed check above and
            # this submit. Undo both halves of the bookkeeping we just did,
            # or the buffer leaks and the key stays "in progress" forever.
            self._release_memory_obj(memory_obj)
            with self._put_lock:
                self._put_tasks.discard(key)
            logger.error(
                "DaosGdrBackend: executor rejected put for key %s: %s",
                key.to_string(),
                e,
            )
            failed: Future = Future()
            failed.set_exception(e)
            return failed

    def _do_put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
        timer: Optional["_Timer"],
    ) -> bool:
        """Runs on a worker thread. Everything here -- including pack_meta
        and the ctypes marshalling -- is off vLLM's critical path, which is
        why the validation checks stayed in the worker rather than being
        hoisted into _submit_put_task: a failure is reported through the
        Future (and the log), not by blocking the caller.

        Uses self._ctx / self._lib, i.e. the ONE shared DAOS object handle.
        That sharing is measured-safe -- see the module docstring.
        """
        if timer is not None:
            timer.mark("queued")

        ok = False
        try:
            tensor = memory_obj.tensor
            if tensor is None:
                raise RuntimeError(
                    f"DaosGdrBackend.put: MemoryObj for key {key.to_string()} "
                    "has no tensor"
                )
            if not tensor.is_cuda:
                raise RuntimeError(
                    f"DaosGdrBackend.put: tensor for key {key.to_string()} is "
                    f"not on a CUDA device (got {tensor.device})"
                )
            if not tensor.is_contiguous():
                # Not silently .contiguous()'d: that would allocate a new
                # buffer at a new address, which is not the buffer
                # storage_manager just allocated/synced for us via
                # get_allocator_backend() -- see module docstring. A
                # non-contiguous tensor here means something upstream
                # violated that contract, so fail loudly instead.
                raise RuntimeError(
                    f"DaosGdrBackend.put: tensor for key {key.to_string()} is "
                    "not contiguous"
                )

            meta = pack_meta(tensor, memory_obj.metadata.fmt)
            if timer is not None:
                timer.mark("pack_meta")

            meta_cbuf = ctypes.create_string_buffer(meta, len(meta))
            if timer is not None:
                timer.mark("create_string_buffer")

            rc = self._lib.daosgdr_put(
                self._ctx,
                self._key_bytes(key),
                ctypes.c_void_p(tensor.data_ptr()),
                ctypes.c_size_t(tensor.numel() * tensor.element_size()),
                meta_cbuf,
                ctypes.c_size_t(len(meta)),
            )
            if timer is not None:
                timer.mark("ctypes_put")

            ok = rc == 0
            if not ok:
                logger.error(
                    "daosgdr_put failed for key %s: rc=%d", key.to_string(), rc
                )
        except Exception:
            # Logged here because storage_manager discards our Futures --
            # without this, an exception would vanish into an unread Future.
            # Re-raised so a caller that DOES read the Future (e.g.
            # async_put_test.py) still sees it.
            logger.exception("DaosGdrBackend.put failed for key %s", key.to_string())
            raise
        finally:
            # Release the GPU buffer only now: the RDMA is done (or failed),
            # so nothing is reading the buffer any more. Via the helper
            # because this runs on a worker thread -- see its docstring.
            self._release_memory_obj(memory_obj)
            with self._put_lock:
                self._put_tasks.discard(key)
            if timer is not None:
                timer.mark("cleanup")
                self._record_put_timing(key, timer)

        # Deliberately after the finally block, so the key is no longer
        # advertised as in-progress when the callback runs -- same ordering
        # as GdsBackend._async_save_bytes_to_disk (gds_backend.py:706-719).
        #
        # WARNING: this runs ON A WORKER THREAD, not on the thread that
        # called batched_submit_put_task. In v1 (synchronous) the callback
        # ran inline on the caller's thread; anything the callback touches
        # must now be thread-safe. LMCache's own callers of this parameter
        # were written against GdsBackend, whose callback also runs off the
        # submitting thread, so this matches the framework's expectation --
        # but a callback that was only ever exercised against a synchronous
        # backend has not been tested under that assumption.
        if ok and on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception:
                logger.exception(
                    "on_complete_callback failed for key %s", key.to_string()
                )
        return ok

    def _release_memory_obj(self, memory_obj: MemoryObj) -> None:
        """ref_count_down() under the allocator's own lock.

        Necessary because of how LMCache wires MemoryObj back to its
        allocator, and it only became necessary once put moved to worker
        threads:

          - MemoryObj.ref_count_down() (memory_management.py:759-775) calls
            self.parent_allocator.free(self) when the count reaches zero.
          - parent_allocator is the INNER TensorMemoryAllocator
            (tensor_memory_allocator.py:121), not the GPUMemoryAllocator
            wrapper we hold.
          - TensorMemoryAllocator.free (tensor_memory_allocator.py:199-220)
            takes no lock: it mutates address_manager's free list and
            num_active_allocations directly.
          - The lock that serializes this arena,
            GPUMemoryAllocator.device_mem_lock
            (gpu_memory_allocator.py:66), is only held by the wrapper's own
            allocate/free/batched_*/memcheck (:87, :111, :123, :139, :144)
            -- so a free reached via ref_count_down() bypasses it entirely.

        In v1 that was harmless: the put ran inline, so this backend only
        ever released a buffer from the same thread that allocated it. In
        v2 the last reference is dropped by a worker thread (storage_manager
        drops its own at :430-432, before the write finishes), which would
        otherwise mutate the free list concurrently with an allocate() on
        vLLM's thread -- silent arena corruption, not a clean error.

        Lock ordering is safe: this takes device_mem_lock and then (inside
        ref_count_down) the MemoryObj's own lock. No path in LMCache does
        the reverse -- ref_count_down holds the object lock but reaches only
        the inner allocator, which never touches device_mem_lock -- so there
        is no cycle.
        """
        lock = getattr(self.memory_allocator, "device_mem_lock", None)
        if lock is None:
            memory_obj.ref_count_down()
            return
        with lock:
            memory_obj.ref_count_down()

    def _record_put_timing(self, key: CacheEngineKey, timer: "_Timer") -> None:
        d = timer.log(logger, f"put key={key.to_string()}")
        with self._timing_lock:
            self._last_put_timing_ms = d
            self._put_timings.append(d)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        timer = _Timer() if _TIMING_ENABLED else None
        try:
            meta_buf = ctypes.create_string_buffer(META_BUF_CAP)
            meta_len = ctypes.c_size_t(META_BUF_CAP)
            rc = self._lib.daosgdr_stat(
                self._ctx, self._key_bytes(key), meta_buf, ctypes.byref(meta_len)
            )
            if timer is not None:
                timer.mark("daosgdr_stat")
            if rc != 0:
                logger.error(
                    "daosgdr_stat failed in get_blocking() for key %s: rc=%d",
                    key.to_string(),
                    rc,
                )
                return None
            if meta_len.value == 0:
                return None  # not found -- rc==0 && meta_len==0, confirmed by shim_test.c

            try:
                shape, dtype, fmt, nbytes = unpack_meta(meta_buf.raw[: meta_len.value])
            except ValueError:
                logger.exception(
                    "Failed to decode meta for key %s -- corrupt or foreign entry",
                    key.to_string(),
                )
                return None
            if timer is not None:
                timer.mark("unpack_meta")

            memory_obj = self.allocate(shape, dtype, fmt=fmt)
            if timer is not None:
                timer.mark("allocate")
            if memory_obj is None:
                logger.error(
                    "GPU allocation failed during get_blocking() for key %s",
                    key.to_string(),
                )
                return None

            tensor = memory_obj.tensor
            if tensor is None or not tensor.is_cuda:
                logger.error(
                    "Allocated MemoryObj for key %s has no valid CUDA tensor",
                    key.to_string(),
                )
                self._release_memory_obj(memory_obj)
                return None

            rc = self._lib.daosgdr_get(
                self._ctx,
                self._key_bytes(key),
                ctypes.c_void_p(tensor.data_ptr()),
                ctypes.c_size_t(nbytes),
            )
            if timer is not None:
                timer.mark("daosgdr_get")
            if rc != 0:
                logger.error(
                    "daosgdr_get failed for key %s: rc=%d", key.to_string(), rc
                )
                self._release_memory_obj(memory_obj)
                return None

            return memory_obj
        finally:
            if timer is not None:
                d = timer.log(logger, f"get key={key.to_string()}")
                with self._timing_lock:
                    self._last_get_timing_ms = d

    def pin(self, key: CacheEngineKey) -> bool:
        # No eviction policy in this backend, same as GdsBackend
        # (gds_backend.py:1104-1112).
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        rc = self._lib.daosgdr_remove(self._ctx, self._key_bytes(key))
        if rc != 0:
            logger.error(
                "daosgdr_remove failed for key %s: rc=%d", key.to_string(), rc
            )
            return False
        return True

    def close(self) -> None:
        """Shutdown order is load-bearing.

        In-flight workers are still issuing daos_obj_update_gpu against
        self._ctx and still holding references to GPU buffers owned by
        self.memory_allocator. So: (1) refuse new submissions, (2) drain the
        executor and WAIT, and only then (3) daosgdr_fini() the ctx and (4)
        close the allocator. Finalizing the ctx first would close the
        object/container/pool handles out from under a live RDMA; closing
        the allocator first would free the GPU buffer a live RDMA is
        reading. Both are use-after-free, and neither would necessarily
        show up as a clean error.

        Idempotent: a second call finds _closed already True and skips the
        drain, and the _ctx / memory_allocator steps are already guarded.
        """
        # getattr-guarded like the steps below: close() may be reached from
        # a partially-constructed instance if __init__ raised (e.g.
        # daosgdr_init failure) before these attributes existed.
        lock = getattr(self, "_put_lock", None)
        if lock is None:
            already_closed, pending = True, 0
        else:
            with lock:
                already_closed = self._closed
                self._closed = True
                pending = len(self._put_tasks)

        executor = getattr(self, "_executor", None)
        if executor is not None and not already_closed:
            if pending:
                logger.info(
                    "DaosGdrBackend.close(): draining %d in-flight put(s)",
                    pending,
                )
            # wait=True is the whole point -- see the docstring above.
            executor.shutdown(wait=True)

        if getattr(self, "_ctx", None):
            self._lib.daosgdr_fini(self._ctx)
            self._ctx = None
        if getattr(self, "memory_allocator", None) is not None:
            self.memory_allocator.close()
        logger.info("DaosGdrBackend closed.")

