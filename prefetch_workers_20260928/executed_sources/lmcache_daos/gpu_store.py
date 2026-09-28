"""Opt-in, process-local GPU-only KV store routing for LMCache 0.5.x.

No installed files are changed. Only managers with a gpu_direct DAOS backend
are routed differently; host_staged remains the default. Payload stays on GPU,
but keys, metadata, pointer tables and DAOS control traffic still use the CPU.
"""
import functools
import contextvars

import torch

_store_context = contextvars.ContextVar('daos_store_context', default=None)


def validate_config(config, path):
    mirror = (getattr(config, 'extra_config', None) or {}).get('daosgds.async_dram', False)
    if mirror and (path != 'gpu_direct' or not getattr(config, 'local_cpu', False)):
        raise ValueError('async_dram requires gpu_direct and local_cpu enabled')
    if path not in {"host_staged", "gpu_direct"}:
        raise ValueError("daosgds.store_path must be host_staged or gpu_direct")
    if path == "host_staged":
        return
    for name in ("local_cpu", "local_disk", "remote_url", "enable_pd",
                 "use_layerwise", "enable_blending"):
        if name == 'local_cpu' and mirror:
            continue
        if getattr(config, name, False):
            raise ValueError(f"gpu_direct store requires {name} disabled")


def select_backend(manager):
    direct = [(name, backend) for name, backend in manager.storage_backends.items()
              if getattr(backend, "store_path", None) == "gpu_direct"]
    if not direct:
        return None
    if len(direct) != 1:
        raise ValueError("gpu_direct store requires exactly one DAOS backend")
    name, backend = direct[0]
    if set(manager.storage_backends) - {name, "LocalCPUBackend"}:
        raise ValueError("gpu_direct store does not support additional storage tiers")
    cpu = manager.storage_backends.get("LocalCPUBackend")
    mirror = getattr(backend, 'dram_mirror', None)
    if cpu is not None and getattr(cpu, "use_hot", False) and (mirror is None or mirror.cpu is not cpu):
        raise ValueError("gpu_direct store requires CPU cache retention disabled")
    return name, backend


def put_direct(manager, selected, keys, objects, transfer_spec=None, location=None):
    name, backend = selected
    # V2's from_gpu is asynchronous for CUDA destinations. Synchronize BEFORE
    # any skip/error/ref release, not just inside the DAOS worker: otherwise a
    # skipped duplicate could free a buffer while the gather kernel writes it.
    try:
        context = _store_context.get()
        mirror = getattr(backend, 'dram_mirror', None)
        if mirror is not None and context is not None and context[0] is manager:
            # Wait ONLY for gather, not for unrelated asynchronous D2H copies.
            context[1].synchronize()
        else:
            # Non-engine callers must supply already-written payloads; retain
            # the conservative original barrier for those manual integrations.
            torch.cuda.synchronize(backend.device_id)
        if len(keys) != len(objects):
            raise ValueError("GPU store key/object count mismatch")
        for obj in objects:
            tensor = obj.tensor
            if tensor is None or not tensor.is_cuda or tensor.device.index != backend.device_id:
                raise ValueError("gpu_direct store only accepts local GPU payloads")
        if location is not None and location != name:
            raise ValueError(f"gpu_direct store location must be {name!r} or None")
        with manager._bypass_lock:
            bypassed = name in manager._bypassed_backends
        if not bypassed:
            kwargs = {'transfer_spec': transfer_spec}
            if mirror is not None:
                kwargs['payload_ready'] = True
            backend.batched_submit_put_task(keys, objects, **kwargs)
    finally:
        # Submitted workers retain their own references. Also release on
        # store-disabled, duplicate, bypass and failed-submit branches.
        for obj in objects:
            obj.ref_count_down()


def install():
    from lmcache.v1.storage_backend.storage_manager import StorageManager

    if getattr(StorageManager, "_daos_gpu_store_installed", False):
        return
    original_allocator = StorageManager._get_allocator_backend
    original_put = StorageManager.batched_put

    @functools.wraps(original_allocator)
    def allocator(self, config):
        selected = select_backend(self)
        self._daos_gpu_store = selected
        if selected is not None:
            validate_config(config, "gpu_direct")
            return selected[1]
        return original_allocator(self, config)

    @functools.wraps(original_put)
    def put(self, keys, memory_objs, transfer_spec=None, location=None):
        selected = getattr(self, "_daos_gpu_store", None)
        if selected is None:
            return original_put(self, keys, memory_objs, transfer_spec, location)
        return put_direct(self, selected, keys, memory_objs, transfer_spec, location)

    StorageManager._get_allocator_backend = allocator
    StorageManager.batched_put = put
    StorageManager._daos_gpu_store_installed = True


def install_store_context():
    from lmcache.v1.cache_engine import LMCacheEngine
    if getattr(LMCacheEngine, '_daos_store_context_installed', False):
        return
    original = LMCacheEngine.store

    @functools.wraps(original)
    def store(self, *args, **kwargs):
        selected = getattr(self.storage_manager, '_daos_gpu_store', None)
        if selected is None or getattr(selected[1], 'dram_mirror', None) is None:
            return original(self, *args, **kwargs)
        stream = getattr(self.gpu_connector, 'store_stream', None)
        if stream is None:
            raise ValueError('async_dram needs a connector with a dedicated store_stream')
        token = _store_context.set((self.storage_manager, stream))
        try:
            return original(self, *args, **kwargs)
        finally:
            _store_context.reset(token)

    LMCacheEngine.store = store
    LMCacheEngine._daos_store_context_installed = True
