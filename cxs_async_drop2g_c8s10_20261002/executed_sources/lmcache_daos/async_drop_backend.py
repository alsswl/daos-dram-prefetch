"""Opt-in original async/drop experiment, with an isolated OID and telemetry.

Native store stops at its first allocation failure. Native async lookup reports
the successfully fetched contiguous prefix to the scheduler. No window/retry.
"""
import os
import threading

from .capacity_probe_backend import CapacityProbeBackend
from .gds_backend import DaosGdsBackend, _cfg
from .gpu_store import install_store_context
from .no_dram import DisabledDramMirror, validate_no_dram
from .placement_backend import PlacementObjectStore
from .staging_probe_backend import ProbeMixin
from .store_probe_backend import StoreProbeMixin


class OriginalKeyStore(PlacementObjectStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scheme.mode != 'baseline' or self._spread_shards is not None:
            self.close()
            raise ValueError('Original hash placement required')

    def address(self, key):
        # Baseline ignores the position. Preserve the full original dkey bytes.
        return super().address(key + '\x1f0')


class AsyncDropBackend(StoreProbeMixin, ProbeMixin, DaosGdsBackend):
    _observed_get = CapacityProbeBackend._observed_get
    batched_get_non_blocking = CapacityProbeBackend.batched_get_non_blocking

    def __init__(self, config, *args, **kwargs):
        validate_no_dram(config)
        if not config.enable_async_loading or config.use_layerwise:
            raise ValueError('Requires native async non-layerwise loading')
        if _cfg(config, 'retrieve_window_mib', 0) or _cfg(config, 'store_window_mib', 0):
            raise ValueError('Original drop mode cannot use windows')
        self.placement_manifest = _cfg(config, 'placement_manifest')
        self._read_probe_local = threading.local()
        self.dram_mirror = DisabledDramMirror(None)
        self.cpu_prefetch = None
        # Keep LMCache's original async serializer; the optional weighted
        # serializer rejects requests larger than the whole staging arena.
        os.environ['DAOS_GDS_MULTI_PREFETCH'] = '0'
        super().__init__(config, *args, **kwargs)
        original = self.memory_allocator.allocate

        def allocate(*args, **kwargs):
            obj = original(*args, **kwargs)
            current = getattr(self._read_probe_local, 'current', None)
            if current is not None and obj is None:
                current['allocation_failed'] = True
            return obj

        self.memory_allocator.allocate = allocate
        install_store_context()
        self._staging_trace.emit('async_drop_enabled', read_retry=False,
            store_retry=False, native_serializer='AsyncSingleSerializer',
            oid=self._object.manifest['oid'])

    def _create_object_store(self, **kwargs):
        return OriginalKeyStore(self.placement_manifest, **kwargs)

    def close(self):
        self._staging_trace.emit('async_drop_final_stats', stats=dict(self.stats))
        super().close()
