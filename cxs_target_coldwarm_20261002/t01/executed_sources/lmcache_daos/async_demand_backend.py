"""Same metadata-first control path, optional speculative payload submission.

Both experimental arms use this class. early_payload_prefetch=false leaves
published plans queued: retrieve resolves CPU views directly or performs DAOS
reads using the existing demand path. It does not disable async lookup.
"""
from .early_lookup_backend import EarlyLookupBackend, EarlyLookupCoordinator
from .gds_backend import _cfg


class PayloadTimingCoordinator(EarlyLookupCoordinator):
    def start(self, manager, batch, keys):
        if self.backend.early_payload_prefetch:
            return super().start(manager, batch, keys)
        self.emit('early_payload_demand_only', request_id=batch.rid,
                  tier=batch.tier, chunks=len(keys))
        # No worker/serializer admission and no staging reservation. resolve()
        # owns the original queued->demand handoff, including abort and pins.


class AsyncDemandBackend(EarlyLookupBackend):
    def __init__(self, config, dst_device='cuda', metadata=None,
                 local_cpu_backend=None, loop=None):
        enabled = _cfg(config, 'early_payload_prefetch', True)
        if type(enabled) is not bool:
            raise ValueError('early_payload_prefetch must be a YAML boolean')
        if _cfg(config, 'early_lookup', False) is not True:
            raise ValueError('AsyncDemandBackend requires early_lookup=true')
        self.early_payload_prefetch = enabled
        super().__init__(config, dst_device, metadata, local_cpu_backend, loop)
        # No lookups exist during construction. The inherited close hook uses
        # self.early_lookup at close time, so it drains this coordinator.
        self.early_lookup = PayloadTimingCoordinator(self, local_cpu_backend)
        self._staging_trace.emit('early_payload_policy',
                                 speculative=enabled, lookup_backoff='config')
