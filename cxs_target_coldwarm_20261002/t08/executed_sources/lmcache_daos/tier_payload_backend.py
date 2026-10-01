"""Independent DAOS/DRAM prefetch switches with identical async lookup.

Disabled tiers keep the published plan queued until retrieve. They retain
the existing demand handoff, pin ownership and error handling unchanged.
"""
from .early_lookup_backend import EarlyLookupBackend, EarlyLookupCoordinator
from .gds_backend import _cfg


class TierPayloadCoordinator(EarlyLookupCoordinator):
    def start(self, manager, batch, keys):
        if self.backend.payload_prefetch_tiers[batch.tier]:
            return super().start(manager, batch, keys)
        self.emit('early_payload_demand_only', request_id=batch.rid,
                  tier=batch.tier, chunks=len(keys))


class TierPayloadBackend(EarlyLookupBackend):
    def __init__(self, config, dst_device='cuda', metadata=None,
                 local_cpu_backend=None, loop=None):
        self.payload_prefetch_tiers = {
            tier: _cfg(config, f'early_{tier}_prefetch', True)
            for tier in ('daos', 'dram')}
        if any(type(v) is not bool for v in self.payload_prefetch_tiers.values()):
            raise ValueError('early_daos_prefetch and early_dram_prefetch must be YAML booleans')
        if _cfg(config, 'early_lookup', False) is not True:
            raise ValueError('TierPayloadBackend requires early_lookup=true')
        super().__init__(config, dst_device, metadata, local_cpu_backend, loop)
        # The inherited CPU close hook resolves self.early_lookup at close time.
        # No lookup jobs have been admitted during backend construction.
        self.early_lookup = TierPayloadCoordinator(self, local_cpu_backend)
        self._staging_trace.emit('tier_payload_policy', **self.payload_prefetch_tiers)
