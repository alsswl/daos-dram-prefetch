"""Versioned experimental addressing; absolute index must come from the caller.

No implicit enumeration of a request/window: those indices change for suffix
hits. This module does not patch LMCache's existing key-only backend API.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Addressing:
    mode: str
    placement_keys: tuple[str, ...]
    version: int = 1

    def __post_init__(self):
        if self.mode not in ('baseline', 'balanced') or self.version != 1:
            raise ValueError('Unsupported addressing scheme')
        if not self.placement_keys or len(set(self.placement_keys)) != len(self.placement_keys):
            raise ValueError('Expected one distinct placement key per shard')

    def address(self, full_key: str, absolute_chunk_index: int) -> tuple[bytes, bytes]:
        if not full_key or '\0' in full_key or absolute_chunk_index < 0:
            raise ValueError('A full KV identity and nonnegative absolute index are required')
        if self.mode == 'baseline':
            return full_key.encode(), b'kv'
        bucket = absolute_chunk_index % len(self.placement_keys)
        return self.placement_keys[bucket].encode(), full_key.encode()

    def to_dict(self):
        return dict(mode=self.mode, placement_keys=list(self.placement_keys), version=self.version)

    @classmethod
    def from_dict(cls, data):
        return cls(data['mode'], tuple(data['placement_keys']), data['version'])
