"""Unit tests for the DFS-bypass ctypes wrapper (no DAOS or GPU needed)."""

import ctypes

import pytest

from lmcache_daos.object_binding import DaosObjectStore, _key_bytes


class FakeObjectLibrary:
    def __init__(self):
        self.entries = {}
        self.put_device = None
        self.get_device = None
        self.closed = False

    def daosgdr_init(self, pool, container):
        assert pool == b"pool" and container == b"container"
        return 123

    def daosgdr_stat(self, ctx, key, buf, actual_ptr):
        assert ctx == 123
        data = self.entries.get(key)
        if data is None:
            actual_ptr._obj.value = 0
            return 0
        capacity = actual_ptr._obj.value
        actual_ptr._obj.value = len(data)
        if capacity < len(data):
            return -2013
        ctypes.memmove(buf, data, len(data))
        return 0

    def daosgdr_put_device(
        self, ctx, key, gpu_ptr, size, meta_buf, meta_size, device_id
    ):
        assert ctx == 123
        self.entries[key] = bytes(meta_buf.raw[: meta_size.value])
        self.put_device = device_id.value
        return 0

    def daosgdr_get_device(self, ctx, key, gpu_ptr, size, device_id):
        assert ctx == 123 and key in self.entries
        self.get_device = device_id.value
        return 0

    def daosgdr_remove(self, ctx, key):
        assert ctx == 123
        self.entries.pop(key, None)
        return 0

    def daosgdr_fini(self, ctx):
        assert ctx == 123
        self.closed = True


def test_object_store_roundtrip_and_metadata_resize():
    lib = FakeObjectLibrary()
    store = DaosObjectStore(
        "pool", "container", library=lib, initial_meta_capacity=4
    )
    assert store.stat("missing") is None

    metadata = b"metadata-longer-than-four-bytes"
    store.put("cache-key", 0x1234, 4096, metadata, device_id=2)
    assert store.stat("cache-key") == metadata
    store.get("cache-key", 0x5678, 4096, device_id=2)
    assert lib.put_device == 2 and lib.get_device == 2
    assert store.remove("cache-key")
    assert store.stat("cache-key") is None

    store.close()
    store.close()
    assert lib.closed


def test_object_key_validation():
    assert _key_bytes("abc") == b"abc"
    with pytest.raises(ValueError):
        _key_bytes("")
    with pytest.raises(ValueError):
        _key_bytes(b"bad\x00key")

