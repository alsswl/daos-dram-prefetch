"""Backend-level object transport tests with fake GPU/DAOS objects."""

import threading
import time

import torch

from lmcache.v1.memory_management import MemoryFormat
from lmcache_daos.gds_backend import DaosGdsBackend, _meta_pack


class FakeKey:
    def to_string(self):
        return "model@1@worker@deadbeef@float16"


class FakeTensor:
    is_cuda = True

    def __init__(self, ptr=0x12340000):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr


class FakeMetadata:
    fmt = MemoryFormat.KV_2LTD


class FakeMemoryObj:
    def __init__(self, size=32):
        self.tensor = FakeTensor()
        self.metadata = FakeMetadata()
        self.size = size
        self.released = False

    def get_size(self):
        return self.size

    def get_shapes(self):
        return [torch.Size([2, 8])]

    def get_dtypes(self):
        return [torch.float16]

    def ref_count_down(self):
        self.released = True


class FakeAllocator:
    def __init__(self):
        self.next_obj = FakeMemoryObj()

    def allocate(self, shapes, dtypes, fmt):
        assert shapes == [torch.Size([2, 8])]
        assert dtypes == [torch.float16]
        assert fmt == MemoryFormat.KV_2LTD
        return self.next_obj


class FakeObjectStore:
    def __init__(self):
        self.metadata = None
        self.put_args = None
        self.get_args = None

    def stat(self, key):
        return self.metadata

    def put(self, key, ptr, size, metadata, device_id):
        self.put_args = (key, ptr, size, device_id)
        self.metadata = metadata

    def get(self, key, ptr, size, device_id):
        self.get_args = (key, ptr, size, device_id)


def make_backend():
    be = DaosGdsBackend.__new__(DaosGdsBackend)
    be.transport = "object"
    be.object_namespace = "minji-v2:"
    be.device_id = 2
    be._object = FakeObjectStore()
    be.memory_allocator = FakeAllocator()
    be._known = set()
    be._known_lock = threading.Lock()
    be._put_tasks = set()
    be._put_lock = threading.Lock()
    be.stats = {
        "put": 0, "put_bytes": 0, "get": 0, "get_bytes": 0,
        "miss": 0, "alloc_fail": 0, "get_ms": 0.0, "put_ms": 0.0,
    }
    be._ensure_cuda_ctx = lambda: None
    return be


def test_ready_payload_put_does_not_wait_unrelated_cuda_streams(monkeypatch):
    be = make_backend()
    def fail_sync(*args):
        raise AssertionError('device-wide sync would wait for background D2H')
    monkeypatch.setattr(torch.cuda, 'synchronize', fail_sync)
    obj = FakeMemoryObj()
    be._put_one(FakeKey(), obj, None, payload_ready=True)
    assert be.stats['put'] == 1
    assert obj.released


def test_object_transport_put_and_get(monkeypatch):
    be = make_backend()
    key = FakeKey()
    source = FakeMemoryObj()
    callback = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    be._put_one(key, source, callback.append)
    assert be._object.put_args == (
        "minji-v2:" + key.to_string(), source.tensor.data_ptr(),
        source.get_size(), 2
    )
    assert source.released and callback == [key]

    # Force the read path to consult the fake DAOS store rather than the
    # positive in-process contains cache populated by the put.
    be._known.clear()
    result = be._get_object_blocking(key, time.perf_counter())
    assert result is be.memory_allocator.next_obj
    assert be._object.get_args == (
        "minji-v2:" + key.to_string(), result.tensor.data_ptr(),
        result.get_size(), 2
    )
    assert be.stats["put"] == 1 and be.stats["get"] == 1


def test_object_transport_metadata_is_multi_tensor_capable():
    metadata = _meta_pack(
        32,
        [torch.Size([2, 4]), torch.Size([2, 4])],
        [torch.float16, torch.float16],
        MemoryFormat.KV_2LTD,
    )
    assert b'"shapes":[[2,4],[2,4]]' in metadata
