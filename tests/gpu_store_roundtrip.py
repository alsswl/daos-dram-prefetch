"""Live GPU gather -> DAOS -> GPU scatter correctness, isolated UUID dkey."""
import json
from types import SimpleNamespace as NS
import uuid

import torch
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.memory_management import MemoryFormat
from lmcache_daos.object_binding import DaosObjectStore


def main():
    torch.manual_seed(20260926)
    layers, heads, dim, chunk = 40, 8, 128, 128
    source = [torch.randn((2, 9, 16, heads, dim), dtype=torch.bfloat16, device='cuda')
              for _ in range(layers)]
    slots = torch.randperm(144, device='cuda')[:chunk].long()
    shape = (2, layers, chunk, heads * dim)
    host = torch.empty(shape, dtype=torch.bfloat16, pin_memory=True)
    direct = torch.empty(shape, dtype=torch.bfloat16, device='cuda')
    fetched = torch.empty_like(direct)
    obj = lambda t: NS(tensor=t, metadata=NS(fmt=MemoryFormat.KV_2LTD))
    connector = VLLMPagedMemGPUConnectorV2(heads * dim, layers)
    torch.cuda.synchronize()
    connector.batched_from_gpu([obj(host)], [0], [chunk], kvcaches=source, slot_mapping=slots)
    connector.batched_from_gpu([obj(direct)], [0], [chunk], kvcaches=source, slot_mapping=slots)
    torch.cuda.synchronize()
    assert torch.equal(host.to('cuda').view(torch.uint8), direct.view(torch.uint8))
    store = DaosObjectStore('discospool', 'kvcache')
    key = 'minji-gpu-store-correctness:' + uuid.uuid4().hex
    size = direct.numel() * direct.element_size()
    try:
        store.put(key, direct.data_ptr(), size, b'gpu-store-correctness', 0)
        store.get(key, fetched.data_ptr(), size, 0)
        torch.cuda.synchronize()
        assert store.stat(key) == b'gpu-store-correctness'
        assert torch.equal(direct.view(torch.uint8), fetched.view(torch.uint8))
        destination = [torch.zeros_like(t) for t in source]
        consumer = VLLMPagedMemGPUConnectorV2(heads * dim, layers)
        consumer.batched_to_gpu([obj(fetched)], [0], [chunk],
                               kvcaches=destination, slot_mapping=slots)
        torch.cuda.synchronize()
        for src, dst in zip(source, destination):
            assert torch.equal(src.view(2, 144, heads, dim)[:, slots].view(torch.uint8),
                               dst.view(2, 144, heads, dim)[:, slots].view(torch.uint8))
        print(json.dumps(dict(result='PASS', bytes=size, layers=layers, chunk=chunk,
                              host_vs_direct='byte_equal', daos_roundtrip='byte_equal',
                              paged_scatter='byte_equal')))
    finally:
        store.remove(key)
        assert store.stat(key) is None
        store.close()
        print('Removed own temporary UUID dkey:', key)


if __name__ == '__main__':
    main()
