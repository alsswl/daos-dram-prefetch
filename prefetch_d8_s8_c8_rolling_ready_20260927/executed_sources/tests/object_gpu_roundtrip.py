#!/usr/bin/env python3
"""Real DAOS object-API GPU-direct round-trip smoke test.

Unlike the unit tests, this requires a CUDA GPU and a live DAOS server.  It
uses a unique dkey, verifies the host metadata and every GPU payload byte,
then removes the dkey in ``finally``.
"""

from __future__ import annotations

import argparse
import time
import uuid

import torch

from lmcache_daos.object_binding import DaosObjectStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="discospool")
    parser.add_argument("--container", default="kvcache")
    parser.add_argument("--size-mib", type=int, default=64)
    args = parser.parse_args()

    if args.size_mib <= 0:
        raise ValueError("--size-mib must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device_id = torch.cuda.current_device()
    size = args.size_mib * 1024 * 1024
    key = f"minji-v2:gpu-roundtrip:{uuid.uuid4().hex}"
    metadata = f"gpu-roundtrip:size={size}:device={device_id}".encode()
    store = DaosObjectStore(args.pool, args.container)

    try:
        torch.manual_seed(20260916)
        source = torch.randint(0, 256, (size,), dtype=torch.uint8, device="cuda")
        destination = torch.empty_like(source)
        torch.cuda.synchronize(device_id)

        started = time.perf_counter()
        store.put(key, source.data_ptr(), source.numel(), metadata, device_id)
        torch.cuda.synchronize(device_id)
        put_seconds = time.perf_counter() - started

        observed_metadata = store.stat(key)
        if observed_metadata != metadata:
            raise AssertionError(
                f"metadata mismatch: {observed_metadata!r} != {metadata!r}"
            )

        started = time.perf_counter()
        store.get(
            key, destination.data_ptr(), destination.numel(), device_id
        )
        torch.cuda.synchronize(device_id)
        get_seconds = time.perf_counter() - started

        equal = torch.equal(source, destination)
        if not equal:
            mismatches = int((source != destination).sum().item())
            raise AssertionError(f"GPU payload mismatch: {mismatches} byte(s)")

        print(
            "PASS "
            f"pool={args.pool} container={args.container} device={device_id} "
            f"bytes={size} metadata=match payload=match "
            f"put={put_seconds * 1e3:.3f}ms "
            f"put_GBps={size / put_seconds / 1e9:.3f} "
            f"get={get_seconds * 1e3:.3f}ms "
            f"get_GBps={size / get_seconds / 1e9:.3f}"
        )
        return 0
    finally:
        try:
            store.remove(key)
            if store.stat(key) is not None:
                raise AssertionError("test dkey still exists after remove")
            print(f"CLEANUP removed={key}")
        finally:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
