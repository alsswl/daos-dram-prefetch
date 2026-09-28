#!/usr/bin/env python3
"""Real DFS GPU-direct round-trip smoke test against a live DAOS server."""

from __future__ import annotations

import argparse
import time
import uuid

import torch

from lmcache_daos.dfs_binding import DfsSys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="discospool")
    parser.add_argument("--container", default="kvcache")
    parser.add_argument("--root", default="/minji-v2")
    parser.add_argument("--size-mib", type=int, default=64)
    args = parser.parse_args()

    if args.size_mib <= 0:
        raise ValueError("--size-mib must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not DfsSys.gpu_supported():
        raise RuntimeError("loaded libdfs has no dfs_read_gpu/dfs_write_gpu")

    device_id = torch.cuda.current_device()
    size = args.size_mib * 1024 * 1024
    root = args.root.rstrip("/") or "/"
    path = f"{root}/gpu-roundtrip-{uuid.uuid4().hex}"
    dfs = DfsSys(args.pool, args.container)
    handle = None

    try:
        dfs.mkdir_p(root)
        torch.manual_seed(20260916)
        source = torch.randint(0, 256, (size,), dtype=torch.uint8, device="cuda")
        destination = torch.empty_like(source)
        torch.cuda.synchronize(device_id)

        handle = dfs.open_rdwr_create(path)
        started = time.perf_counter()
        written = dfs.write_gpu_from(
            handle, 0, size, source.data_ptr(), device_id
        )
        torch.cuda.synchronize(device_id)
        put_seconds = time.perf_counter() - started
        dfs.close_obj(handle)
        handle = None
        if written != size:
            raise AssertionError(f"short DFS GPU write: {written}/{size}")

        handle = dfs.open_rdonly(path)
        started = time.perf_counter()
        read = dfs.read_gpu_into(
            handle, 0, size, destination.data_ptr(), device_id
        )
        torch.cuda.synchronize(device_id)
        get_seconds = time.perf_counter() - started
        dfs.close_obj(handle)
        handle = None
        if read != size:
            raise AssertionError(f"short DFS GPU read: {read}/{size}")

        if not torch.equal(source, destination):
            mismatches = int((source != destination).sum().item())
            raise AssertionError(f"GPU payload mismatch: {mismatches} byte(s)")

        print(
            "PASS "
            f"pool={args.pool} container={args.container} path={path} "
            f"device={device_id} bytes={size} payload=match "
            f"write={put_seconds * 1e3:.3f}ms "
            f"write_GBps={size / put_seconds / 1e9:.3f} "
            f"read={get_seconds * 1e3:.3f}ms "
            f"read_GBps={size / get_seconds / 1e9:.3f}"
        )
        return 0
    finally:
        if handle is not None:
            dfs.close_obj(handle)
        removed = dfs.remove(path)
        print(f"CLEANUP removed={removed} path={path}")
        dfs.close()


if __name__ == "__main__":
    raise SystemExit(main())
