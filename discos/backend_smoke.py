#!/usr/bin/env python3
"""
backend_smoke.py

Standalone smoke test for DaosGdrBackend -- no vLLM, no LMCacheEngine/
CacheEngine, no storage_plugin_launcher. Exercises DaosGdrBackend's public
API directly against a hand-built minimal config, so a failure here can
only be in DaosGdrBackend/libdaosgdr.so, not in anything upstream of it.

Config construction: LMCacheEngineConfig is a dataclass assembled by
create_config_class() (lmcache/v1/config_base.py:218-...), which builds
each field from lmcache/v1/config.py's _CONFIG_DEFINITIONS (name -> type +
default), and registers three ways to build an instance:
  - from_env()  (config_base.py:260)  -- reads LMCACHE_* env vars
  - from_file() (config_base.py:318)  -- reads a YAML file
  - from_defaults(**kwargs) (config_base.py:350-368, exposed at :450)
    -- "Create configuration from defaults" with keyword overrides; every
    field not passed falls back to its declared default. This is the
    direct-construction path used below -- no file, no env vars.
No storage_plugin.<name>.module_path/class_name extra_config keys are
needed here, since we construct DaosGdrBackend ourselves instead of going
through storage_plugin_launcher (lmcache/v1/storage_backend/__init__.py:
40-108) -- those two keys only matter to the launcher's importlib lookup.

Run (PYTHONPATH is not needed here specifically: Python auto-inserts a
script's own directory at sys.path[0], so `import daosgdr_backend`
resolves as long as this file stays next to ~/daosgdr_backend.py --
LD_LIBRARY_PATH/D_GPU_DIRECT are still required, same as every other
script in this session, since libdaosgdr.so itself needs them):

  sudo env LD_LIBRARY_PATH=/opt/daos-gdr/prereq/release/ofi/lib:/opt/daos-gdr/lib64:/usr/local/cuda/lib64 \\
      D_GPU_DIRECT=1 ~/lmc-env/bin/python3 ~/backend_smoke.py testpool gdrcont2
"""

import sys
import traceback

import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat

from daosgdr_backend import DaosGdrBackend

# A400 has 4GB total VRAM -- keep the backend's own GPU allocator small so
# this smoke test never comes close to contending for VRAM with anything
# else that might be running.
_BUFFER_SIZE_MB = 64


def try_step(label: str, fn):
    """Run fn(), print PASS/FAIL with the label, and always return a bool
    -- never raises, so one failing step doesn't stop the rest from
    running. fn() itself is expected to print whatever diagnostic detail
    is useful and return True/False for pass/fail; an uncaught exception
    inside fn() is treated as a fail with a traceback printed."""
    print(f"\n--- {label} ---")
    try:
        ok = fn()
        print("PASS" if ok else "FAIL")
        return bool(ok)
    except Exception:
        traceback.print_exc()
        print("FAIL (exception)")
        return False


def main() -> int:
    pool = sys.argv[1] if len(sys.argv) > 1 else "testpool"
    cont = sys.argv[2] if len(sys.argv) > 2 else "gdrcont2"

    results: dict[str, bool] = {}
    state: dict = {}  # carries objects between steps (config, backend, key, tensors)

    # ---- 1. minimal config, direct construction (no YAML, no env) ----
    def step1():
        config = LMCacheEngineConfig.from_defaults(
            extra_config={
                "daos_gdr_pool": pool,
                "daos_gdr_cont": cont,
                "daos_gdr_buffer_size_mb": _BUFFER_SIZE_MB,
            }
        )
        print(f"extra_config={config.extra_config}")
        state["config"] = config
        return True

    results["1_config"] = try_step("1. LMCacheEngineConfig.from_defaults(...)", step1)
    if not results["1_config"]:
        return summarize(results)

    # ---- 2. construct DaosGdrBackend directly (launcher bypassed) ----
    def step2():
        backend = DaosGdrBackend(
            dst_device="cuda",
            config=state["config"],
            metadata=None,
            local_cpu_backend=None,
            loop=None,
        )
        print(f"{backend} constructed, memory_allocator={backend.memory_allocator}")
        state["backend"] = backend
        return True

    results["2_construct"] = try_step("2. DaosGdrBackend(...) direct construction", step2)
    if not results["2_construct"]:
        return summarize(results)

    backend: DaosGdrBackend = state["backend"]

    # CacheEngineKey's 5 fields (lmcache/utils.py:400-406): model_name,
    # world_size, worker_id, chunk_hash, dtype (request_configs is
    # optional and defaults to {}).
    key = CacheEngineKey(
        model_name="smoke-test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=0xC0FFEE,
        dtype=torch.float16,
    )
    print(f"\nkey.to_string() = {key.to_string()!r}")

    shape = torch.Size([2, 16, 4096])  # small KV-shaped tensor, ~256KiB at fp16
    dtype = torch.float16

    # ---- 3. allocate() + fill with random values ----
    def step3():
        orig_obj = backend.allocate(shape, dtype, fmt=MemoryFormat.KV_2LTD)
        if orig_obj is None or orig_obj.tensor is None:
            print("allocate() returned None or tensor is None")
            return False
        orig_obj.tensor.normal_()
        print(
            f"allocated shape={tuple(orig_obj.tensor.shape)} "
            f"dtype={orig_obj.tensor.dtype} device={orig_obj.tensor.device} "
            f"sample={orig_obj.tensor.flatten()[:4].tolist()}"
        )
        state["orig_obj"] = orig_obj
        return True

    results["3_allocate"] = try_step("3. backend.allocate(...) + fill tensor", step3)

    # ---- 4. batched_submit_put_task([key], [orig_obj]) ----
    def step4():
        orig_obj = state["orig_obj"]
        futures = backend.batched_submit_put_task([key], [orig_obj])
        if futures is None or len(futures) != 1:
            print(f"unexpected futures={futures!r}")
            return False
        fut = futures[0]
        exc = fut.exception()
        if exc is not None:
            print(f"future raised: {exc!r}")
            return False
        put_ok = fut.result()
        print(f"future.result()={put_ok}")
        return bool(put_ok)

    if results["3_allocate"]:
        results["4_put"] = try_step(
            "4. backend.batched_submit_put_task([key], [orig_obj])", step4
        )
    else:
        print("\n--- 4. skipped (step 3 failed) ---")
        results["4_put"] = False

    # ---- 5. contains(key) == True ----
    def step5():
        c = backend.contains(key)
        print(f"contains()={c}")
        return c

    results["5_contains_true"] = try_step("5. backend.contains(key) == True", step5)

    # ---- 6. get_blocking(key) -> torch.equal(original, fetched) ----
    def step6():
        fetched_obj = backend.get_blocking(key)
        if fetched_obj is None or fetched_obj.tensor is None:
            print("get_blocking() returned None or tensor is None")
            return False
        state["fetched_obj"] = fetched_obj
        orig_obj = state.get("orig_obj")
        equal = (
            orig_obj is not None
            and orig_obj.tensor is not None
            and torch.equal(orig_obj.tensor, fetched_obj.tensor)
        )
        print(
            f"fetched shape={tuple(fetched_obj.tensor.shape)} "
            f"dtype={fetched_obj.tensor.dtype} torch.equal(original, fetched)={equal} "
            f"orig ptr={orig_obj.tensor.data_ptr():#x} "
            f"fetched ptr={fetched_obj.tensor.data_ptr():#x}"
        )
        return equal

    results["6_get_equal"] = try_step(
        "6. backend.get_blocking(key) + torch.equal(original, fetched)", step6
    )

    # ---- 7. remove(key) -> contains(key) == False ----
    def step7():
        remove_ok = backend.remove(key)
        contains_after = backend.contains(key)
        print(f"remove()={remove_ok} contains_after_remove={contains_after}")
        return remove_ok and not contains_after

    results["7_remove"] = try_step(
        "7. backend.remove(key) -> backend.contains(key) == False", step7
    )

    # ---- release the MemoryObjs this script owns before closing ----
    def step_cleanup():
        orig_obj = state.get("orig_obj")
        fetched_obj = state.get("fetched_obj")
        if orig_obj is not None:
            orig_obj.ref_count_down()
        if fetched_obj is not None:
            fetched_obj.ref_count_down()
        return True

    try_step("cleanup: release MemoryObjs (ref_count_down)", step_cleanup)

    # ---- 8. close() ----
    def step8():
        backend.close()
        return True

    results["8_close"] = try_step("8. backend.close()", step8)

    return summarize(results)


def summarize(results: dict) -> int:
    print("\n=== SUMMARY ===")
    width = max((len(k) for k in results), default=0)
    for k, v in results.items():
        print(f"{k:<{width}}  {'PASS' if v else 'FAIL'}")
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

