"""Run vLLM with direct-scatter initialization hooks before GPU allocation.

LMCACHE_CONFIG_FILE=... python -m lmcache_daos.direct_scatter_launch serve ...
"""
import functools


def install():
    import lmcache.v1.gpu_connector as connectors
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    from .direct_scatter_backend import direct_enabled, NoPayloadAllocator, install_engine_hooks, find_backend
    from lmcache.v1.cache_engine import LMCacheEngine
    if getattr(connectors, "_daos_direct_launch", False):
        return
    original = connectors.need_gpu_interm_buffer
    connectors.need_gpu_interm_buffer = lambda config: False if direct_enabled(config) else original(config)
    original_allocator = LocalCPUBackend.initialize_allocator

    @functools.wraps(original_allocator)
    def allocator(self, config, metadata=None):
        if direct_enabled(config):
            if config.local_cpu:
                raise ValueError("Direct scatter requires local_cpu=false")
            return NoPayloadAllocator()
        return original_allocator(self, config, metadata)

    LocalCPUBackend.initialize_allocator = allocator
    original_post_init = LMCacheEngine.post_init

    @functools.wraps(original_post_init)
    def post_init(engine, **kwargs):
        result = original_post_init(engine, **kwargs)
        if direct_enabled(engine.config):
            # The plugin loader catches constructor exceptions. Turn a missing
            # plugin into an explicit initialization failure, never a baseline
            # fallback that would invalidate the experiment.
            find_backend(engine)
            connector = engine.gpu_connector
            if connector is not None and getattr(connector, "gpu_buffer", None) is not None:
                raise RuntimeError("Connector staging was allocated before direct-scatter initialization")
        return result

    LMCacheEngine.post_init = post_init
    install_engine_hooks()
    connectors._daos_direct_launch = True


def main():
    import os
    import runpy
    import sys
    # vLLM workers use spawn. Propagate the initializer through a dedicated
    # sitecustomize directory, so child workers install the same hooks.
    from pathlib import Path
    bootstrap = str(Path(__file__).resolve().parent.parent / "experiment_plugins" / "direct_scatter")
    paths = [bootstrap, str(Path(__file__).resolve().parent.parent)]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    os.environ["DAOS_DIRECT_SCATTER_BOOTSTRAP"] = "1"
    install()
    sys.argv[0] = "vllm"
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")


if __name__ == "__main__":
    main()
