"""Explicit opt-in initializer inherited by spawned vLLM workers."""
import os

if os.environ.get("DAOS_DIRECT_SCATTER_BOOTSTRAP") == "1":
    from lmcache_daos.direct_scatter_launch import install
    install()
