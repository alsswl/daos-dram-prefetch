#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/root/discos_minji${PYTHONPATH:+:$PYTHONPATH}"
transport="${DAOSGDS_TRANSPORT:-dfs}"
case "$transport" in
  dfs|object)
    # Keep the DAOS/OFI stack identical; only the storage API changes.
    daos_prefix=/opt/daos-gds-gpu
    fabric_lib=/opt/ofi-cuda/lib64
    ;;
  *)
    echo "DAOSGDS_TRANSPORT must be 'dfs' or 'object' (got '$transport')" >&2
    exit 2
    ;;
esac
export DAOSGDS_TRANSPORT="$transport"
export PATH="$daos_prefix/bin:$PATH"
export LD_LIBRARY_PATH="$fabric_lib:$daos_prefix/lib64:$daos_prefix/prereq/release/mercury/lib64:/usr/local/cuda/lib64"
export LMCACHE_DAOS_LIBDIR="$daos_prefix/lib64"
export D_MEM_DEVICE=1
export D_GPU_DIRECT=1
export DAOSGDR_LIB=/root/discos_minji/libdaosgdr.so
export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/root/discos_minji/lmcache_config_daosgds_unified.yaml}"
export HF_HOME=/home/hf/hf_cache
export PYTHONHASHSEED=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export DAOSGDR_TIMING="${DAOSGDR_TIMING:-1}"
exec "$@"
