#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/root/discos${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/discos-daos-gdr/prereq/release/ofi/lib64:/opt/discos-daos-gdr/lib64:/usr/local/cuda/lib64"
export D_GPU_DIRECT=1
export DAOSGDR_LIB=/root/discos/libdaosgdr.so
export LMCACHE_CONFIG_FILE=/root/discos/lmcache_config_daosgdr.yaml
export HF_HOME=/home/hf/hf_cache
export PYTHONHASHSEED=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export DAOSGDR_TIMING=1
exec "$@"
