#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DAOSGDS_TRANSPORT=object DAOSGDR_TIMING=0
exec /root/discos_minji/run_vllm.sh /root/discos_minji/venv/bin/python -u "$experiment_dir/target_spread.py" "$@"
