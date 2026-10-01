#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
gcc -shared -fPIC -O2 -Wall -Wextra -Werror \
    -isystem /opt/daos-gds-gpu/include "$experiment_dir/placement.c" \
    -L/opt/daos-gds-gpu/lib64 -ldaos -lgurt -o "$experiment_dir/placement.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DAOSGDS_TRANSPORT=object DAOSGDR_TIMING=0
exec /root/discos_minji/run_vllm.sh /root/discos_minji/venv/bin/python -u "$experiment_dir/run.py" "$@"
