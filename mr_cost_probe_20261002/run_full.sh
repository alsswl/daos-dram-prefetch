#!/usr/bin/env bash
set -euo pipefail
cd /root/discos_minji
exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DAOSGDS_TRANSPORT=object DAOSGDR_TIMING=0 LD_PRELOAD=/root/discos_minji/bench/lock_cost_probe.so DAOS_COST_LIB=/root/discos_minji/bench/lock_cost_probe.so DAOS_LOCK_COST_LOG=/root/discos_minji/mr_cost_probe_20261002/full_locks.jsonl ./run_vllm.sh venv/bin/python -u bench/profile_mr_cost.py --workers 16 64 --repeats 2 --output /root/discos_minji/mr_cost_probe_20261002/full > /root/discos_minji/mr_cost_probe_20261002/full.log 2>&1
