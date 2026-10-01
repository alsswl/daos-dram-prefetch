#!/usr/bin/env bash
set -euo pipefail
set -C
cd /root/discos_minji
unset DAOS_DIRECT_SCATTER_BOOTSTRAP
export HF_HOME=/home/hf/hf_cache
export PYTHONPATH=/root/discos_minji/experiment_plugins:/root/discos_minji
export VLLM_PLUGINS=daos_resume_tokens
export DAOSGDR_TIMING=0
exec /root/discos_minji/venv/bin/python realqa_split_experiment.py run \
  --output /root/discos_minji/realqa_q4_64k_split_r1536_s512_w500_20261001 \
  > /root/discos_minji/realqa_q4_64k_split_r1536_s512_w500_20261001/runner.log 2>&1
