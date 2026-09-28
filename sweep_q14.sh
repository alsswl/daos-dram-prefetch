#!/usr/bin/env bash
set -uo pipefail

OUT=~/discos/q14_$(date +%m%d_%H%M)
mkdir -p "$OUT"
ROOT=~/discos/discoverybench/discoverybench/synth/train
MODEL=Qwen/Qwen3-14B
PY=~/discos/venv/bin/python3
E=/opt/discos-daos-gdr/prereq/release/ofi/lib64:/opt/discos-daos-gdr/lib64:/usr/local/cuda/lib64
DAOS=/opt/discos-daos-gdr/bin/daos

declare -A PAD=( [256]=300  [512]=800  [1024]=1500 [2048]=3000 )
declare -A MML=( [256]=2048 [512]=2048 [1024]=4096 [2048]=8192 )
declare -A KVMB=( [256]=40  [512]=80   [1024]=160  [2048]=320  )

CHUNKS=(256 512 1024 2048)
[[ $# -ge 1 ]] && CHUNKS=("$@")

for chunk in "${CHUNKS[@]}"; do
  echo "=== chunk $chunk (KV ${KVMB[$chunk]}MB, pad ${PAD[$chunk]}) ==="

  # 컨테이너 초기화 — run1이 깨끗한 상태에서 시작
  env LD_LIBRARY_PATH=$E $DAOS cont destroy discospool gdrcont 2>/dev/null
  env LD_LIBRARY_PATH=$E $DAOS cont create discospool gdrcont --properties rd_fac:0 >/dev/null || { echo "  cont create FAILED"; exit 1; }

  sed -i "s/^chunk_size:.*/chunk_size: $chunk/" ~/discos/lmcache_config_daosgdr.yaml

  for run in run1 run2; do
    tag="c${chunk}_${run}"
    echo "  [$run]"
    LMCACHE_LOG_LEVEL=DEBUG ~/discos/run_vllm.sh $PY ~/discos/kv_measure.py \
        --root $ROOT --model $MODEL \
        --max-model-len ${MML[$chunk]} --tasks 1 --steps 6 \
        --chunk-size $chunk --pad-tokens ${PAD[$chunk]} --no-prefix-cache \
        --out "$OUT/D_GDR_${tag}.csv" > "$OUT/D_GDR_${tag}.log" 2>&1
    echo "      Retrieved=$(grep -c Retrieved "$OUT/D_GDR_${tag}.log")"
    sleep 3
  done
  echo
done

echo "출력: $OUT"
echo "  $PY ~/discos/summarize_bench.py $OUT"
