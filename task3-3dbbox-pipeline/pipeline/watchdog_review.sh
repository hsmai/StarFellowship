#!/bin/bash
# 검증 러너 워치독 — 사용법: watchdog_review.sh <bc|he> <라운드> <샤드idx> <샤드수> <GPU>
KIND=$1; RND=$2; SH=$3; NSH=$4; GPU=$5
ROOT=$HOME/task3
LOG=$ROOT/v2/logs/wd_${KIND}${SH}.log
mkdir -p $ROOT/v2/logs
source $(conda info --base)/etc/profile.d/conda.sh
conda activate task3
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$ROOT/pipeline
echo "[$(date '+%F %T')] ===== $KIND shard$SH/$NSH GPU$GPU 라운드 $RND =====" >> $LOG
for i in 1 2 3 4; do
  python $ROOT/pipeline/run_review.py $KIND $RND $SH $NSH >> $LOG 2>&1
  RC=$?
  echo "[$(date '+%F %T')] 시도 $i 종료코드 $RC" >> $LOG
  [ $RC -eq 0 ] && break
  sleep 10
done
echo "[$(date '+%F %T')] 종료" >> $LOG
