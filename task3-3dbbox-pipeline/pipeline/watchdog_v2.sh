#!/bin/bash
# V2 워치독 — 사용법: watchdog_v2.sh <bc|he> <물리GPU번호>
# tmux 안에서 실행. 실패 시 최대 6회 재시작, 마커로 완료분은 건너뜀.
KIND=$1; GPU=$2
ROOT=$HOME/task3
LOG=$ROOT/logs/watchdog_v2_$KIND.log
mkdir -p $ROOT/logs $ROOT/markers $ROOT/results_v2
echo "[$(date '+%F %T')] ===== V2 $KIND 워치독 시작 (GPU $GPU) =====" >> $LOG

source $(conda info --base)/etc/profile.d/conda.sh
conda activate task3
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$ROOT/pipeline

if [ "$KIND" = "bc" ]; then
  SCRIPT=run_brainco_v2.py; PREFIX=V2_BC; N=8
else
  SCRIPT=run_he_v2.py; PREFIX=V2_HE; N=7
fi

for i in $(seq 1 6); do
  DONE=$(ls $ROOT/markers/${PREFIX}_*.done 2>/dev/null | wc -l)
  if [ "$DONE" -ge "$N" ]; then break; fi
  echo "[$(date '+%F %T')] 시도 $i (완료 $DONE/$N)" >> $LOG
  python $ROOT/pipeline/$SCRIPT >> $LOG 2>&1
  RC=$?
  echo "[$(date '+%F %T')] 종료코드 $RC" >> $LOG
  [ $RC -eq 0 ] && break
  sleep 20
done

DONE=$(ls $ROOT/markers/${PREFIX}_*.done 2>/dev/null | wc -l)
if [ "$DONE" -ge "$N" ]; then
  touch $ROOT/markers/${PREFIX}_ALL_FINISHED
  echo "[$(date '+%F %T')] $KIND 전부 완료 ($DONE/$N)" >> $LOG
else
  echo "[$(date '+%F %T')] $KIND 미완료 ($DONE/$N) — 재시도 소진" >> $LOG
fi
