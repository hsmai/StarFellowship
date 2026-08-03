#!/bin/bash
# 큐 워커 워치독 — 사용법: watchdog_queue.sh <worker_id> <물리GPU번호>
# 워커가 죽어도 자동 재시작. 완료분은 done 마커로 건너뛰므로 끊긴 지점부터 이어서 진행된다.
WID=$1; GPU=$2; MODE=$3   # MODE=rep 이면 대표 15건만
ROOT=$HOME/task3
V2=$ROOT/v2
LOG=$V2/logs/watchdog_$WID.log
mkdir -p $V2/logs

source $(conda info --base)/etc/profile.d/conda.sh
conda activate task3
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$ROOT/pipeline

echo "[$(date '+%F %T')] ===== 워커 $WID 시작 (GPU $GPU) =====" >> $LOG

if [ "$MODE" = "rep" ]; then TOTAL=15; ARG="--rep"; else TOTAL=$(wc -l < $V2/queue.jsonl); ARG=""; fi
for i in $(seq 1 200); do
  if [ "$MODE" = "rep" ] && [ "$i" -gt 3 ]; then break; fi
  DONE=$(ls $V2/done/ 2>/dev/null | wc -l)
  if [ "$DONE" -ge "$TOTAL" ]; then
    echo "[$(date '+%F %T')] 큐 전부 완료 ($DONE/$TOTAL)" >> $LOG
    touch $V2/QUEUE_ALL_FINISHED
    break
  fi
  echo "[$(date '+%F %T')] 시도 $i — 진행 $DONE/$TOTAL" >> $LOG
  python $ROOT/pipeline/queue_runner.py work $WID $ARG >> $LOG 2>&1
  RC=$?
  echo "[$(date '+%F %T')] 워커 종료코드 $RC (진행 $(ls $V2/done/ 2>/dev/null | wc -l)/$TOTAL)" >> $LOG
  # 정상 종료(큐 소진)면 남은 작업이 있는지 한 번 더 확인 후 루프
  sleep 15
done
echo "[$(date '+%F %T')] 워치독 $WID 종료 (완료 $(ls $V2/done/ 2>/dev/null | wc -l)/$TOTAL)" >> $LOG
