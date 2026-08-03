#!/bin/bash
# 심화작업 A~D 워치독 — GPU 1장 직렬 실행, 최대 8회 자동 재시도
# 각 작업은 markers/{A_track,B_agibot,C_multiview,D_tasks}.done 으로 재개 안전
ROOT=$HOME/task3
LOG=$ROOT/logs/watchdog_extra.log
source $(conda info --base)/etc/profile.d/conda.sh
conda activate task3
export CUDA_VISIBLE_DEVICES=2          # 물리 GPU 2번 1장만 사용
export PYTHONPATH=$ROOT/pipeline:$PYTHONPATH
export HF_HUB_DISABLE_TELEMETRY=1

for i in $(seq 1 8); do
  ALL=1
  for m in A_track B_agibot C_multiview D_tasks; do
    [ -f $ROOT/markers/$m.done ] || ALL=0
  done
  if [ $ALL -eq 1 ]; then
    echo "[$(date '+%F %T')] A~D 전부 완료 — 워치독 종료" >> $LOG
    touch $ROOT/markers/EXTRA_ALL_FINISHED
    break
  fi
  echo "[$(date '+%F %T')] 시도 $i/8 — run_extra 실행 (완료분 건너뜀)" >> $LOG
  python $ROOT/pipeline/run_extra.py all >> $LOG 2>&1
  echo "[$(date '+%F %T')] 종료코드 $? | 마커: $(ls $ROOT/markers | grep -E 'A_|B_|C_|D_' | tr '\n' ' ')" >> $LOG
  sleep 30
done
echo "[$(date '+%F %T')] 최종 결과: $(ls $ROOT/results | tr '\n' ' ')" >> $LOG
