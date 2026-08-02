#!/bin/bash
# Task 3 전체 오케스트레이터 — 환경구축 완료를 기다렸다가 파이프라인 자동 실행
# 각 단계는 .done 마커로 재개 가능. 실패해도 다음 단계 시도.
ROOT=$HOME/task3
LOG=$ROOT/logs/run_all.log
mkdir -p $ROOT/logs $ROOT/markers $ROOT/results
log(){ echo "[$(date '+%F %T')] $1" | tee -a $LOG; }

log "=== Task3 오케스트레이터 시작 ==="

# 1) 환경 구축 (재실행해도 완료분은 건너뜀)
log "[1/5] 환경 구축"
bash $ROOT/pipeline/run_step1_env.sh >> $LOG 2>&1
log "  마커: $(ls $ROOT/markers | tr '\n' ' ')"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate task3 || { log "!! conda 활성화 실패"; exit 1; }
cd $ROOT

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$ROOT/pipeline:$PYTHONPATH

# 2) 스모크 테스트
log "[2/5] 모델 스모크 테스트"
python $ROOT/pipeline/run_pipeline.py smoke >> $LOG 2>&1 || log "  !! smoke 일부 실패 (계속 진행)"

# 3) 프레임 1장 관통
log "[3/5] STEP2 프레임 관통"
python $ROOT/pipeline/run_pipeline.py frame >> $LOG 2>&1 || log "  !! frame 실패"

# 4) Brainco 에피소드
log "[4/5] STEP3 Brainco 에피소드"
python $ROOT/pipeline/run_pipeline.py brainco >> $LOG 2>&1 || log "  !! brainco 실패"

# 5) HE 검증
log "[5/5] STEP4 HE depth 검증"
python $ROOT/pipeline/run_pipeline.py he >> $LOG 2>&1 || log "  !! he 실패"

log "=== 전체 종료 ==="
log "결과: $(ls $ROOT/results 2>/dev/null | tr '\n' ' ')"
touch $ROOT/markers/ALL_FINISHED
