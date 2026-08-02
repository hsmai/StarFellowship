#!/bin/bash
# STEP 1: 환경 구축 — 단계별 .done 마커로 재개 가능
# 중간에 끊겨도 재실행하면 완료된 단계는 건너뜀
ROOT=$HOME/task3
MK=$ROOT/markers; LOG=$ROOT/logs/step1_env.log
mkdir -p $MK $ROOT/logs $ROOT/models
log(){ echo "[$(date '+%F %T')] $1" | tee -a $LOG; }
done_ok(){ [ -f "$MK/$1.done" ]; }
mark(){ touch "$MK/$1.done"; log "  -> $1 완료"; }

source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source $(conda info --base)/etc/profile.d/conda.sh

# --- 1a. conda 환경 ---
if ! done_ok 1a_conda; then
  log "[1a] conda 환경 생성 (python 3.10)"
  conda create -y -n task3 python=3.10 >>$LOG 2>&1 && mark 1a_conda || log "  !! conda 실패"
fi
conda activate task3 || { log "!! 환경 활성화 실패, 중단"; exit 1; }
log "python: $(which python) $(python --version 2>&1)"

# --- 1b. PyTorch ---
if ! done_ok 1b_torch; then
  log "[1b] PyTorch (cu121)"
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 >>$LOG 2>&1 \
    && python -c "import torch;assert torch.cuda.is_available()" >>$LOG 2>&1 && mark 1b_torch || log "  !! torch 실패"
fi

# --- 1c. 공통 라이브러리 ---
if ! done_ok 1c_libs; then
  log "[1c] 공통 라이브러리"
  pip install -q numpy opencv-python-headless scipy pandas pyarrow h5py matplotlib timm einops \
      transformers addict yapf pycocotools supervision >>$LOG 2>&1 && mark 1c_libs || log "  !! libs 실패"
fi

# --- 1d. GroundingDINO ---
if ! done_ok 1d_gdino; then
  log "[1d] GroundingDINO"
  cd $ROOT/models
  [ -d GroundingDINO ] || git clone -q https://github.com/IDEA-Research/GroundingDINO.git >>$LOG 2>&1
  cd GroundingDINO && pip install -q -e . >>$LOG 2>&1
  mkdir -p weights
  [ -f weights/groundingdino_swint_ogc.pth ] || wget -q -P weights \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth >>$LOG 2>&1
  [ -f weights/groundingdino_swint_ogc.pth ] && mark 1d_gdino || log "  !! gdino 실패"
fi

# --- 1e. SAM 2.1 ---
if ! done_ok 1e_sam2; then
  log "[1e] SAM 2.1"
  cd $ROOT/models
  [ -d sam2 ] || git clone -q https://github.com/facebookresearch/sam2.git >>$LOG 2>&1
  cd sam2 && pip install -q -e . >>$LOG 2>&1
  mkdir -p checkpoints && cd checkpoints
  [ -f sam2.1_hiera_large.pt ] || wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt >>$LOG 2>&1
  [ -f sam2.1_hiera_large.pt ] && mark 1e_sam2 || log "  !! sam2 실패"
fi

# --- 1f. UniDepthV2 ---
if ! done_ok 1f_unidepth; then
  log "[1f] UniDepthV2"
  cd $ROOT/models
  [ -d UniDepth ] || git clone -q https://github.com/lpiccinelli-eth/UniDepth.git >>$LOG 2>&1
  cd UniDepth && pip install -q -e . >>$LOG 2>&1 && mark 1f_unidepth || log "  !! unidepth 실패(HF 자동 로드로 대체 가능)"
fi

# --- 1g. WildCamera (선택) ---
if ! done_ok 1g_wildcam; then
  log "[1g] WildCamera"
  cd $ROOT/models
  [ -d WildCamera ] || git clone -q https://github.com/ShngJZ/WildCamera.git >>$LOG 2>&1
  [ -d WildCamera ] && mark 1g_wildcam || log "  !! wildcamera 실패(FOV 가정으로 대체 가능)"
fi

log "[1] 환경 구축 종료. 마커: $(ls $MK 2>/dev/null | tr '\n' ' ')"
