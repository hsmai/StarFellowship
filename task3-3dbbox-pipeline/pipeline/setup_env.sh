#!/bin/bash
# STEP 1 환경 구축 스크립트 (GPU 승인 후 실행)
# 서버 기본 python 3.9라 SAM2(>=3.10) 불가 -> conda 별도 환경 필수
set -e
ENV=task3
echo "[1/5] conda 환경 생성 (python 3.10)"
conda create -y -n $ENV python=3.10
source activate $ENV

echo "[2/5] PyTorch (CUDA 12.1 빌드 — 서버 CUDA 12.3 호환)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[3/5] 공통 라이브러리"
pip install numpy opencv-python scipy pandas pyarrow h5py matplotlib open3d timm einops

echo "[4/5] 모델 4종"
mkdir -p ~/task3/models && cd ~/task3/models
# GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e . && mkdir -p weights && \
  wget -q -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth && cd ..
# SAM 2.1
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd checkpoints && bash download_ckpts.sh && cd ../..
# UniDepth V2
git clone https://github.com/lpiccinelli-eth/UniDepth.git
cd UniDepth && pip install -e . && cd ..
# WildCamera
git clone https://github.com/ShngJZ/WildCamera.git

echo "[5/5] 스모크 테스트 (각 모델 단독 추론 + VRAM 측정)"
python ~/task3/pipeline/smoke_test.py

echo "완료. conda activate $ENV 로 사용"
