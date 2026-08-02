# Task 3 — Unitree G1 데이터 3D Bounding Box 추출 파이프라인

> RoboBrain의 Pseudo-3D Object Annotation Pipeline을 재현해, 정답 라벨이 없는 real-world 로봇 데이터(Brainco·Humanoid Everyday)에서 객체 3D Bounding Box를 추출한다.
> 완료 기준: **에피소드 1개의 추출 결과를 눈으로 확인 가능한 시각화**.
>
> **✅ 2026-08-02 완료** — Brainco 442프레임 전부 성공, HE depth 정확도 검증 완료. 결과: [docs/step1-5-results.md](docs/step1-5-results.md)

## 파이프라인 (6단계)

| 단계 | 모듈 | 역할 |
|---|---|---|
| 0 | RGB 프레임 | 입력 (G1 Brainco 4뷰 / Humanoid Everyday / AgiBot World) |
| 1 | GroundingDINO | 텍스트 프롬프트 → 객체 2D bounding box |
| 2 | SAM 2.1 | 박스 프롬프트 → 인스턴스 마스크 + 프레임 간 추적 |
| 3 | UniDepthV2 + WildCamera | 메트릭 depth 추정 + 카메라 intrinsics 복원 |
| 4 | Back-projection | 마스크 + depth → 객체별 point cloud |
| 5 | 3D Bounding Box | 점군에서 axis-aligned 박스 산출 |

## 진행 상태

- [x] STEP 0 — RoboBrain 2.0/2.5 리포트 파이프라인 파악 완료 (8/2) → [docs/step0-robobrain-pipeline.md](docs/step0-robobrain-pipeline.md)
- [x] STEP 0.5 — **GPU 불필요 구간 완료 (8/2)** → [docs/step05-cpu-verification.md](docs/step05-cpu-verification.md)
  - 4~5단계(역투영·3D박스) 구현 및 실측 depth로 검증 완료 → `pipeline/geometry.py`
  - 발견: RGB-depth 20px 정렬 오차 / percentile만으로는 박스가 5배 부풀어 모드 기반 전경분리 필수
- [x] STEP 1 — 환경 구축 + 스모크 테스트 완료 (8/2) — 실측 VRAM 총 3.4GB
- [x] STEP 2 — 프레임 1장 관통 성공 (Brainco·HE 양쪽)
- [x] STEP 3 — Brainco 442프레임 전부 성공 → `results/step3_brainco.mp4`
- [x] STEP 4 — HE depth 검증 완료 (MAE 0.43m, 상대오차 43%)
- [x] STEP 5 — 문서화 완료 → [docs/step1-5-results.md](docs/step1-5-results.md)

## 리소스 (실측)

- GPU: pleiades1 RTX 3090 × 4장 · CUDA 12.3 · **실사용 VRAM 3.4GB** (사전 추정 16~24GB보다 훨씬 적음)
- 처리 속도: 프레임당 0.7초 (442프레임 약 5분)
- 대상 데이터: `/data2/humanoid_dataset_isangmin/{G1_Brainco_*, humanoid-everyday}`

## 폴더 구조

- `docs/` — 조사·설계·결과 문서
- `pipeline/` — 파이프라인 코드
- `results/` — 시각화 산출물
