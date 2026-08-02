# Task 3 — Unitree G1 데이터 3D Bounding Box 추출 파이프라인

> RoboBrain의 Pseudo-3D Object Annotation Pipeline을 재현해, 정답 라벨이 없는 real-world 로봇 데이터(Brainco·Humanoid Everyday)에서 객체 3D Bounding Box를 추출한다.
> 완료 기준: **에피소드 1개의 추출 결과를 눈으로 확인 가능한 시각화**.

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

- [ ] STEP 0 — RoboBrain 2.0/2.5 Technical Report 파이프라인 파악 → [docs/step0-robobrain-pipeline.md](docs/step0-robobrain-pipeline.md)
- [ ] STEP 0.5 — GPU 불필요 준비 (대상 선정·프레임 추출·코드 골격)
- [ ] STEP 1 — 환경 구축 + 모델 4종 스모크 테스트 (GPU 승인 대기)
- [ ] STEP 2 — 프레임 1장 관통
- [ ] STEP 3 — Brainco 에피소드 1개 → 시각화
- [ ] STEP 4 — HE 에피소드 1개 + depth 정확도 검증
- [ ] STEP 5 — 문서화

## 리소스

- GPU: pleiades1 RTX 3090(24GB) × 4장 (승인 대기 중) · CUDA 12.3
- 대상 데이터: `/data2/humanoid_dataset_isangmin/{G1_Brainco_*, humanoid-everyday}`

## 폴더 구조

- `docs/` — 조사·설계·결과 문서
- `pipeline/` — 파이프라인 코드
- `results/` — 시각화 산출물
