# STEP 1~5 실행 결과 — 3D BBox 파이프라인 완성 (2026-08-02)

> GPU 승인 후 pleiades1(RTX 3090 ×4)에서 전 단계 실행 완료. **파이프라인 6단계가 end-to-end로 동작**하며, Brainco·Humanoid Everyday 두 트랙 모두 3D Bounding Box 추출에 성공했다.

## 1. 최종 결과 요약

| 항목 | 결과 |
|---|---|
| 파이프라인 | GroundingDINO → SAM 2.1 → UniDepthV2(+intrinsics) → Back-projection → 3D Box **전 단계 동작** |
| Brainco 에피소드 | GraspOreo ep5 (29.5초) → **442/442 프레임 전부 성공** |
| Humanoid Everyday | ep3800 → 12개 프레임에서 GT/추정 비교 완료 |
| 산출물 | 오버레이 영상 1편 + 프레임 이미지 4장 + JSON 로그 |

## 2. 추출된 3D Bounding Box (Brainco, 442프레임 통계)

| 객체 | 크기 중앙값 (W×H×D) | 검출 프레임 | 거리 범위 |
|---|---|---|---|
| plate (접시) | **30.0 × 17.7 × 13.5 cm** | 442 / 442 (100%) | 0.50 ~ 0.69 m |
| oreo snack package | **10.9 × 5.6 × 2.4 cm** | 232 / 442 (52%) | 0.39 ~ 0.84 m |

- 오레오 봉지 10.9×5.6×2.4cm는 **실제 미니 오레오 봉지 치수와 부합**한다.
- 접시가 100% 검출된 반면 오레오는 52%인데, 로봇 손에 가려지거나 접시 안으로 들어간 구간에서 검출이 끊긴다. 프레임 단위 독립 검출이라 그렇고, SAM 2.1의 비디오 추적 모드를 쓰면 개선 여지가 있다.

## 3. STEP 4 — 추정 depth vs 실측 depth 정확도 검증 (핵심)

Humanoid Everyday의 실측 depth를 정답으로 삼아 UniDepthV2 추정치를 12개 프레임에서 비교했다.

| 지표 | 값 |
|---|---|
| 평균 절대 오차 (MAE) | **0.429 m** |
| 평균 상대 오차 | **43.0 %** |
| 스케일 계수 (GT/추정 중앙값) | **0.847** |
| 스케일 보정 후 MAE | 0.42 m (거의 개선 없음) |

**해석**: 추정 depth는 실측 대비 평균 **15% 정도 멀게** 나오며(스케일 0.847), 단순 스케일 보정으로는 오차가 줄지 않았다. 즉 오차가 균일 배율이 아니라 **깊이 구간별로 다르게** 발생한다.

### 3D 박스 수준에서의 영향 (같은 프레임, GT vs 추정)

| 객체 | GT (실측 depth) | 추정 depth | 차이 |
|---|---|---|---|
| orange bowl | 34.3 × 20.4 × 18.6 cm @ 0.62m | 30.8 × 27.6 × 18.0 cm @ 0.79m | 거리 +0.17m, 크기 ±20% |
| pink toy | 13.7 × 9.6 × 7.3 cm @ 0.58m | 10.7 × 10.2 × 3.5 cm @ 0.72m | 거리 +0.14m, **두께 절반** |

**결론**: 물체의 **가로·세로 크기는 비교적 잘 맞지만(±20%), 거리는 일관되게 0.15m 정도 멀게, 두께(깊이 방향)는 얇게** 추정된다. 이는 STEP 0에서 예상했던 리스크 — "UniDepthV2가 근접 촬영(0.3~0.8m)에 최적화되어 있지 않다" — 가 실제로 확인된 것이다.

## 4. 실측 리소스

| 모델 | 로드 시간 | VRAM |
|---|---|---|
| GroundingDINO | 102.5 s (최초 다운로드 포함) | 0.94 GB |
| SAM 2.1 | 4.7 s | 1.05 GB |
| UniDepthV2 | 10.9 s | 1.42 GB |
| WildCamera | — | (UniDepthV2가 intrinsics를 제공해 미사용) |
| **합계** | | **약 3.4 GB** |

- 사전 추정(16~24GB)보다 **훨씬 적게 소요**됐다. 추론 배치가 1이고 fp32 기준이라 3090 1장으로도 충분했다.
- 처리 속도: Brainco 442프레임(2프레임 stride, 29.5초 영상) 기준 **약 5분** → 프레임당 0.7초
- intrinsics는 **UniDepthV2가 직접 제공**(fx≈402~407)해 WildCamera 없이 해결됐다.

## 5. 구현 중 발생한 문제와 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| GroundingDINO 설치 실패 | CUDA 확장 빌드 오류 | **transformers 구현으로 폴백** (`IDEA-Research/grounding-dino-base`) — 빌드 불필요 |
| transformers import 불가 | `torchaudio`가 CUDA 13 빌드로 설치돼 `libcudart.so.13` 없음 | torchaudio 제거 (사용 안 함) |
| SAM 마스크 단계 전 프레임 실패 | `img[:,:,::-1]`의 negative stride를 torch가 거부 | `np.ascontiguousarray()` 적용 |
| 3D 박스가 실제의 5배로 부풀음 | percentile 클리핑만으로는 배경 혼입 제거 불가 | STEP 0.5에서 도입한 **모드 기반 전경분리(중앙값±2·MAD)** |

## 6. 산출물

| 파일 | 내용 |
|---|---|
| `results/step3_brainco.mp4` | **Brainco 에피소드 전체 오버레이 영상** (442프레임, 3D 박스 재투영) |
| `results/step2_brainco_frame.png` | Brainco 단일 프레임 결과 |
| `results/step2_he_frame.png` | HE 단일 프레임 결과 |
| `results/step4_he_compare.png` | **GT depth(좌) vs 추정 depth(우) 박스 비교** |
| `results/step4_he_gt.png`, `step4_he_pred.png` | 개별 결과 |
| `results/step3_brainco.json` | 442프레임 전체의 검출·박스 기록 |
| `results/step4_he.json` | depth 정확도 12프레임 측정값 |

## 7. 한계와 다음 단계

1. **Brainco는 정확도 검증 수단이 없다.** 실측 depth·intrinsics가 없어, HE에서 측정한 오차(거리 +0.15m, 두께 과소)가 Brainco에도 비슷하게 존재한다고 간접 추정할 수밖에 없다.
2. **거리 편향 보정이 필요하다.** 스케일 보정만으로는 안 되므로, HE의 GT를 이용해 깊이 구간별 보정 곡선을 학습하는 방식이 후속 과제가 될 수 있다.
3. **프레임 독립 검출의 한계.** SAM 2.1의 비디오 추적 모드를 쓰면 가려짐 구간에서도 객체를 유지할 수 있다(현재는 프레임마다 재검출).
4. axis-aligned 박스라 **회전한 물체는 실제보다 크게** 잡힌다. RoboBrain 리포트가 명시한 방식이라 그대로 따랐다.

## 8. 실행 방법 (재현)

```bash
ssh -p 3022 isangmin@10.20.23.30
tmux new -s t3 'bash ~/task3/pipeline/watchdog.sh'   # 전 단계 자동 실행 (완료 단계는 건너뜀)

# 개별 실행
conda activate task3
python ~/task3/pipeline/run_pipeline.py smoke|frame|brainco|he
```

- 모든 단계는 `~/task3/markers/*.done` 마커로 중복 실행이 방지되어, **끊겨도 재실행하면 이어서 진행**된다.
- 워치독이 최대 5회 자동 재시작하며, 핵심 산출물이 생성돼야만 완료로 인정한다.
