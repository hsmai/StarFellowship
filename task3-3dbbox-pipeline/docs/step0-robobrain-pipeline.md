# STEP 0 — RoboBrain 2.0 / 2.5 파이프라인 조사 결과 (2026-08-02)

> 목적: 사수님 지시대로 RoboBrain Technical Report에서 Pseudo-3D Object Annotation Pipeline의 구체 구성을 파악하고, 우리(G1 데이터) 적용 시 무엇을 그대로 쓰고 무엇을 바꿔야 하는지 확정한다.

## 1. 파이프라인의 출처 계보 (조사로 확인)

```
RoboRefer / RefSpatial (arXiv 2506.04308, NeurIPS 2025)   ← 파이프라인의 원류
        │  "pseudo-3D scene graph" 구축 파이프라인 설계
        ▼
RoboBrain 2.0 Technical Report (arXiv 2507.02029)          ← RefSpatial 파이프라인을 부분 채택
        │  OpenImage 1.7M→466K 처리, 826K 샘플 생성
        ▼
RoboBrain 2.5 Technical Report (arXiv 2601.14352)          ← 동일 파이프라인 유지 + 3D 능력 확장
           "2D→3D, 상대→절대(메트릭), 점→궤적" + 실로봇 데이터(AgiBot-Beta, DROID) 추가 큐레이션
```

- 사수님이 주신 그림의 6단계(GroundingDINO → SAM 2.1 → UniDepthV2+WildCamera → Back-projection → 3D Box)는 **RoboBrain 2.0 §스페이셜 데이터 구축 절의 구성과 정확히 일치**함을 원문 인용으로 확인.
- RoboBrain 2.5에서도 같은 도구 조합("RAM for object category prediction and GroundingDINO for 2D boxes... UniDepth V2 and WildeCamera for depth and camera intrinsics... masks from SAM 2.1... axis-aligned 3D boxes")을 사용.

## 2. 원 파이프라인의 정확한 절차 (리포트 인용 기반)

| 순서 | 원 파이프라인 (웹 이미지용) | 산출물 |
|---|---|---|
| 0 | OpenImage 1.7M장 → 필터링(SigLIP2 코사인 유사도 → Qwen2.5-VL-7B 품질 검사) → 466K장 | 입력 이미지 |
| 1 | **RAM**: 이미지에 어떤 객체가 있는지 카테고리 예측 | 객체 이름 목록 |
| 2 | **GroundingDINO**: RAM이 준 이름을 텍스트 프롬프트로 2D 박스 검출 | 2D bounding box |
| 3 | **SAM 2.1**: 박스를 프롬프트로 인스턴스 마스크 생성 | 객체별 마스크 |
| 4 | **UniDepth V2**: 메트릭(절대 거리) depth 추정 / **WildCamera**: 카메라 intrinsics 복원 | depth 맵 + 렌즈 정보 |
| 5 | **Back-projection**: 마스크 영역 픽셀을 depth+intrinsics로 3D 점군 복원 | 객체별 point cloud |
| 6 | 점군에서 **axis-aligned 3D box** 산출 | 3D bounding box |
| +α | Qwen2.5-VL로 계층적 캡션 생성 → pseudo-3D scene graph (라벨·박스·마스크·점군·공간관계 엣지) | QA 데이터 |

## 3. 우리 적용과의 차이 (G1 로봇 데이터에 맞춘 수정)

| 항목 | 원본 (웹 이미지) | 우리 (Brainco / Humanoid Everyday) |
|---|---|---|
| 입력 | 서로 무관한 웹 사진 | 30fps 연속 로봇 영상 → **SAM 2.1의 비디오 추적 기능을 실제로 활용** (원본은 프레임 단독) |
| 객체 이름(1단계 RAM) | 필요 (뭐가 있는지 모름) | **생략 가능** — 태스크 라벨로 조작 객체를 이미 앎 (예: "Oreo", "plate"). 배경 객체까지 원하면 RAM 추가 |
| intrinsics | WildCamera로 추정 | 동일하게 추정 (두 데이터셋 모두 메타에 intrinsics 없음 — EDA에서 확인) |
| depth | UniDepthV2 추정 | Brainco: 추정만 가능 / **HE: 실측 depth 보유 → 추정치를 검증할 수 있는 유일한 수단** |
| 검증 | 없음 (대규모 자동 주석) | HE 실측 depth 비교 + 시각화 눈 확인 |

## 4. 리포트에 공개되지 않은 세부 → 우리 구현 방침

논문·코드 조사 결과 **데이터 생성 파이프라인 코드는 미공개** (RoboRefer 레포 TODO: "Release the Dataset Generation Pipeline (Maybe 2 months or more)"). 따라서 아래 세부는 표준 관행으로 구현하고 HE 검증으로 보정한다:

| 미공개 항목 | 우리 방침 |
|---|---|
| GroundingDINO 신뢰도 임계값 | 기본값(box 0.35 / text 0.25)에서 시작, 시각화로 조정 |
| 마스크 경계 노이즈 처리 | 마스크 erosion(2~3px)으로 경계 depth 혼입 방지 |
| 점군 이상치 제거 | depth 5~95 percentile 클리핑 + statistical outlier removal(Open3D) |
| 3D 박스 산출 | 정제된 점군의 axis-aligned min/max (리포트 명시 "axis-aligned") |
| 좌표계 | 카메라 좌표계 기준 (원본과 동일 — 월드 정렬은 추후 확장) |

## 5. 구성 모델 4종 스펙 (STEP 1 준비물)

| 모델 | 선택 체크포인트 | 역할 | VRAM(추론) | 저장소 |
|---|---|---|---|---|
| GroundingDINO | SwinT-OGC (694MB) | open-vocab 2D 검출 | ~4GB | IDEA-Research/GroundingDINO |
| SAM 2.1 | hiera-large (856MB) | 마스크 + 비디오 추적 | ~6-8GB | facebookresearch/sam2 |
| UniDepth V2 | ViT-L (~1.3GB) | 메트릭 depth | ~6-10GB | lpiccinelli-eth/UniDepth |
| WildCamera | 공개 ckpt (~500MB) | intrinsics 복원 | ~2-4GB | ShngJZ/WildCamera |

- 합계 체크포인트 ~3.5GB + 의존성 → 홈 디렉토리 저장 (/data2 부담 없음)
- 요구 환경: Python ≥3.10 (SAM2), PyTorch CUDA 12.x — 서버 CUDA 12.3와 호환. 서버 기본 python 3.9라 **conda 별도 환경 필수**

## 6. 대상 데이터 (STEP 0.5에서 프레임 추출 예정)

| 트랙 | 에피소드 | 카메라 | 프롬프트 (태스크 라벨 유래) |
|---|---|---|---|
| Brainco | GraspOreo ep5 (29.5초, 이미 클립 확보) | cam_left_high | "oreo cookie box. plate." |
| HE | ep3800 (put_dumpling, 13.1초, 클립 확보) | egocentric | "pink toy. orange plate." + 실측 depth 비교 |

## 출처

- [RoboBrain 2.0 Technical Report (arXiv 2507.02029)](https://arxiv.org/abs/2507.02029)
- [RoboBrain 2.5 Technical Report (arXiv 2601.14352)](https://arxiv.org/abs/2601.14352) · [GitHub FlagOpen/RoboBrain2.5](https://github.com/FlagOpen/RoboBrain2.5)
- [RoboRefer / RefSpatial (arXiv 2506.04308)](https://arxiv.org/abs/2506.04308) · [GitHub Zhoues/RoboRefer](https://github.com/Zhoues/RoboRefer)
