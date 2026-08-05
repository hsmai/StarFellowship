# Task 3 — Unitree G1 데이터셋 3D Bounding Box 추출 파이프라인

RoboBrain의 Pseudo-3D Object Annotation Pipeline(5단계)을 Unitree G1 데이터셋
(**Brainco 8태스크**, **Humanoid Everyday 246태스크**)에 적용해 3D Bounding Box를 생성한다.

**최종 목표**: 다운로드된 전 데이터셋에 3D box를 생성한다. 지금은 그 생성기를
robust하게 만들고 설정 가능하게 정비하는 단계다.

## 문서

| 문서 | 내용 |
|---|---|
| [pipeline-design.md](docs/pipeline-design.md) | **현행 설계** — 5단계 구성, 게이트, 2가지 방식 |
| [settings.md](docs/settings.md) | **세부 설정** — 실행 파라미터, 태스크별 프로파일, GPU 실행 |
| [generalization.md](docs/generalization.md) | **일반화 전략** — 전 데이터 적용 준비와 품질 확인 체계 |
| [improvement-log.md](docs/improvement-log.md) | 개선 이력 — 라운드별 증상·원인·수정과 방법론 교훈 |
| [step0-robobrain-pipeline.md](docs/step0-robobrain-pipeline.md) | RoboBrain 파이프라인 조사 (계보·논문·미공개 항목) |
| [step05-cpu-verification.md](docs/step05-cpu-verification.md) | 4~5단계 CPU 선행 검증 |
| [step1-5-results.md](docs/step1-5-results.md) | 최초 end-to-end 실행 결과와 depth 정확도 검증 |

## 빠른 시작

```bash
# 실행 설정 확인 (프리셋별 예상 소요 시간)
python pipeline/config.py

# 15개 대표 검증 (개선→평가 루프)
qsub -q pleiades1 pipeline/pbs_r4.sh
```

## 검증 결과 (r6, 2026-08-05)

**두 데이터셋의 모든 종류의 태스크 15종을 전부 검증**했다 (태스크당 에피소드 1개, 39 유닛).

| 데이터셋 | 태스크 | 대상 | 방식 A | 방식 B | 추출 크기 |
|---|---|---|---|---|---|
| Brainco | GraspOreo | 오레오 | 100% | 100% | 9×5×2 cm |
| | GraspRubiksCube | 루빅스 큐브 | 100% | 100% | 7×7×3 cm |
| | PickApple | 사과 | 100% | 100% | 7×6×2 cm |
| | PickCharger | 충전기 | 89% | **99%** | 6×5×2 cm |
| | PickDoll | 인형 | 100% | 100% | 25×27×16 cm |
| | PickDrink | 물병 | 100% | 100% | 6×11×5 cm |
| | PickTissues | 티슈 | 100% | 100% | 11×9×7 cm |
| | PickToothpaste | 치약 | 65% | **100%** | 15×4×3 cm |
| HE | Basic | 분홍 인형 | 96% | 100% | 10×9×4 cm |
| | Articulated | 노트북 | 99% | 99% | 47×19×26 cm |
| | deformable | 수건 | 99% | 100% | 35×30×26 cm |
| | HRI | 장미 | 99% | 100% | 29×12×5 cm |
| | Locomanip | 물병 | 81% | 81% | 9×19×11 cm |
| | Precision | 장미 | 98% | 100% | 31×10×4 cm |
| | Tool_use | 먼지떨이 | 100% | 100% | 31×13×22 cm |

추출 크기가 실물과 부합한다 — 루빅스 큐브(실제 5.7cm), 치약(15×4×3cm), 충전기 등.

충전기·치약처럼 **작은 물체가 옮겨지는 구간에서는 검출이 끊겨** 방식 A가 낮다.
이 구간은 방식 B가 실제 크기를 유지하며 위치를 추정해 덮는다.

## 실행 설정

`pipeline/config.py`에서 조정한다. 자세한 값은 [settings.md](docs/settings.md) 참조.

| 프리셋 | fps | 카메라 | 에피소드 | 예상(GPU 2장) |
|---|---|---|---|---|
| `robust3` | 6 | 머리 2대 | 태스크당 3개 | **4.5h** |
| `fast` | 3 | 머리 1대 | 전수 (5,662) | **25h** |
| `balanced` | 6 | 머리 2대 | 전수 | 78h |
| `full` | 6 | 4대 | 전수 | 137h |

주요 조정: `fps`(완료 시간에 정비례), `cameras`(대수에 정비례),
`episodes_per_task`(0=전수), `amodal`(가림 보정), `save_video`(대량 시 끔).

태스크별로 다른 설정이 필요한 경우 `pipeline/profiles.py`에서 지정한다 —
전역 값을 바꾸면 잘 되던 태스크가 깨지기 때문이다.

## 파이프라인 5단계

```
RGB 프레임
  → 1. GroundingDINO  (텍스트 프롬프트 → 2D 박스)
  → 2. SAM 2.1        (박스 → 인스턴스 마스크 + 프레임 간 추적)
  → 3. UniDepthV2 / 실측 depth  (metric depth + intrinsics)
  → 4. Back-projection (마스크 + depth → 객체별 점군)
  → 5. 3D Bounding Box (점군 → axis-aligned 박스)
```

**3D box 2가지 방식** — 1~4단계가 공통이라 한 번 처리로 동시 산출한다.

| 방식 A (관측된 것만) | 방식 B (가림 보정) |
|---|---|
| 실제로 보이는 픽셀만으로 박스 | 가려진 부분을 고려한 물체 실제 크기 |
| 관측이 끊기면 그리지 않음 | 실제 크기를 유지하며 위치 추정(최대 2.5초) |

## 산출물

```
review/<라운드>/
├── brainco/<태스크>_ep<N>/<카메라>/
│   ├── A_visible.mp4  B_amodal.mp4     방식 A/B 오버레이 영상
│   ├── AB_compare.png  AB_occluded.png 좌우 비교 / 가림 순간
│   ├── stats.json                      커버리지·실관측·크기·기각 사유
│   └── frames.json                     프레임별 3D box와 진단량
└── he/<카테고리>_ep<N>/                 (HE는 카메라 1대)
```

## GPU 사용 — 연구실 정책 준수

**PBS batch job으로만 실행한다.** interactive job과 job 없는 직접 실행은 금지다.

- Node 1은 job당 CPU 4개, Node 3은 8개 제한
- job 이름 규칙: `G{GPU수}C{CPU수}_{이니셜}_{프로젝트}`
- job 내부에서 재시도 루프를 돌리고, 완료된 유닛은 결과 파일 존재로 건너뛴다 → **중단 시 이어서 진행**

```bash
qsub -q pleiades1 -l select=1:ncpus=4:ngpus=1 -l walltime=05:00:00 pipeline/pbs_r4.sh
qstat -u isangmin
```

## 알려진 데이터 특성

| 항목 | 내용 |
|---|---|
| **PickDrink 라벨 오류** | 지시문은 "red cup"이지만 실물은 파란 뚜껑 물병. **지시문을 그대로 믿으면 안 된다** |
| HE 로봇 혼재 | g1 4,064 + h1 4,885. 본 과제는 G1이므로 g1만 사용 |
| HE target 위치 | 태스크명이 아니라 **description에 명시** |
| HE RGB-depth 정렬 | depth가 RGB 대비 약 20px 어긋남 |
| HE intrinsics | 메타에 없어 FOV 70° 가정 — 3D 크기가 상수배로 어긋날 수 있음 |
| depth 정확도 | UniDepthV2는 근접(0.3~0.8m) 오차 43%, 중거리(1~3m) 21.7% |

## 품질 지표에 대한 방침

산출물 품질을 자동 판정하는 지표를 계산하지만, **이것으로 파라미터를 자동 조정하지 않는다.**

> 실측 사례: "화면의 12% 이상이면 오검출"이라는 지표가 손목 카메라에서 **정상 물체를 잘라냈다.**
> 손목은 근접 촬영이라 물체가 화면을 채우는 것이 정상이다.

지표는 대규모 실행에서 **사람이 볼 표본을 고르는 용도**이고, 판정의 최종 근거는 육안 검증이다.
