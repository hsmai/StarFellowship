# V3 — 검증 루프 체계 (2026-08-03)

## 목표 재정의

본 과제의 산출물은 "전 데이터 3D box"가 아니라 **모든 종류의 task에 robust한 3D box 생성
파이프라인**이다. 따라서 진행은 두 단계로 나눈다.

1. **프로토타입 확정** — 데이터셋별 모든 task를 대표하는 15개(Brainco 8 + HE 7)로
   개선→평가를 반복해 품질을 올린다. ← 현재 단계
2. **robustness 검증** — 품질이 충분해지면 task당 3~5 에피소드로 확대 실행해 확정한다.

## 1. charger 오검출 원인과 해결

PickCharger만 전혀 다른 물체를 잡았다. 프롬프트별로 검출 결과를 실측했다.

| 프롬프트 | 검출 결과 (면적) |
|---|---|
| `white charger . plate .` | plate 9,137px / **white charger 50,482px** |
| `charger . plate .` | plate만 (charger 검출 실패) |
| `power adapter . plate .` | plate 9,076px / power adapter 50,474px |

**원인**: 실제 charger는 로봇 왼손이 이미 쥐고 있는 작은 흰 물체인데, 흰 로봇팔과 색·위치가
겹쳐 GroundingDINO가 **팔 전체를 charger로 잡았다**. 접시의 5.5배 면적이다.

**해결**: 2D 검출 단계에 **면적 상한 게이트**를 넣었다. 주 조작 대상(target object)은
사람이 한 손으로 드는 물체이므로 화면의 12%를 넘을 수 없다. 참조 물체(접시 등)는 45%.
프롬프트도 `small white adapter . charger plug . plate .` 로 좁혔다.

## 2. target object 개념 도입 (`pipeline/spec.py`)

태스크마다 **주 조작 대상**을 명시적으로 지정한다. 이 라벨에만 엄격한 면적 상한이 걸리고,
카메라 뷰별 검증도 이 대상 기준으로 판정한다.

```python
"PickDrink": dict(prompt="red cup . plate .",
                  targets={"cup": ("red cup", True),      # True = target object
                           "plate": ("plate", False)})
```

## 3. HE 대표 재선정 — g1 + target object 명시

기존에는 246개 태스크의 프롬프트를 태스크 이름에서 기계적으로 생성해 품질이 낮았다
(Articulated·Basic은 검출 실패, HRI는 사람까지 포함).

**변경**: HE는 g1(4,064 에피소드)과 h1(4,885)이 섞여 있는데 본 과제 대상이 Unitree G1이므로
**g1만** 쓰고, 카테고리별 대표를 **description에 target object가 명시된 태스크**로 재선정했다.

| 카테고리 | 에피소드 | 태스크 | target object |
|---|---|---|---|
| Basic | 3800 | put_dumpling_into_plate_g1 | 분홍 원형 인형 → 주황 접시 |
| Articulated | 280 | close_a_laptop_g1 | 노트북 |
| deformable | 4838 | fold_towel | 흰 체크무늬 수건 |
| HRI | 5598 | hand_over_flower | 장미 (사람에게 건넴) |
| Locomanip | 7120 | walk…pick_up_a_bottle… | 병 → 용기 |
| Precision | 7918 | insert_flower_into_vase | 장미 → 분홍 꽃병 |
| Tool_use | 8198 | clean_a_table_with_duster | 먼지떨이 |

## 4. 3D box 생성 2가지 방식 — 한 번 처리로 동시 산출

5단계 중 **1~4단계(검출·마스크·depth·역투영)는 공통**이고 5단계에서만 분기하므로,
한 번 실행으로 두 결과를 모두 낸다(처리 시간이 2배가 되지 않는다).

| | **방식 A — 보이는 부분만** | **방식 B — 가려짐 보정** |
|---|---|---|
| 정의 | 로봇 손에 가려진 부분을 뺀, 실제 관측된 픽셀만으로 박스 산출 | 크기가 급감하면 가려짐으로 판정하고, 이력 크기를 유지한 채 물체 전체 범위를 추정 |
| 판정 기준 | — | 현재 부피 < 이력 중앙값의 55% |
| 위치 처리 | 관측 점군 그대로 | 보이는 조각이 박스에 포함되도록 축별 최소 이동 |
| 기록 필드 | `center`, `size` | `center_amodal`, `size_amodal`, `occluded`, `visible_ratio` |

## 5. Brainco 4카메라 전부 검증

머리 좌/우 + 손목 좌/우 4뷰 각각에서 target object 박스를 추출한다.
(이전 측정: 손목 카메라가 조작 물체를 머리 대비 1.6배 잘 검출)

## 6. 산출 폴더 구조 — 육안 검증용

```
review/<라운드>/
├── brainco/<태스크>/<카메라>/
│   ├── A_visible.mp4     방식 A 오버레이 영상
│   ├── B_amodal.mp4      방식 B 오버레이 영상
│   ├── AB_compare.png    중간 프레임 A|B 좌우 비교
│   ├── AB_occluded.png   가림 발생 순간 A|B 비교
│   ├── stats.json        커버리지·기각사유·크기 중앙값
│   └── frames.json       프레임별 상세
└── he/<카테고리>/         (HE는 카메라 1개라 카테고리 바로 아래)
```

## 7. 샘플링 — 검증 단계는 품질 우선

| | 검증 라운드 (현재) | robustness 검증 (이후) |
|---|---|---|
| 샘플링 | **10fps** (stride 3) | 6fps |
| Brainco | 8태스크 × 4카메라 × 1ep = 32 | 8 × 4 × 3ep = 96 |
| HE | 7카테고리 × 1ep = 7 | 246태스크 × 3ep = 738 |
| 소요 | 약 25분 (2GPU) | 별도 산정 |

검증 라운드가 25분이라 개선→평가를 하루에 여러 번 돌릴 수 있다.
