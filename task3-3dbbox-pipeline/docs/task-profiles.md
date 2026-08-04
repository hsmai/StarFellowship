# 태스크별 프로파일 — r2 뼈대 + 선택적 개선

## 배경

라운드 4번(r1~r4)을 돌리며 개선했으나, 육안 검증 결과 **전체적으로는 r2가 가장 좋았다**.

r3에서 얇은 부속물(토끼 귀·꽃 줄기)을 살리려 점군 필터를 완화한 것이 잘 되던 태스크를
무너뜨렸다.

| | r2 | r3 |
|---|---|---|
| GraspOreo/머리좌 | **100%** | 67% |
| PickCharger 크기 | **6×6×2cm** | 8×28×30cm |

그러나 **일부 태스크는 r3/r4가 확실히 나았다** — PickDrink(0%→100%),
HE deformable·Locomanip·Tool_use(유령 제거), HRI·Precision(꽃 줄기 포함).

전역 파라미터 하나로 두 요구를 동시에 만족할 수 없다.
→ **태스크별로 프로파일을 지정**한다. 이것이 "문제 있는 태스크만 건드리고
나머지는 손대지 않는다"를 코드로 보장하는 방법이다.

## 프로파일 정의 (`pipeline/profiles.py`)

| 옵션 | R2 (기본) | R3 | R4 | 의미 |
|---|---|---|---|---|
| `filter_pct` | (10,90) | (2,98) | (10,90) | depth 백분위 클리핑 |
| `filter_mad` | 1.5 | 3.0 | 1.5 | 전경 분리 강도 |
| `components` | largest | same_depth | same_depth | 마스크 성분 처리 |
| `use_distractors` | True | False | False | 프롬프트에 손·팔 어휘 추가 |
| `anchor_prop` | False | True | True | 전파 창을 실검출에 고정 |
| `prop_limit_sec` | 0(무제한) | 0 | 1.0 | 전파 지속 상한 |
| `iou_bonus` | 0.3 | 0.3 | 0.45 | 추적 연속성 가중 |
| `mirror_z` | False | True | True | 방식 B 두께 대칭 보정 |

## 태스크별 지정

육안 검증 결과에 따라 확정한 매핑이다.

### Brainco

| 태스크 | 프로파일 | 근거 |
|---|---|---|
| GraspOreo | **R2** | head/wrist, A/B 모두 완벽 — 건드리지 않는다 |
| GraspRubiksCube | **R2** | 완벽 — 건드리지 않는다 |
| PickApple | R2 + 보완 | head 완벽. 집는 쪽 wrist에서 사과가 화면을 채울 때만 |
| PickCharger | R2 + 보완 | 로봇팔 오인이 남아 있음 |
| PickDoll | **R3** | r3의 head 결과가 최상. wrist는 검출 불가로 판단 |
| PickDrink | **R3** | r3에서 물병 검출 0%→100% |
| PickTissues | **R3** | r3의 head 결과 채택 |
| PickToothpaste | R2 + `erode_kernel=3`, `min_points=8` | 얇고 긴 물체(15×4×3cm)라 5×5 침식이 마스크를 지운다 |

### Humanoid Everyday

| 카테고리 | 프로파일 | 근거 |
|---|---|---|
| Articulated | R2 + 보완 | 노트북이 닫힌 뒤 박스가 작아지는 것만 |
| Basic | R2 + 보완 | pink toy 이동 구간 |
| deformable | **R3** | r3 채택 |
| HRI | **R4** | r4 채택 |
| Locomanip | **R3** | r3 채택 |
| Precision | R4 + `exclusive_tracks=True` | 장미가 시야를 벗어나면 꽃병을 장미로 잡는 것 방지 |
| Tool_use | **R3** | r3 채택 |

## 태스크별 보완의 근거 (r1~r4 결과 전수 분석)

5개 에이전트가 각 실패 태스크의 frames.json을 라운드별로 대조해 원인을 특정했다.

| 태스크 | 원인 (실측) | 처방 |
|---|---|---|
| **PickApple** 손목 | 사과가 화면의 **0.627**을 차지해 면적 상한 0.60에 **1~3%p 차이로** 걸렸다. 기각 56프레임 중 52건이 이 사유이고, 검출 점수 0.827·재투영 정합은 모두 정상이었다 | `area_max` 0.60 → **0.75** |
| **PickCharger** | GroundingDINO가 로봇 전완을 "white charger"로 반환하고 **점수가 겹친다**(팔 0.388 vs charger 0.369). 후보가 하나라도 잡히면 재검출·전파 경로가 열리지 않아 **팔이 슬롯을 빼앗는다**. left_high 손실 109프레임이 팔 오검출 109프레임과 1:1 일치 | `max_phys_size` 0.75 → **0.15m** (charger는 손에 들어가는 물체) |
| **PickToothpaste** | 얇고 긴 물체(15×4×3cm)라 5×5 침식이 마스크를 지운다. 커버리지 100%인데 육안으로 끊겨 보이는 것은 **전파로만 유지되는 구간** 때문 | 침식 3, 최소점 8, **실제 치수를 이력 시드로**, 전파 붕괴 시 기각 |
| **HE Articulated** | 노트북이 닫히면 형상이 납작해져 점군이 잘린다 | 점군 필터만 완화 (성분 처리는 r2 유지) |
| **HE Basic** | 이력이 빈 초반에 가림 판정이 과민해 기준이 서지 않는다 | `cold_start_hist=5` |
| **HE Precision** | 장미가 시야를 벗어나면 **꽃병 박스를 장미로** 잡는다. r2·r4의 프롬프트가 다른데도 같은 실패 → 프롬프트 원인이 아니다 | `exclusive_tracks` |

**전역 변경은 하지 않았다.** 면적 상한을 전역으로 올리면 PickDoll(+91프레임) 등
기준선이 r2가 아닌 태스크의 판정이 뒤집힌다. 물리 크기 상한을 전역 0.15m로 하면
PickDoll 승인분이 전멸한다. 모두 해당 태스크에만 건다.

## 회귀 기준선

**GraspOreo·GraspRubiksCube에는 어떤 override도 넣지 않았다.**
이 둘의 결과가 r2와 일치하는지로 "프로파일 구조 도입이 무해했는가"를 판정한다.
차이가 나오면 그 변경을 되돌린다.

## 트랙 간 박스 독점 (`exclusive_tracks`)

한 프레임에서 두 target이 IoU 0.7 넘게 겹치는 박스를 쓰면, 검출 점수가 높은 쪽만 남긴다.
"target이 사라지면 비슷한 다른 물체로 옮겨 붙는" 실패를 막는다.

전역이 아니라 **필요한 태스크에만** 켠다 — 잘 되는 태스크의 동작을 바꾸지 않기 위해서다.

## 사용법

```python
from profiles import profile_for
prof = profile_for("GraspOreo")     # R2
prof = profile_for("PickDrink")     # R3
trk = EpisodeTracker(targets, det, seg, prompt, profile=prof)
```

러너(`run_review.py`)가 태스크 이름으로 프로파일을 자동 선택하므로,
실행 시 별도 지정이 필요 없다.

## 원칙

1. **문제 없는 태스크는 건드리지 않는다.** GraspOreo·GraspRubiksCube는 R2 그대로다.
2. **전역 변경을 피한다.** 필요한 보완은 해당 태스크의 override로만 건다.
3. **근거를 남긴다.** 각 지정에 육안 검증 결과를 주석으로 붙여, 나중에 왜 그런지 알 수 있게 한다.
