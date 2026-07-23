# 데이터셋 구조 분석 (전체 실측 기반, 2026-07-23)

> 서버 `/data2/humanoid_dataset_isangmin/` 다운로드본 전수 실측. 두 링크는 포맷 버전과 파일 배치가 서로 다름.

---

# 링크 1. UniFoLM G1 Brainco 컬렉션 (8개 데이터셋, 57.3GB)

## 공통 개요

| 항목 | 값 |
|---|---|
| 포맷 | **LeRobot v3.0** (HF 카드의 v2.1 표기와 다름 — 실측 `codebase_version: v3.0`) |
| 로봇 | Unitree_G1_Brainco (G1 휴머노이드 + Brainco 덱스터러스 핸드) |
| 수집 방식 | 텔레오퍼레이션 |
| fps | 30 |
| 태스크 | 데이터셋당 1개 (자연어 라벨) |
| 카메라 | 4대 — 머리 좌/우(cam_left_high, cam_right_high) + 손목 좌/우(cam_left_wrist, cam_right_wrist), 480×640, AV1 코덱 |
| state/action | 각 26차원 float32 (구성 동일) |
| split | 전부 train |
| 합계 | 8개 데이터셋, 1,598 에피소드, 1,656,316 프레임 |

## 데이터셋별 현황

| 데이터셋 | 태스크(자연어) | 에피소드 | 프레임 | 평균 길이 | 용량 |
|---|---|---|---|---|---|
| GraspOreo | Pick up the Oreo | 201 | 234,959 | 39.0초 | 8.4G |
| GraspRubiksCube | Pick up the Rubik's Cube and put it in the plate | 197 | 220,788 | 37.4초 | 7.8G |
| PickApple | Put the apples into the plate. | 200 | 153,668 | 25.6초 | 4.7G |
| PickCharger | Pick up the charger. | 200 | 216,532 | 36.1초 | 6.7G |
| PickDoll | Put the doll into the plate. | 200 | 313,401 | 52.2초 | 11G |
| PickDrink | Pick up the red cup on the table. | 201 | 174,483 | 28.9초 | 6.0G |
| PickTissues | Put the tissue paper into the plate. | 206 | 198,622 | 32.1초 | 6.1G |
| PickToothpaste | Put the toothpaste on the table into the plate. | 193 | 143,863 | 24.8초 | 6.6G |

## 디렉토리 구조 (8개 모두 동일)

```
G1_Brainco_XXX_Dataset/
├── README.md
├── meta/
│   ├── info.json                  # 스펙 전체: feature 정의, 경로 규칙, fps, 에피소드/프레임 수
│   ├── stats.json                 # 전체 feature별 min/max/mean/std
│   ├── tasks.parquet              # 자연어 태스크 ↔ task_index 매핑
│   └── episodes/chunk-000/*.parquet   # 에피소드별 메타 (아래 참조)
├── data/chunk-000/
│   └── file-000.parquet           # ★ 전 에피소드 프레임 데이터가 한 파일에 통합
└── videos/
    ├── observation.images.cam_left_high/chunk-000/file-000.mp4 ...
    ├── observation.images.cam_right_high/chunk-000/...
    ├── observation.images.cam_left_wrist/chunk-000/...
    └── observation.images.cam_right_wrist/chunk-000/...
        # ★ 에피소드별 파일이 아님 — 카메라별로 여러 에피소드를 ~500MB 단위로 이어붙인 mp4 (총 12~25개)
```

## 프레임 데이터 스키마 (data parquet 컬럼)

| 컬럼 | 타입 | 내용 |
|---|---|---|
| observation.state | float32[26] | 현재 관절 각도 |
| action | float32[26] | 목표 관절 명령 (state와 같은 26차원 구성) |
| timestamp | float32 | 에피소드 내 경과 시간(초) |
| frame_index | int64 | 에피소드 내 프레임 번호 |
| episode_index | int64 | 에피소드 번호 |
| index | int64 | 데이터셋 전역 프레임 번호 |
| task_index | int64 | 태스크 번호 (전부 0, 단일 태스크) |

**26차원 관절 구성** (state/action 공통, 순서 고정):
- 팔 14 = 좌/우 × (ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow, WristRoll, WristPitch, WristYaw)
- 손 12 = 좌/우 × Brainco 핸드 (Thumb, ThumbAux, Index, Middle, Ring, Pinky)
- ※ 다리/허리 관절은 없음 (상체 조작 데이터)

## 에피소드 메타 (meta/episodes/)

에피소드별 1행씩, 주요 컬럼:
- `data/chunk_index`, `data/file_index`, `dataset_from_index`, `dataset_to_index` — 통합 parquet 안에서 해당 에피소드의 행 범위
- `videos/<카메라>/chunk_index, file_index, from_timestamp, to_timestamp` — **통합 mp4 안에서 해당 에피소드의 시간 구간** (프레임 복원에 필수)
- `tasks`(자연어), `length`(프레임 수)
- `stats/<feature>/min·max·mean·std·count` — 에피소드 단위 통계

## 포함된 것 / 없는 것

**있음**: 4시점 RGB, 26-DoF state/action(팔+손가락), 자연어 태스크, 에피소드 경계·통계, 30fps 동기화
**없음**: depth, 객체 pose/세그멘테이션, 접촉/힘, 카메라 캘리브레이션(intrinsics/extrinsics), 다리/허리 관절, 성공/실패 라벨

## 로딩

```python
# lerobot 라이브러리 (v3.0 지원 버전 필요)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("unitreerobotics/G1_Brainco_GraspOreo_Dataset",
                    root="/data2/humanoid_dataset_isangmin/G1_Brainco_GraspOreo_Dataset")

# 직접 로딩
import pandas as pd
df = pd.read_parquet(".../data/chunk-000/file-000.parquet")
ep = pd.read_parquet(".../meta/episodes/chunk-000/file-000.parquet")  # 에피소드↔비디오 timestamp 매핑
# 이미지: 통합 mp4에서 from_timestamp~to_timestamp 구간을 ffmpeg/torchcodec으로 디코딩 (AV1)
```

---

# 링크 2. Humanoid Everyday (872GB)

## 개요

| 항목 | 값 |
|---|---|
| 포맷 | **LeRobot v2.1** (README에는 v2.0 호환 표기) — v3.0과 달리 **에피소드별 파일 분리** |
| 로봇 | **mixed — Unitree G1 4,064 + H1 4,885 에피소드** (에피소드별 `robot_type` 라벨) |
| 규모 | 8,949 에피소드 / 3,436,171 프레임 / 246 태스크 / 7 카테고리 |
| 에피소드 길이 | 110~2,402 프레임 (평균 384 = 12.8초 @30fps) |
| 수집 | 사람 감독 텔레오퍼레이션, 30Hz |
| 용량 분해 | data(parquet) **865G** + videos 6.1G + meta 2.9M + resume_meta 125M |
| 출처 | arXiv 2510.08807, 전용 데이터로더: github.com/ausbxuse/Humanoid-Everyday |

※ 용량의 99%가 parquet — **depth 맵과 LiDAR 포인트클라우드가 프레임마다 parquet 안에 원시값으로 저장**되어 있기 때문.

## 디렉토리 구조

```
humanoid-everyday/
├── README.md
├── meta/
│   ├── info.json              # 스펙 (feature 정의, 총계, 경로 규칙)
│   ├── tasks.jsonl            # 246개 태스크: task_index, task(카테고리/이름), category, description
│   ├── episodes.jsonl         # 8,949행: episode_index, tasks, length, robot_type, instruction(자연어)
│   ├── episodes_stats.jsonl   # 에피소드별 feature 통계
│   ├── errors.jsonl           # 제외된 에피소드 기록 (3건: 27, 640, 6702 — zero valid frames)
│   └── stats.json             # 전체 통계
├── data/chunk-000 ~ chunk-008/
│   └── episode_000000.parquet ...   # ★ 에피소드당 1파일 (총 8,949개)
├── videos/chunk-000 ~ chunk-008/
│   └── egocentric/episode_000000.mp4 ...  # ★ 에고센트릭 RGB, 에피소드당 1파일 (총 8,954개)
└── resume_meta/               # 업로드 재개용 부산물 (episode_*.parquet.meta.json, 분석에 불필요)
```

## 프레임 데이터 스키마 (episode parquet 컬럼)

| 컬럼 | 타입/형태 | 내용 |
|---|---|---|
| observation.images.egocentric | video 참조 [480,640,3] | 에고센트릭 RGB (mp4 별도 저장) |
| observation.depth.egocentric | float32 [480,640] | **per-pixel depth 맵** (mm 단위 추정, parquet 내 원시 저장) |
| observation.lidar | float32 [N,3] | **LiDAR 포인트클라우드** (프레임당 약 4,700~4,800 점) |
| observation.imu.quaternion / accelerometer / gyroscope / rpy | float32 [4]/[3]/[3]/[3] | IMU |
| observation.odometry.position / velocity / rpy / quat | float32 [3]/[3]/[3]/[4] | 오도메트리 (몸통 위치·속도·자세) |
| observation.arm_joints | float32 [14] | 팔 관절 (G1/H1 공통 14) |
| observation.leg_joints | float32 [가변] | 다리 관절 — **G1: 15, H1: 13** |
| observation.hand_joints | float32 [가변] | 손 관절 — **G1: 14, H1: 12** |
| observation.tactile.sensor_id | int64 [가변] | 촉각 센서 ID — **G1: 18개, H1: 0개(H1엔 촉각 없음)** |
| observation.tactile.values | float32 [센서수,4] | 촉각 값 (센서당 4채널) |
| action | float32 [가변] | 액션 — **G1: 28, H1: 26** |
| timestamp / frame_index / episode_index / index / task_index | 스칼라 | 인덱싱 |
| next.done | bool | 에피소드 종료 플래그 |

**⚠️ 로봇별 가변 차원**: leg/hand/tactile/action 차원이 G1과 H1이 다름 → 학습에 쓸 때 robot_type별 분리 또는 패딩 필요.

## 태스크 구성 (246개, 7 카테고리)

| 카테고리 | 태스크 수 | 에피소드 수 | 예시 |
|---|---|---|---|
| Basic | 64 | 2,638 | 기본 pick & place류 |
| Articulated | 48 | 1,918 | 관절 있는 물체 조작 (모니터 각도 조절, 폰 거치대 조절 등) |
| Locomanip | 47 | 879 | 이동+조작 결합 |
| HRI | 41 | 1,638 | 인간-로봇 상호작용 |
| deformable | 19 | 800 | 변형체 조작 (천, 종이 등) |
| Tool_use | 19 | 756 | 도구 사용 (지우개로 책상 닦기 등) |
| Precision | 8 | 320 | 정밀 조작 |

- `tasks.jsonl`의 각 태스크에 **문장 단위 상세 description** 포함 (예: "the robot uses its right hand to hold the eraser ... wipe the desk by swiping its surface twice")
- `episodes.jsonl`의 에피소드마다 **자연어 instruction**이 별도로 붙어 있음 (VLA 학습용 language conditioning에 바로 사용 가능)

## 포함된 것 / 없는 것

**있음**: 에고센트릭 RGB + **depth** + **LiDAR** + IMU + 오도메트리 + 전신 관절(팔/다리/손) + **촉각(G1만)** + 자연어 instruction + 태스크 카테고리 + 에피소드별 robot_type + done 플래그
**없음**: 멀티뷰 카메라(에고센트릭 1시점뿐), 객체 pose/세그멘테이션 GT, 카메라 intrinsics(README/메타에 미포함 — 데이터로더 레포 확인 필요), 성공/실패 라벨

## 로딩

```python
# 방법 1: lerobot (README 권장)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("USC-PSI-Lab/humanoid-everyday",
                    root="/data2/humanoid_dataset_isangmin/humanoid-everyday",
                    tolerance_s=1e-2)

# 방법 2: 직접 로딩 (에피소드 단위라 간단)
import pandas as pd
df = pd.read_parquet(".../data/chunk-000/episode_000000.parquet")
depth0 = np.stack(df["observation.depth.egocentric"].iloc[0])   # (480, 640)
lidar0 = np.stack(df["observation.lidar"].iloc[0])              # (N, 3)
# RGB: videos/chunk-000/egocentric/episode_000000.mp4 (에피소드당 1파일이라 바로 디코딩)

# 방법 3: 저자 제공 수동 데이터로더
# github.com/ausbxuse/Humanoid-Everyday
```

---

# 활용 관점 메모 (요약)

- **객체 관계/3D 정보 추출**: Humanoid Everyday가 유리 — depth 맵 + LiDAR가 프레임 단위로 있어, RGB 세그멘테이션(SAM 등)과 depth를 결합하면 카메라 좌표계 3D 객체 위치·관계를 메타데이터 없이 복원 가능. 오도메트리로 월드 좌표 근사도 가능
- **접촉/grasp 시점 검출**: HE의 G1 에피소드는 촉각 센서(18개×4채널)로 직접 검출 가능. Brainco는 손가락 관절값(12차원) 변화를 프록시로 사용
- **Language conditioning**: 양쪽 다 자연어 라벨 보유. HE는 에피소드별 상세 instruction까지 있어 VLA류 학습에 바로 사용 가능
- **주의**: HE는 로봇 2종(G1/H1)이 섞여 있고 관절/액션 차원이 다름. Brainco는 비디오가 에피소드별로 안 나뉘어 있어 전처리(timestamp 기반 분리) 필요
