# 데이터셋 구조 분석

## G1 Brainco 시리즈 — LeRobot v3.0 포맷

> GraspOreo 기준 실측 분석 (2026-07-22). 8개 데이터셋 모두 동일 구조, 태스크만 다름.
> ⚠️ HF 카드에는 v2.1로 표기되어 있으나 **실제 다운로드본은 `codebase_version: v3.0`** (v2와 파일 배치가 다름 — 에피소드별 parquet이 아니라 통합 파일 방식)

### 디렉토리 구조

```
G1_Brainco_GraspOreo_Dataset/
├── README.md
├── meta/
│   ├── info.json                  # 데이터셋 스펙 전체 (feature 정의, 경로 규칙, fps 등)
│   ├── stats.json                 # 전체 통계 (feature별 min/max/mean/std)
│   ├── tasks.parquet              # 자연어 태스크 ↔ task_index 매핑
│   └── episodes/chunk-000/        # 에피소드별 메타데이터 parquet
├── data/chunk-000/
│   └── file-000.parquet           # 전체 에피소드 프레임 데이터 통합 (234,959행)
└── videos/
    ├── observation.images.cam_left_high/chunk-000/file-00X.mp4
    ├── observation.images.cam_right_high/chunk-000/...
    ├── observation.images.cam_left_wrist/chunk-000/...
    └── observation.images.cam_right_wrist/chunk-000/...
        # 카메라별 ~500MB 단위로 이어붙인 mp4 (에피소드별 파일 아님)
```

### info.json 핵심 (GraspOreo)

| 항목 | 값 |
|---|---|
| codebase_version | v3.0 |
| robot_type | Unitree_G1_Brainco |
| total_episodes / frames | 201 / 234,959 |
| fps | 30 |
| splits | train: 0~201 (전부 train) |
| 용량 | 12GB |

### Features (프레임 단위 데이터)

| Feature | 타입/형태 | 내용 |
|---|---|---|
| `observation.state` | float32 [26] | 관절 각도 26차원 |
| `action` | float32 [26] | 목표 관절 명령 26차원 (state와 동일 구조) |
| `observation.images.cam_{left,right}_high` | video [3,480,640] AV1 | 머리 좌/우 카메라 |
| `observation.images.cam_{left,right}_wrist` | video [3,480,640] AV1 | 손목 좌/우 카메라 |
| `timestamp` | float32 | 에피소드 내 시간 (초) |
| `frame_index` / `episode_index` / `index` / `task_index` | int64 | 인덱싱 |

**26차원 관절 구성** (state/action 공통, 좌→우 순):
- 팔 7×2 = 14: ShoulderPitch/Roll/Yaw, Elbow, WristRoll/Pitch/Yaw
- 손 6×2 = 12 (Brainco 핸드): Thumb, ThumbAux, Index, Middle, Ring, Pinky

### 에피소드 메타 (`meta/episodes/`)

에피소드별로 다음 정보 보유:
- 해당 에피소드가 어느 data parquet / 어느 video 파일의 어느 timestamp 구간에 있는지 (`from_timestamp`~`to_timestamp`) → **비디오가 통합 파일이라 이 매핑으로 잘라서 읽어야 함**
- `tasks` (자연어), `length` (프레임 수)
- 에피소드별 통계: 모든 feature의 min/max/mean/std/count

### tasks.parquet

자연어 태스크 → task_index 매핑. GraspOreo는 단일 태스크: `"Pick up the Oreo"`

### 로딩 방법

```python
# 방법 1: lerobot 라이브러리 (권장)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("unitreerobotics/G1_Brainco_GraspOreo_Dataset",
                    root="/data2/humanoid_dataset_isangmin/G1_Brainco_GraspOreo_Dataset")

# 방법 2: 직접 로딩
import pandas as pd
df = pd.read_parquet(".../data/chunk-000/file-000.parquet")   # state/action/인덱스
ep = pd.read_parquet(".../meta/episodes/chunk-000/file-000.parquet")  # 에피소드↔비디오 매핑
# 프레임 이미지는 mp4에서 timestamp 기준으로 디코딩 (AV1 코덱 주의 — ffmpeg/torchcodec 필요)
```

### 활용 관점 메모

- **있는 것**: 4시점 RGB, 26-DoF 관절 state/action, 자연어 태스크 라벨, 에피소드 경계, 통계
- **없는 것 (real이라 메타데이터 부재)**: depth, 객체 pose/세그멘테이션, 접촉/힘 정보, 카메라 캘리브레이션(extrinsics/intrinsics 미포함 여부 README 추가 확인 필요)
- 객체 관계 정보를 얻으려면 RGB에서 비전 모델(검출/세그멘테이션/6D pose, VLM 등)로 추출하는 파이프라인 필요
- 손가락 관절값(12차원)은 grasp 시점 검출(접촉 프록시)에 활용 가능

## Humanoid Everyday — 다운로드 후 분석 예정

카드 기준: LeRobot v2.0, 260+ 태스크/7 카테고리, RGB+Depth+LiDAR+촉각+IMU/오도메트리, 30Hz, 935GB.
Depth/LiDAR가 있어 3D 공간 정보(객체 관계 추출)에 Brainco 시리즈보다 유리할 것으로 예상.
