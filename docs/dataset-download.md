# 데이터셋 다운로드 현황 (2026-07-22 시작)

## 현재 상태 (2026-07-23 완료)

**✅ 두 링크 전체 다운로드 완료** — FAILED/미완료 파일 0건.

- **G1 Brainco 컬렉션 8개**: 2026-07-22 19:31~20:15 (44분, 57.3GB)
- **Humanoid Everyday**: 2026-07-22 20:15 ~ 2026-07-23 05:12 (약 9시간, 872GB)
- 합계 **926GB**, 디스크 `/data2` 여유 1.7T (89% 사용)
- Humanoid Everyday 최상위 구조: `data/`, `videos/`, `meta/`, `resume_meta/`, `README.md`

> 구조 분석은 사용자 요청 시 시작 (2026-07-23 오후 사수님 보고 목표).

---

## (참고) Brainco 완료 시점 기록 (2026-07-22 20:17)

**G1 Brainco 컬렉션 8개 전부 완료** (19:31~20:15, 44분, 합계 57.3GB)

| 데이터셋 | 용량 | 완료 시각 |
|---|---|---|
| G1_Brainco_GraspOreo | 8.4G | 19:37 |
| G1_Brainco_GraspRubiksCube | 7.8G | 19:43 |
| G1_Brainco_PickApple | 4.7G | 19:46 |
| G1_Brainco_PickCharger | 6.7G | 19:51 |
| G1_Brainco_PickDoll | 11G | 19:59 |
| G1_Brainco_PickDrink | 6.0G | 20:05 |
| G1_Brainco_PickTissues | 6.1G | 20:12 |
| G1_Brainco_PickToothpaste | 6.6G | 20:15 |

**Humanoid Everyday (935GB)**: 20:15 자동 시작, 진행 중. 약 25MB/s 기준 **2026-07-23 오전 6~7시 완료 예상**.

디스크: `/data2` 여유 2.5T (84% 사용).

> 구조 분석은 두 링크 전체 다운로드 완료 후 일괄 수행 예정 (2026-07-23 오후 보고 목표).

## 다운로드 위치

- **서버**: 연구실 서버 `pleiades1` (`ssh -p 3022 isangmin@10.20.23.30`)
- **실제 다운로드 경로**: `/data2/humanoid_dataset_isangmin/`
- **원래 목표 경로**: `/data2/humanoid_dataset/`

### ⚠️ 경로가 달라진 이유

`/data2/humanoid_dataset`은 `s20245171` 계정 소유(권한 755)로 생성되어 있어 `isangmin` 계정으로 쓰기가 불가능. `/data2`는 sticky bit가 걸린 공용 디렉토리라 타 계정 폴더의 권한 변경/삭제도 불가.

→ 같은 파일시스템(`pleiades2:/data2` NFS 마운트)에 `/data2/humanoid_dataset_isangmin/`을 만들어 다운로드 중.

**나중에 원래 경로로 옮기는 방법** (`s20245171` 계정에서 실행):

```bash
chmod 777 /data2/humanoid_dataset
```

이후 `isangmin` 계정에서 `mv /data2/humanoid_dataset_isangmin/* /data2/humanoid_dataset/` — 같은 마운트라 복사 없이 즉시 이동됨.

## 다운로드 방식

- 서버의 **tmux 세션 `hfdl`** 에서 실행 중 → **노트북을 꺼도 서버에서 계속 진행됨**
- 스크립트: `/data2/humanoid_dataset_isangmin/download_all.sh`
  - `hf download` (huggingface_hub CLI 1.8.0, `hf_transfer` 활성화)
  - 데이터셋별 실패 시 30초 간격 5회 재시도, 중단 시 이어받기 지원
- 로그: `/data2/humanoid_dataset_isangmin/download.log`

### 다운로드 순서

1. `unitreerobotics/G1_Brainco_GraspOreo_Dataset`
2. `unitreerobotics/G1_Brainco_GraspRubiksCube_Dataset`
3. `unitreerobotics/G1_Brainco_PickApple_Dataset`
4. `unitreerobotics/G1_Brainco_PickCharger_Dataset`
5. `unitreerobotics/G1_Brainco_PickDoll_Dataset`
6. `unitreerobotics/G1_Brainco_PickDrink_Dataset`
7. `unitreerobotics/G1_Brainco_PickTissues_Dataset`
8. `unitreerobotics/G1_Brainco_PickToothpaste_Dataset`
9. `USC-PSI-Lab/humanoid-everyday` (935GB, 마지막 자동 시작)

## 진행상황 확인 방법

로컬(노트북)에서:

```bash
bash scripts/check_download.sh
```

또는 서버 접속 후:

```bash
bash /data2/humanoid_dataset_isangmin/status.sh
```

실시간 다운로드 화면을 직접 보려면 서버에서 `tmux attach -t hfdl` (빠져나올 땐 `Ctrl+b` 후 `d`).

## 다운로드 속도 관련

현재 **HF 토큰 없이(익명)** 받는 중 → HuggingFace가 익명 요청에 속도/횟수 제한을 걸기 때문에, 특히 935GB 다운로드 전에 토큰 설정 권장:

1. https://huggingface.co/settings/tokens 에서 Read 토큰 생성
2. 서버 접속 후 직접 실행:
   ```bash
   ~/.local/bin/hf auth login
   ```
   (토큰 입력하면 `~/.cache/huggingface/token`에 저장되어 이후 다운로드에 자동 적용)

`hf_transfer`(멀티스레드 전송)는 이미 활성화되어 있어 토큰만 넣으면 대역폭을 거의 최대로 사용함.

## 디스크 용량 주의

- `/data2`: 총 15T 중 여유 **2.6T** (사용률 83%)
- Humanoid Everyday(935GB)까지 받으면 사용률 약 **93%** 예상

## 데이터셋 사전 정보 (HF 카드 기준)

### UniFoLM G1 Brainco 시리즈 (LeRobot v2.1)

- Unitree G1 휴머노이드 + Brainco 덱스터러스 핸드, 텔레오퍼레이션 수집
- 카메라 4대: 머리 좌/우 + 손목 좌/우 (480×640, AV1 코덱, 30fps)
- 관절 상태(observation.state) / 액션 모두 26차원 (양팔 어깨·팔꿈치·손목 + 손가락)
- 태스크당 약 94~272 에피소드, parquet(`data/chunk-*/episode_*.parquet`) + mp4 비디오 구조

### Humanoid Everyday (LeRobot v2.0, 935GB)

- 260+ 태스크, 7개 카테고리 (manipulation, HRI, locomotion 등)
- 모달리티: RGB, **Depth, LiDAR 포인트클라우드, 촉각 센서, IMU(쿼터니언/가속도/자이로), 오도메트리**, 관절각(팔/다리/손)
- 30Hz, 사람 감독 텔레오퍼레이션 수집
- 전체 태스크 목록/카테고리가 공개 스프레드시트로 제공됨
- → real 데이터에서 메타데이터 대용으로 쓸 수 있는 모달리티가 풍부 (depth/LiDAR 기반 3D 정보, 촉각 기반 접촉 정보 등)

## 다음 단계

- [ ] 첫 데이터셋(GraspOreo) 완료 후 실제 파일 구조 분석
- [ ] LeRobot 포맷 메타(`meta/info.json`, `episodes.jsonl` 등) 정리
- [ ] 객체 간 관계 정보 추출 가능성 검토
- [ ] Humanoid Everyday 다운로드 완료 후 구조 분석
