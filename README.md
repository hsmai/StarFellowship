# StarFellowship

휴머노이드 로봇 real-world 데이터셋 분석 프로젝트.

## 목표

- Real-world 휴머노이드 데이터셋의 구조 파악 (포함된 정보, 객체 간 관계 정보 등)
- 데이터셋 정리 및 base 적용, 활용처 탐색
- Sim이 아닌 real 데이터에서 필요한 정보(메타데이터 부재)를 추출하는 방법 조사

## Task별 관리 구조

| Task | 내용 | 위치 | 상태 |
|---|---|---|---|
| Task 0 (7월) | Brainco 8종 + Humanoid Everyday 다운로드·EDA | [docs/](docs/) | ✅ 완료 |
| Task 1 | AgiBot World 2026 샘플 다운로드·EDA | [docs/AgiBot_EDA.docx](docs/AgiBot_EDA.docx) | ✅ 완료 |
| Task 2 | EgoDex 테스트셋 다운로드·EDA | [docs/EgoDex_EDA_보고서.docx](docs/EgoDex_EDA_보고서.docx) | ✅ 완료 |
| **Task 3** | **G1 데이터 3D BBox 추출 파이프라인** | [task3-3dbbox-pipeline/](task3-3dbbox-pipeline/) | 🔄 진행 중 |

## 대상 데이터셋

| 데이터셋 | 포맷 | 크기 | 상태 |
|---|---|---|---|
| [UniFoLM G1 Brainco 컬렉션](https://huggingface.co/collections/unitreerobotics/unifolm-g1-brainco-dataset) (8개) | LeRobot v3.0 | 57.3GB | ✅ 다운로드·분석 완료 |
| [Humanoid Everyday](https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday) | LeRobot v2.1 | 872GB | ✅ 다운로드·분석 완료 |
| [AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) 샘플 | LeRobot v2.1 | 89GB(해제 포함 175GB) | ✅ 다운로드·EDA 완료 (6/6 조합·스키마 5/5종) |
| [EgoDex](https://github.com/apple/ml-egodex) 테스트셋 | mp4+hdf5 | 16GB(해제 포함 35GB) | ✅ 다운로드·EDA 완료 |

## 진행 중 Task

- [신규 Task 정리 (2026-07-31)](docs/tasks-20260731.md) — AgiBot/EgoDex EDA + 3D BBox 파이프라인

## 문서

- [데이터셋 다운로드 현황 및 확인 방법](docs/dataset-download.md)
- [데이터셋 구조 분석 (전체 실측)](docs/dataset-structure.md)
- [📄 보고서 (Word)](docs/휴머노이드_데이터셋_구조분석_보고서.docx) — 샘플 이미지 10장 포함 ([docs/assets/](docs/assets/))
- [📄 EgoDex EDA 보고서 (Word)](docs/EgoDex_EDA_보고서.docx) — 스켈레톤 재투영·구조도 등 그림 16장 + 샘플 클립 2개
- [📄 AgiBot EDA 보고서 (Word)](docs/AgiBot_EDA.docx) — 전체 요약 도식·7카메라·3계층 주석 타임라인 등 그림 11장 + BBox 오버레이 클립
- [🔍 AgiBot 3자 정합성 검증 리포트](docs/agibot-verification-20260801.md) — 문서↔서버↔HF원본 대조 결과
- [🎬 에피소드 샘플 영상](docs/samples/) — Brainco GraspOreo 5번(4카메라 그리드 + 개별 클립), Humanoid Everyday 3800번(RGB+depth+LiDAR 3분할)

## 진행상황 빠른 확인

```bash
bash scripts/check_download.sh
```
