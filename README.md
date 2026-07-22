# StarFellowship

휴머노이드 로봇 real-world 데이터셋 분석 프로젝트.

## 목표

- Real-world 휴머노이드 데이터셋의 구조 파악 (포함된 정보, 객체 간 관계 정보 등)
- 데이터셋 정리 및 base 적용, 활용처 탐색
- Sim이 아닌 real 데이터에서 필요한 정보(메타데이터 부재)를 추출하는 방법 조사

## 대상 데이터셋

| 데이터셋 | 포맷 | 크기 | 상태 |
|---|---|---|---|
| [UniFoLM G1 Brainco 컬렉션](https://huggingface.co/collections/unitreerobotics/unifolm-g1-brainco-dataset) (8개) | LeRobot v3.0 | 데이터셋당 ~12GB | 다운로드 중 (1/8 완료) |
| [Humanoid Everyday](https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday) | LeRobot v2.0 | 935GB | 대기 (최후순위) |

## 문서

- [데이터셋 다운로드 현황 및 확인 방법](docs/dataset-download.md)
- [데이터셋 구조 분석](docs/dataset-structure.md)

## 진행상황 빠른 확인

```bash
bash scripts/check_download.sh
```
