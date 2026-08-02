# AgiBot World 2026 — 3자 정합성 검증 (2026-08-01)

> 검증 대상: **EDA 보고서(AgiBot_EDA.docx) ↔ 서버 다운로드본 ↔ HF 원본 저장소**
> 방법: 문서의 모든 수치 주장을 추출해 서버 실측(8태스크 전수) 및 HF API·README와 대조.

## 결론

구조·스키마·주석에 관한 주장은 **전부 정확**. 용량 관련 수치 4건이 스키마 B(task_3405) 추가 다운로드 이전 시점 값이라 갱신 필요 → **문서 반영 완료**.

## ✅ 검증 통과 항목

| 항목 | 문서 | 서버 실측 | HF 원본 |
|---|---|---|---|
| 로봇 `g2a` 단일 기종 | ✓ | 8/8 태스크 | README "AGIBOT G2" |
| LeRobot v2.1 | ✓ | `codebase_version: v2.1` | README 명시 |
| 라이선스 CC BY-NC-SA 4.0 / gated 아님 | ✓ | — | HF API 확인 |
| 태스크 8개 / 429 에피소드 / 558,180 프레임(5.2h) | ✓ | 정확히 일치 | — |
| 카메라 7종 + 해상도(768×960 / 400×640 / 528×640) | ✓ | `info.json` shape 일치 | README 키 목록과 정합 |
| depth = PNG 무손실 gray16be / RGB = AV1 | ✓ | ffprobe 확인 | — |
| Task Frame 672 / 2D BBox 814 / 스킬 세그먼트 4,132 | ✓ | 정확히 일치 | 3계층 구조 README 일치 |
| state 구성 (관절14·머리3·허리5·EE 8+6·카메라 extrinsics 72·mode 23·베이스 7) | ✓ | `field_descriptions` 전부 일치 | README schema 설명 일치 |
| action 구성 (그리퍼2 + EE14 + 관절14 + 머리3 + 허리5 + 베이스 2/6) | ✓ | 8/8 태스크 일치 | — |
| reward 컬럼 부재 | ✓ | parquet 컬럼 7개 전수 확인 | — |
| depth 0값 ~20% | ✓ | 21% | — |
| `episodes_stats` 비디오 통계 0 | ✓ | — | README "set to 0" 명시 |
| 스키마 5종 · tar(수집 구간)별로 상이 | ✓ | 8태스크 실측 일치 | — |

## ⚠️ 수정한 수치 4건

| # | 위치 | 기존 | 수정 | 원인 |
|---|---|---|---|---|
| ① | 1장 요약 / 그림 1·2 | 표본 175GB | **tar 93GB → 해제 후 187GB** | `du -sh`의 175G는 GiB 단위. GB 환산 시 187GB |
| ② | 1장 요약 / 그림 2 | 전체의 1.6% | **0.9%** | 해제본(187GB) vs tar(10.7TB) 비교 오류. tar끼리 비교하면 93GB/10.7TB |
| ③ | 4장 그림 6 캡션 | depth 70GB / 86% | **79GB / 85%** | 스키마 B(task_3405, +9GB) 추가 전 값 |
| ④ | 4장 그림 6 캡션 | RGB ~2MB, depth ~300MB | **평균 RGB 3.6MB, depth 185MB(최대 550MB)** | 단일 에피소드 값 → 429개 평균으로 교체 |
| ⑤ | 7장 표 | task_5015 66초 | **67초** | 66.5초 반올림 (task_3777과 표기 불일치) |

## 📌 참고 — 판단 유보 1건

**전체 용량 "10.7TB"** 표기는 유지. 근거:
- tar.gz 313개 크기 합산 = **10.7TB** (실제 내려받는 용량)
- HF API `usedStorage` = **13.27TB** (LFS 히스토리·중복 blob 포함한 내부 관리 용량)

사수님이 HF 페이지에서 다른 숫자를 볼 수 있으므로, 그림 2 하단에 각주를 추가함.

## 📎 참고 — README와의 차이 (문서가 맞음)

HF README의 "Common camera keys" 목록에는 `head_stereo_left_color`, `head_stereo_right_color`, `head_back_fisheye_color` 등이 등장하나, **37개 태스크 전수 census 결과 AgiBotWorld2026에 실제 존재하는 카메라는 7종뿐**. README가 AgiBot 시리즈 공통 문서라 타 데이터셋 키까지 나열한 것으로 판단됨. 문서의 "카메라 7종, 전 태스크 동일" 주장이 정확.

## 검증 산출물

- 서버 census 전문: `/data2/humanoid_dataset_isangmin/agibot_full_census.json` (37태스크 스키마·카메라·주석)
- 스키마 census 로그: [agibot-schema-census.txt](agibot-schema-census.txt)
