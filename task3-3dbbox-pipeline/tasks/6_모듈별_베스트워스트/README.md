# ⑥ 파이프라인 모듈별 베스트/워스트 사례 — 완료

**요구**: 모듈별(GroundingDINO·SAM2.1·UniDepthV2·Back-projection) 결과를
샘플로 확인해 가장 잘하는 것/못하는 것 위주로 정리.

**구조**: `<데이터셋>/<모듈>/{베스트, 워스트}/` — 파일명이 곧 사례 요지.
모듈별 이미지: GD=2D박스, SAM=픽셀마스크 오버레이, 깊이=연속 2프레임 컬러맵 비교, BP=3D박스.

**선정 방법**: 후보 약 100장 전량 렌더 → 전량 육안 검증 → 라벨 부합만 채택
(수치상 워스트지만 눈으로 멀쩡한 후보 9건 제외). 판정 근거 전체는
[육안검증_판정기록.json](육안검증_판정기록.json), 선정 기준·사례 수 사유는
[샘플선정기준.txt](샘플선정기준.txt) 참조.

**코드**: [render_candidates.py](render_candidates.py)(후보 렌더),
`tasks/도구/grab_frames*.py`(정지컷 추출), [assemble_samples.py](assemble_samples.py)(폴더 조립)
**전수 수치 분석**: [전수분석_module-best-worst.md](전수분석_module-best-worst.md)
