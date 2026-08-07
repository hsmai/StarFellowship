# 공용 도구 — 신규 task용 실험·샘플 추출 스크립트

여러 task가 함께 쓰는 스크립트. 전부 서버(`~/task3/`)에서 실행하며,
경로 상수(`ROOT`)는 각 파일 상단에 있다.

| 파일 | 용도 | 쓰는 task |
|---|---|---|
| run_probe.py | 소형 실험 3종 (bigobj/gripper/h1) | ④⑤⑧ |
| grab_frames.py | 결과 영상에서 정지컷 대량 추출 (2D/3D박스·유령 전후·G1 컷) | ⑥⑦⑧ |
| grab_frames_he보충.py | HE 쪽 후보 보충 추출 2차 | ⑥ |
| render_modules.py | 보고서 그림용 SAM 마스크·점군 렌더 | ⑥ (docx 그림) |

task 전용 코드는 각 task 폴더에 있다 (④ bigobj2.py, ⑥ render_candidates.py 등).
