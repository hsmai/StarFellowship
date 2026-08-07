# ③ 데이터셋 정보 사용가능/불가 분리 + 입력 대체 검토 — 진행 중

**요구**: 높이·공간 정확도 개선을 위해 데이터셋의 카메라 변수·depth 등을
파이프라인 입력으로 대체할 수 있는지 분류·검토.

**완료**: 메타 전수 실측으로 3분류표 작성 —
사용 중(RGB·HE 실측 depth·추정 intrinsics) / 존재하나 미활용(HE LiDAR 프레임당
약 4,800점·IMU·odometry) / 부재(양쪽 모두 intrinsics·extrinsics, Brainco depth).
높이 대체안 3건 검토: LiDAR 평면 앵커, 테이블 평면 RANSAC 밑면 고정, Depth Pro 초점거리.

**남은 것**: 실측 비교 실험 (LiDAR 평면 vs depth 평면, RANSAC 전/후 높이 비교).

상세: [dataset-info-inventory.md](dataset-info-inventory.md)
