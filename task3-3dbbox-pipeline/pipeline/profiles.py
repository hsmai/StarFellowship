"""태스크별 처리 프로파일 — r2를 뼈대로, 일부 태스크만 r3/r4 설정을 쓴다.

## 왜 이런 구조인가

라운드 4번(r1~r4)을 돌리며 개선했는데, 육안 검증 결과 **전체적으로는 r2가 가장 좋았다**.
r3에서 얇은 부속물을 살리려 점군 필터를 완화했더니 잘 되던 태스크가 무너졌다
(GraspOreo/머리좌 100%→67%, PickCharger 6×6×2→8×28×30cm).

그러나 **일부 태스크는 r3/r4가 확실히 나았다** — PickDrink(0%→100%),
HE deformable·Locomanip·Tool_use(유령 제거), HRI·Precision(꽃 줄기 포함).

전역 파라미터 하나로 두 요구를 동시에 만족할 수 없으므로, **태스크별로 프로파일을 지정**한다.
이것이 "문제 있는 태스크만 건드리고 나머지는 손대지 않는다"를 코드로 보장하는 방법이다.

## 사용법

    from profiles import profile_for
    prof = profile_for("GraspOreo")        # -> R2 프로파일
    prof = profile_for("PickDrink")        # -> R3 프로파일
"""
from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class Profile:
    """한 태스크를 처리할 때의 알고리즘 설정."""

    # --- 점군 정제 (크기 정확도에 가장 큰 영향) ---
    filter_pct: Tuple[float, float] = (10.0, 90.0)
    """depth 백분위 클리핑. 완화하면(2,98) 얇은 부속물이 살지만 배경·로봇팔이 들어와
    박스가 부푼다(실측: charger 6×6×2 → 8×28×30cm)."""

    filter_mad: float = 1.5
    """전경 분리 강도. 위와 같은 트레이드오프."""

    # --- 마스크 처리 ---
    components: str = "largest"
    """'largest' = 가장 큰 연결 덩어리만 (r2). 케이블·그림자를 확실히 배제하지만
    얇은 부속물(토끼 귀·꽃 줄기)도 잘린다.
    'same_depth' = 깊이가 비슷한 성분의 합집합 (r3). 부속물을 살리되 손이 섞일 위험."""

    erode_kernel: int = 5
    """마스크 침식 커널. 얇고 긴 물체(치약)에서는 3이 적합하다."""

    # --- 검출·추적 ---
    use_distractors: bool = True
    """프롬프트에 'robot hand/arm/gripper' 등을 추가할지 (r2 방식).
    로봇 팔 오검출을 막지만 프롬프트가 길어져 target 검출이 밀릴 수 있다
    (r2에서 5개 유닛이 이것 때문에 0%가 됐다)."""

    anchor_prop: bool = False
    """마스크 전파의 탐색창을 마지막 실검출에 고정할지 (r3 이후).
    자기 출력으로 창을 갱신하면 복리로 붕괴해 유령이 된다."""

    prop_limit_sec: float = 0.0
    """실검출 없이 전파를 이어갈 최대 시간(초). 0이면 무제한(r2/r3).
    r4는 1.0초로 제한했는데 유령은 사라졌지만 정상 구간도 끊겼다."""

    iou_bonus: float = 0.3
    """후보 선정 시 직전 위치와 겹치는 정도에 주는 가중. 높이면 추적이 이어지지만
    엉뚱한 물체에 락온되면 그것도 유지된다."""

    # --- 3D box ---
    mirror_z: bool = False
    """방식 B에서 깊이 방향 두께를 대칭 확장할지 (r3 이후). 단일 시점의
    구조적 두께 결손을 보정하지만 기울어진 물체에서 과대해질 수 있다."""

    min_points: int = 10
    """3D box를 만들 최소 점 수. 얇은 물체에서는 낮춰야 한다."""

    # --- 태스크 고유 ---
    max_phys_size: float = 0.75
    """target 후보의 물리 폭 상한(m). 화면을 채우는 근접 촬영에서도 유효하도록
    거리로 정규화된 값이다."""

    exclusive_tracks: bool = False
    """한 프레임에서 두 target이 같은 2D 박스를 쓰지 못하게 한다.
    'target이 사라지면 비슷한 다른 물체로 옮겨 붙는' 실패를 막는다
    (HE Precision에서 장미가 화면을 벗어나자 꽃병을 장미로 잡았다)."""


# ---------------------------------------------------------------- 기준 프로파일
R2 = Profile()                                   # 뼈대. 전체적으로 가장 좋았다.

R3 = replace(R2,
             filter_pct=(2.0, 98.0), filter_mad=3.0,
             components="same_depth", use_distractors=False,
             anchor_prop=True, mirror_z=True)

R4 = replace(R3,
             filter_pct=(10.0, 90.0), filter_mad=1.5,   # r3의 완화를 되돌린 것
             prop_limit_sec=1.0, iou_bonus=0.45)


# ------------------------------------------------- 태스크별 지정 (육안 검증 결과)
# 사용자가 라운드별 산출물을 직접 보고 확정한 매핑이다. 근거를 주석으로 남긴다.
TASK_PROFILE = {
    # ---- Brainco ----
    "GraspOreo":       ("R2", "head/wrist, A/B 모두 완벽 — 건드리지 않는다"),
    "GraspRubiksCube": ("R2", "완벽 — 건드리지 않는다"),
    "PickApple":       ("R2", "head 완벽. 집는 쪽 wrist에서 사과가 화면을 채울 때만 보완"),
    "PickCharger":     ("R2", "로봇팔 오인이 남아 있어 별도 보완 필요"),
    "PickDoll":        ("R3", "r3의 head 결과가 가장 좋았다. wrist는 검출 불가로 판단"),
    "PickDrink":       ("R3", "r3에서 물병 검출 0%→100%. head/wrist 모두 양호"),
    "PickTissues":     ("R3", "r3의 head 결과 채택"),
    "PickToothpaste":  ("R2", "전 라운드 실패. 얇은 물체 전용 보완 필요"),
    # ---- Humanoid Everyday ----
    "Articulated":     ("R2", "전반 양호. 노트북이 닫힌 뒤 박스가 작아지는 것만 보완"),
    "Basic":           ("R2", "pink toy 이동 구간 보완 필요"),
    "deformable":      ("R3", "r3 채택"),
    "HRI":             ("R4", "r4 채택"),
    "Locomanip":       ("R3", "r3 채택"),
    "Precision":       ("R4", "r4 채택. 장미가 화면을 벗어나면 꽃병을 장미로 잡는 것만 보완"),
    "Tool_use":        ("R3", "r3 채택"),
}

_BASE = {"R2": R2, "R3": R3, "R4": R4}

# 태스크 고유 보완 — 위 기준선 위에 덧붙인다. 다른 태스크에는 영향이 없다.
TASK_OVERRIDE = {
    # 얇고 긴 물체(15x4x3cm)라 5x5 침식이 마스크를 지운다. 최소 점 수도 낮춘다.
    "PickToothpaste": dict(erode_kernel=3, min_points=8),
    # 장미가 시야를 벗어나면 꽃병으로 옮겨 붙는다 — 트랙 간 박스 독점으로 막는다.
    "Precision":      dict(exclusive_tracks=True),
}


def profile_for(task: str) -> Profile:
    """태스크 이름(Brainco 태스크명 또는 HE 카테고리명)으로 프로파일을 얻는다."""
    base_name, _ = TASK_PROFILE.get(task, ("R2", "미지정 — 기본값"))
    prof = _BASE[base_name]
    ov = TASK_OVERRIDE.get(task)
    return replace(prof, **ov) if ov else prof


def describe(task: str) -> str:
    base_name, reason = TASK_PROFILE.get(task, ("R2", "미지정"))
    ov = TASK_OVERRIDE.get(task)
    s = f"{task}: {base_name} — {reason}"
    if ov:
        s += f" (+보완 {ov})"
    return s


if __name__ == "__main__":
    for t in TASK_PROFILE:
        print(describe(t))
