"""추적 + 검증 게이트 + 3D box 2가지 방식 (V5).

## 라운드2에서 드러난 것

라운드2로 HE 7/7이 전부 개선되고 Brainco head는 대부분 100%가 됐지만, 육안 검증에서
네 갈래 문제가 남았다. frames.json 15,313프레임 전수 분석으로 원인을 특정했다.

**1) 유령 잔존 — 물체가 시야를 벗어나도 작은 박스가 남는다** (사용자 최우선 지적)
`prop`(마스크 전파)이 **자기 출력으로 자기 탐색창을 갱신**하는 닫힌 고리였다.
마스크가 창을 다 못 채우면 창이 프레임당 0.97배로 줄어 복리로 붕괴한다
(실측: 161,839 → 2,112px², 85프레임). 게다가 `accept()`가 src와 무관하게 miss를
0으로 되돌려 감시견이 도달 불가능했다(180프레임 내내 confirmed).
→ **탐색창을 마지막 실검출에 앵커링**해 고리를 끊는다. 임계값이 필요 없는 구조적 수정이다.

**2) 방식 B가 실제보다 작다**
승인 프레임 7,232건 중 이력에 반영된 것이 972건(13.4%)뿐이고, **6,143건(84.9%)이
가림 판정 하나로 차단**됐다. 가림 판정이 고장난 이유는 셋이다 — 기준 깊이가 낡고,
판정 영역이 박스 전체(물체는 박스의 47%만 차지)이며, 이진 차단이었다.
→ 기준 깊이를 최근 관측 중앙값으로, 판정을 연속값으로, 이력 차단을 완화한다.
→ 단일 시점은 물체 뒷면을 볼 수 없어 두께가 구조적으로 결손된다. 방식 B에 한해
   **거울 보정**(테두리 깊이 기준 대칭 확장)으로 복원한다.

**3) 화면을 크게 채우면 못 잡는다** (손목 카메라)
픽셀 면적 상한(화면의 60%)이 근접 촬영에서 정상 물체를 잘랐다. 실측상 면적 기각
154건이 **전부 wrist, head는 0건**이다.
→ 면적비를 버리고 **깊이 정규화 물리 크기**만 쓴다. 거리에 불변이다.

**4) 회귀 5건은 검출 자체가 0** — 방해물 어휘를 프롬프트에 합치면서 target이 밀렸다.
→ 방해물 어휘를 프롬프트에서 뺀다. 가림 판정은 depth로만 해도 성립한다.

## 리뷰에서 되돌린 것

세 명의 리뷰어가 지적한 회귀 위험을 반영했다.
- prop에 **무조건 시간 예산**을 걸면 정상 구간이 대량으로 잘린다(Tool_use 0.99→0.41).
  예산을 유령 신호(마스크 붕괴·경계 접촉)와 **결합된 조건부**로 바꿨다.
- **정체성 실패**(다른 물체를 잡았다)로 기각한 후보로 앵커를 갱신하면 로봇 팔이
  앵커가 된다. 정체성 실패와 품질 실패를 나눠 앵커 갱신을 품질 실패로 한정했다.
- 가림 마진을 MAD로 유도하면 물체가 움직일 때 '추세'를 노이즈로 오인해 마진이
  폭증한다. **추세 제거 후 잔차**에서 유도한다.
"""
import numpy as np
import cv2

from geometry import (Intrinsics, backproject, filter_points, fit_box3d, draw_box3d,
                      align_mask_to_depth, same_depth_components, cluster_depth,
                      check_reprojection, mirror_extend_z)

# --- 전역 안전판 (라벨·태스크·카메라 무관) -----------------------------------
MAX_PHYS_SIZE = 0.75      # m. 후보 2D 박스가 그 깊이에서 가리키는 물리 폭 상한.
                          # 거리에 불변이라 head/wrist에 같은 값을 쓴다. 픽셀 면적비를
                          # 대체한다(면적 기각 154건이 전부 wrist였고 head는 0건이었다).
MAX_SIZE_GLOBAL = 0.80    # m. 3D 박스 최대 변 (전역 안전판)
DIAG_UP = 2.0             # 대각이 이력 중앙값의 2배 초과면 기각 (회전불변)
DIAG_LO = 0.45            # 하한. prop에만, 그리고 유령 신호와 함께일 때만 적용한다.
RX_BAND = (0.40, 1.80)    # 재투영 가로 스케일 (이력 갱신 자격용)
RY_BAND = (0.30, 2.20)    # 세로는 물체가 박스 상하단까지 안 차는 경우가 많아 더 넓게
STRONG_THR = 0.35         # 이 이상이면 확정 검출(det), 아래는 약검출(weak)
GHOST_MASK_RATIO = 0.15   # 마스크가 앵커 창 면적의 이 비율 미만이면 유령 신호
COAST_SEC = 2.5           # 방식 B가 관측 없이 이어갈 수 있는 최대 시간(초)


def iou2d(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def union_box(boxes):
    a = np.array(boxes, dtype=float)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max())]


def inflate(box, ratio, W, H):
    x1, y1, x2, y2 = box
    w, h = (x2 - x1) * ratio / 2, (y2 - y1) * ratio / 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [max(0, cx - w), max(0, cy - h), min(W - 1, cx + w), min(H - 1, cy + h)]


def clip_mask_to_box(mask, box, ratio=1.15):
    H, W = mask.shape
    x1, y1, x2, y2 = [int(v) for v in inflate(box, ratio, W, H)]
    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def phys_width(box2d, z, K):
    """2D 박스가 깊이 z에서 가리키는 물리 폭·높이(m). 거리에 불변한 크기 척도."""
    x1, y1, x2, y2 = box2d
    return (x2 - x1) * z / max(K.fx, 1e-6), (y2 - y1) * z / max(K.fy, 1e-6)


def border_touch(box2d, W, H, tol=2.0):
    """화면 4변 중 몇 변에 접하는가. 시야 이탈의 직접 관측량."""
    x1, y1, x2, y2 = box2d
    return int(x1 <= tol) + int(y1 <= tol) + int(x2 >= W - 1 - tol) + int(y2 >= H - 1 - tol)


# 기각 사유 분류 — 정체성 실패에서는 앵커를 갱신하면 안 된다.
# (로봇 팔을 charger로 오인한 후보 109프레임이 앵커가 되면 그 위에 prop이 계속 생긴다)
IDENTITY_FAIL = ("diag>", "reproj", "physbox", "absmax", "fill=", "depth_shift", "border")


def is_identity_fail(reason):
    return any(reason.startswith(p) or p in reason for p in IDENTITY_FAIL)


class Track:
    """라벨 1개의 시간축 상태.

    설계 원칙 셋:
    - **탐색창은 실검출에만 앵커링**한다. 전파 결과로 창을 갱신하면 되먹임으로 붕괴한다.
    - **그린다**와 **이력에 넣는다**를 분리한다. 게이트가 자기 이력과 비교하므로,
      이력을 무비판 갱신하면 오염이 자기를 강화한다.
    - 방식 A는 **관측된 것만**, 방식 B는 **가림을 보정한 실제 크기**. 이력도 따로 쌓는다.
    """
    HIST = 40

    def __init__(self, label, is_target=False, fps=10.0):
        self.label = label
        self.is_target = is_target
        self.fps = fps
        self.sizes = []            # 관측 크기 이력 (게이트·방식 A용)
        self.sizes_mir = []        # 거울 보정 크기 이력 (방식 B용)
        self.recent_dmed = []      # 최근 실검출의 전경 depth 중앙값 (가림 기준)
        self.center = None
        self.prev_center = None
        self.box2d = None
        self.anchor_box2d = None   # 마지막 실검출 박스 — prop 탐색창의 기준
        self.anchor_vel = np.zeros(2)
        self.frames_since_evi = 0  # 실검출(det/weak) 이후 경과 프레임
        self.ema_c = self.ema_s = None
        self.ema_c_b = self.ema_s_b = None
        self.miss = 0
        self.n_acc = self.n_rej = 0
        self.n_occluded = self.n_coast = self.n_ghost_block = 0
        self.n_prop_acc = self.n_det_acc = 0
        self.confirmed = False
        self._det_hits = []
        self._weak_hits = []
        self.det_gaps = []
        self.det_scores = []

    # ---- 이력 조회
    def med_size(self):
        return np.median(np.array(self.sizes), axis=0) if self.sizes else None

    def size_ref(self):
        """방식 B의 크기 기준. 가려진 관측이 섞여 중앙값이 작아지므로 **상위 분위**를 쓴다.
        (사용자 지적: "실제 크기에 비해 3d box가 좀 작게 잡히는 경향")"""
        src = self.sizes_mir if len(self.sizes_mir) >= 3 else self.sizes
        if not src:
            return None
        a = np.array(src)
        return np.percentile(a, 75, axis=0) if len(a) >= 6 else a.max(axis=0)

    def med_diag(self):
        ms = self.med_size()
        return float(np.linalg.norm(ms)) if ms is not None else None

    def depth_ref(self):
        """가림 판정의 기준 깊이. 낡은 값을 쓰면 손이 없는데도 가림으로 포화된다."""
        return float(np.median(self.recent_dmed)) if self.recent_dmed else None

    def depth_noise(self):
        """depth 잡음 추정 — 1차 추세를 뺀 잔차에서 구한다.
        MAD를 그대로 쓰면 물체가 접근/이동하는 구간의 '추세'를 잡음으로 오인해
        마진이 폭증하고 가림 판정이 통째로 꺼진다."""
        n = len(self.recent_dmed)
        if n < 4:
            return None
        y = np.array(self.recent_dmed, dtype=float)
        x = np.arange(n, dtype=float)
        k, b = np.polyfit(x, y, 1)
        res = y - (k * x + b)
        return float(np.median(np.abs(res)) * 1.4826)

    def det_budget(self):
        """관측 없이 트랙을 유지할 프레임 수. 상수가 아니라 실측 det 간격에서 유도한다."""
        if len(self.det_gaps) >= 3:
            return max(int(3.0 * float(np.median(self.det_gaps))), int(4.0 * self.fps))
        return int(8.0 * self.fps)

    # ---- 게이트
    def gate(self, b, mask, box2d, K, WH, src, mask_px, ghost_sig):
        """반환 (통과, 사유, 진단). 하드 기각은 물리적으로 불가능하거나 정체성이 틀린 것만."""
        W, H = WH
        diag = {}
        du, dv, rx, ry, inside = check_reprojection(b, box2d, K, 0)
        barea = max((box2d[2] - box2d[0]) * (box2d[3] - box2d[1]), 1.0)
        z = float(b.center[2])
        pw, ph = phys_width(box2d, z, K)
        diag.update(du=round(du, 1), dv=round(dv, 1), rx=round(rx, 3), ry=round(ry, 3),
                    area_ratio=round(barea / (W * H), 4), phys_box=round(max(pw, ph), 3),
                    border=border_touch(box2d, W, H))

        # 거리 정규화 물리 크기 — 픽셀 면적비를 대체한다(카메라 거리에 불변)
        if max(pw, ph) > MAX_PHYS_SIZE:
            return False, f"physbox>{max(pw,ph):.2f}m", diag
        if np.max(b.size) > MAX_SIZE_GLOBAL:
            return False, f"absmax>{MAX_SIZE_GLOBAL}", diag
        if not inside:
            return False, "reproj", diag
        fill = mask.sum() / barea
        diag["fill"] = round(float(fill), 3)
        if fill > 0.995:
            return False, f"fill={fill:.2f}", diag

        dm = self.med_diag()
        d = float(np.linalg.norm(b.size))
        diag["diag"] = round(d, 4)
        if dm is not None and len(self.sizes) >= 5:
            diag["d_over_dm"] = round(d / max(dm, 1e-9), 3)
            if d > DIAG_UP * dm:
                return False, f"diag>{DIAG_UP}x", diag

        # 전파 프레임은 이력 크기와 크게 다르면 무조건 기각한다(유령의 주 형태).
        if src == "prop":
            r = diag.get("d_over_dm")
            if r is not None and not (0.55 <= r <= 1.8):
                self.n_ghost_block += 1
                return False, "ghost_size", diag
        # 유령 신호(마스크 붕괴·경계 접촉·깊이 이탈)가 동반되면 추가로 차단한다.
        if src == "prop" and ghost_sig:
            r = diag.get("d_over_dm")
            if r is not None and r < DIAG_LO:
                self.n_ghost_block += 1
                return False, "ghost_shrink", diag
            if diag["border"] >= 2 and diag["area_ratio"] < 0.15:
                self.n_ghost_block += 1
                return False, "ghost_border", diag
            dref = self.depth_ref()
            if dref is not None and self.dmed_now is not None:
                if not (0.6 * dref <= self.dmed_now <= 1.6 * dref):
                    self.n_ghost_block += 1
                    return False, "ghost_depth", diag
        return True, "", diag

    def update_ok(self, src, occ_frac, diag, n_points, WH):
        """이 관측이 크기 '기준'이 될 자격이 있는가.

        라운드2에서는 가림 판정이 조금이라도 서면 무조건 차단했는데, 그 결과 승인
        7,232건 중 972건(13.4%)만 이력에 들어가 방식 B의 크기가 작아졌다.
        가림이 심한 경우만 배제하고, 나머지는 형상 정합으로 거른다."""
        if src not in ("det", "weak"):
            return False
        if occ_frac >= 0.5:
            return False
        rx, ry = diag.get("rx", 0), diag.get("ry", 0)
        if not (RX_BAND[0] <= rx <= RX_BAND[1] and RY_BAND[0] <= ry <= RY_BAND[1]):
            return False
        W, H = WH
        if n_points < max(30, 0.00015 * W * H):
            return False
        r = diag.get("d_over_dm")
        if r is not None:
            return 0.7 <= r <= 1.4
        # cold start: 이력이 얕을 때는 서로 합의하는 관측만 받는다
        if self.sizes:
            d = float(np.linalg.norm(np.array(diag.get("diag", 0.0))))
            dm = self.med_diag()
            if dm and not (0.6 <= d / max(dm, 1e-9) <= 1.7):
                return False
        return True

    def accept(self, b, box2d, dmed, size_mir, update_history, src, is_evidence):
        if update_history:
            self.sizes.append(np.array(b.size)); self.sizes = self.sizes[-self.HIST:]
            if size_mir is not None:
                self.sizes_mir.append(np.array(size_mir)); self.sizes_mir = self.sizes_mir[-self.HIST:]
        if is_evidence and dmed is not None:
            # 기준 깊이는 이력 자격과 무관하게 실검출이면 항상 갱신한다.
            # (낡은 기준을 쓰면 손이 화면 밖인데도 가림으로 포화된다)
            self.recent_dmed.append(dmed); self.recent_dmed = self.recent_dmed[-7:]
        self.prev_center = self.center
        self.center = np.array(b.center)
        a = 0.45
        self.ema_c = self.center if self.ema_c is None else (1 - a) * self.ema_c + a * self.center
        s = np.array(b.size)
        self.ema_s = s if self.ema_s is None else (1 - a) * self.ema_s + a * s
        if is_evidence:
            prev = self.anchor_box2d
            if prev is not None:
                pc = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2])
                nc = np.array([(box2d[0] + box2d[2]) / 2, (box2d[1] + box2d[3]) / 2])
                self.anchor_vel = 0.5 * self.anchor_vel + 0.5 * (nc - pc) / max(self.frames_since_evi, 1)
            self.anchor_box2d = list(box2d)
            self.box2d = list(box2d)
            if self.frames_since_evi > 0:
                self.det_gaps.append(self.frames_since_evi)
            self.frames_since_evi = 0
            self.miss = 0
            self.n_det_acc += 1
        else:
            self.frames_since_evi += 1
            self.miss += 1                      # prop은 예산을 소모한다
            self.n_prop_acc += 1
        self.n_acc += 1

    def reject(self, reason="", had_obs=False):
        self.n_rej += 1
        self.frames_since_evi += 1
        # 정체성 실패(다른 물체를 잡음)로는 앵커를 갱신하지 않는다 — 갱신하면
        # 로봇 팔 후보가 앵커가 되어 그 위에 prop이 계속 생성된다.
        if not had_obs or is_identity_fail(reason):
            self.miss += 1
        if self.miss > self.det_budget():
            seed = self.med_size()
            seed_m = self.size_ref()
            self.box2d = None; self.anchor_box2d = None; self.center = None
            self.prev_center = None; self.anchor_vel = np.zeros(2)
            self.sizes = [seed] if seed is not None else []
            self.sizes_mir = [seed_m] if seed_m is not None else []
            self.recent_dmed = []
            self.confirmed = False
            self._det_hits = []; self._weak_hits = []

    def note_obs(self, fidx, center3d, src, score):
        win = int(1.5 * self.fps)
        hits = self._det_hits if src == "det" else self._weak_hits
        hits.append((fidx, np.array(center3d)))
        hits[:] = [h for h in hits if fidx - h[0] <= win]
        if score is not None:
            self.det_scores.append(float(score))
        if self.confirmed:
            return
        def agree(hs, k):
            if len(hs) < k:
                return False
            c = np.array([h[1] for h in hs[-k:]])
            return float(np.max(np.linalg.norm(c - c.mean(axis=0), axis=1))) <= 0.15
        if agree(self._det_hits, 2):
            self.confirmed = True
        elif len(self._det_hits) >= 1 and agree(self._weak_hits, 3):
            self.confirmed = True      # 고임계 det이 평생 1회뿐인 태스크를 위한 경로

    def coast(self, mask_center_3d):
        """방식 B 전용 — 관측이 끊긴 구간을 실제 크기 + 위치 추정으로 채운다.
        무한히 이어지지 않도록 시간 상한을 둔다(사용자 최우선 지적: 사라져야 한다)."""
        sr = self.size_ref()
        if sr is None or self.center is None:
            return None
        if self.frames_since_evi > int(COAST_SEC * self.fps):
            return None
        if mask_center_3d is not None:
            c = np.array(mask_center_3d)
        elif self.prev_center is not None:
            c = self.center + (self.center - self.prev_center)
        else:
            c = self.center.copy()
        self.n_coast += 1
        return c, sr.copy()

    def amodal(self, b, size_mir, occ_frac):
        """방식 B — 가려진 부분을 고려한 실제 크기.
        크기는 거울 보정된 이력의 상위 분위(size_ref)를 기준으로 하되, 현재 관측이
        더 크면 그것을 쓴다. 위치는 보이는 조각이 박스에 들어가도록 최소 이동."""
        cen = np.array(b.center)
        cur = np.array(size_mir if size_mir is not None else b.size)
        sr = self.size_ref()
        if sr is None or len(self.sizes) < 3:
            return cen, cur
        size = np.maximum(sr, cur) if occ_frac < 0.25 else sr.copy()
        vmin, vmax = cen - np.array(b.size) / 2, cen + np.array(b.size) / 2
        new_c = cen.copy()
        for i in range(3):
            lo, hi = new_c[i] - size[i] / 2, new_c[i] + size[i] / 2
            if vmin[i] < lo:
                new_c[i] -= (lo - vmin[i])
            elif vmax[i] > hi:
                new_c[i] += (vmax[i] - hi)
        return new_c, size

    def ema_b(self, c, s):
        a = 0.45
        self.ema_c_b = c if self.ema_c_b is None else (1 - a) * self.ema_c_b + a * c
        self.ema_s_b = s if self.ema_s_b is None else (1 - a) * self.ema_s_b + a * s
        return self.ema_c_b, self.ema_s_b


class EpisodeTracker:
    """에피소드 1개 처리기. targets = {매칭키: (표시라벨, is_target)}"""

    def __init__(self, targets, det, seg, prompt, low_thr=0.20, fps=10.0):
        self.targets, self.tracks, self.key_alias = {}, {}, {}
        for k, v in targets.items():
            label, is_tgt = (v[0], bool(v[1])) if isinstance(v, (list, tuple)) else (v, False)
            prim = next((kk for kk, ll in self.targets.items() if ll == label), None)
            if prim is None:
                self.targets[k] = label
                self.tracks[k] = Track(label, is_target=is_tgt, fps=fps)
                self.key_alias[k] = k
            else:
                self.key_alias[k] = prim
        # 방해물 어휘는 프롬프트에 넣지 않는다 — 라운드2에서 프롬프트가 길어져
        # target 검출이 밀렸고 5개 유닛이 통째로 0%가 됐다.
        # 가림 판정은 depth만으로 성립한다(물체 실루엣 안에서 물체보다 앞선 픽셀).
        self.prompt = prompt.rstrip().rstrip(".").strip() + " ."
        self.det, self.seg = det, seg
        self.low_thr = low_thr
        self.fps = fps
        self.fidx = -1

    def _match(self, key, boxes, phrases, scores, WH, K, depth, dscale):
        """후보 선정. 같은 물체를 가리키는 별칭 박스들은 **합집합**으로 확장한다
        (꽃봉오리 박스 ∪ '줄기 포함' 박스 — 사용자 지적: 꽃 줄기가 빠진다)."""
        tr = self.tracks[key]
        W, H = WH
        aliases = [k for k, prim in self.key_alias.items() if prim == key] or [key]
        cands = []
        for b, p, s in zip(boxes, phrases, scores):
            if not any(a in str(p) for a in aliases):
                continue
            x1, y1, x2, y2 = [int(max(0, v)) for v in b]
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                d = depth[y1:y2, x1:x2]
                d = d[np.isfinite(d) & (d > 0)]
                if d.size > 20:
                    z = float(np.median(d)) * dscale
                    pw, ph = phys_width(b, z, K)
                    if max(pw, ph) > MAX_PHYS_SIZE:
                        continue
            bonus = 0.45 * iou2d(b, tr.box2d) if tr.box2d is not None else 0.0
            cands.append((s + bonus, float(s), list(map(float, b)), str(p)))
        if not cands:
            return None
        cands.sort(key=lambda x: -x[0])
        _, best_s, best_b, best_p = cands[0]
        merged = [best_b]
        for _, s2, b2, p2 in cands[1:]:
            if iou2d(best_b, b2) > 0.3:       # 같은 물체의 다른 표현 — 합쳐서 전체를 덮는다
                merged.append(b2)
        box = union_box(merged) if len(merged) > 1 else best_b
        return box, best_s, best_p, len(merged)

    def step(self, img, depth, K, dscale, align_dx=0):
        self.fidx += 1
        H, W = img.shape[:2]
        # 저임계 단일 패스 — 강/약을 트래커 내부에서 나눈다(검출 호출 1회로 감소)
        old_thr = self.det.box_thr
        self.det.box_thr = self.low_thr
        boxes, phrases, scores = self.det(img, self.prompt)
        self.det.box_thr = old_thr

        results, pend = [], {}
        for key in self.targets:
            tr = self.tracks[key]
            m = self._match(key, boxes, phrases, scores, (W, H), K, depth, dscale)
            if m is not None:
                box, sc, phr, ncomp = m
                pend[key] = (box, "det" if sc >= STRONG_THR else "weak", sc, phr, ncomp)
            elif (tr.confirmed and tr.anchor_box2d is not None
                  and tr.frames_since_evi < int(1.0 * self.fps)):
                # 실검출 없이 1초를 넘기면 전파하지 않는다. 사용자 지적:
                # "target object가 카메라에 안 잡힐 때 다른 물체에 3d box를 씌운다",
                # "물건을 집지 않는 쪽 wrist에서 공백에 3d box를 잡는다".
                # 방식 B의 coast가 그 구간을 명시적 추정으로 덮으므로 정보 손실은 없다.
                # 전파 창은 **마지막 실검출**에 앵커링한다. 자기 출력으로 갱신하면
                # 창이 프레임당 0.97배로 붕괴해 유령이 된다(실측 161,839->2,112px²).
                dt = max(tr.frames_since_evi, 1)
                shifted = [tr.anchor_box2d[0] + tr.anchor_vel[0] * dt,
                           tr.anchor_box2d[1] + tr.anchor_vel[1] * dt,
                           tr.anchor_box2d[2] + tr.anchor_vel[0] * dt,
                           tr.anchor_box2d[3] + tr.anchor_vel[1] * dt]
                pend[key] = (inflate(shifted, 1.12 + 0.01 * dt, W, H), "prop", 0.0, "", 1)

        masks = {}
        if pend:
            keys = list(pend.keys())
            bxs = np.array([pend[k][0] for k in keys], dtype=np.float32)
            ms = self.seg(img, bxs)
            for k, m in zip(keys, ms):
                masks[k] = m.astype(bool)

        for key in self.targets:
            tr = self.tracks[key]
            base = dict(key=key, label=tr.label, frame=self.fidx,
                        track_state="confirmed" if tr.confirmed else "unconfirmed",
                        fse=tr.frames_since_evi)
            if key not in pend:
                rec = dict(base, src="none", accepted=False, reason="no_candidate")
                cb = tr.coast(None) if tr.confirmed else None
                tr.reject("no_candidate", had_obs=False)
                if cb is not None:
                    c, s = tr.ema_b(*cb)
                    rec.update(accepted_amodal=True, coasting=True, evidence="prop",
                               center_amodal=list(map(float, c)), size_amodal=list(map(float, s)),
                               conf=round(float(np.exp(-tr.frames_since_evi / 8.0)), 3))
                results.append(rec)
                continue

            box2d, src, score, phrase, ncomp = pend[key]
            is_evi = src in ("det", "weak")
            m = masks.get(key)
            mk = clip_mask_to_box(m, box2d, 1.15)
            # 얇은 부속물(토끼 귀·꽃 줄기)이 별개 성분으로 잘리지 않게, 깊이가 비슷한
            # 성분은 모두 유지한다. 케이블처럼 깊이가 다른 부속물은 여전히 배제된다.
            # 부속물 허용 깊이차를 작게 고정한다. 두께 비례로 두면 두꺼운 물체에서
            # 허용치가 커져 손까지 합쳐진다.
            mk = same_depth_components(mk, depth, dscale, box2d, dz=0.02)
            kk = 3 if min(box2d[2] - box2d[0], box2d[3] - box2d[1]) < 80 else 5
            mk_e = cv2.erode(mk.astype(np.uint8), np.ones((kk, kk), np.uint8), 1).astype(bool)
            mk_a = align_mask_to_depth(mk_e, dx=align_dx)
            mask_px = int(mk_a.sum())
            dmed = float(np.median(depth[mk_a])) * dscale if mask_px > 0 else None
            tr.dmed_now = dmed

            # --- 가림 정도: 물체 실루엣 안에서 '물체보다 앞선' 픽셀 비율 ---
            occ_frac = 0.0
            dref, dnoise = tr.depth_ref(), tr.depth_noise()
            if mask_px > 30 and dref is not None and tr.frames_since_evi <= int(1.0 * self.fps):
                ms_ = tr.med_size()
                marg = max(0.02, 3.0 * (dnoise or 0.0), 0.4 * float(ms_[2]) if ms_ is not None else 0.0)
                dz = depth[mk_a] * dscale
                dz = dz[np.isfinite(dz) & (dz > 0)]
                if dz.size > 20:
                    occ_frac = float((dz < dref - marg).mean())

            ghost_sig = False
            if src == "prop":
                aarea = max((tr.anchor_box2d[2] - tr.anchor_box2d[0]) *
                            (tr.anchor_box2d[3] - tr.anchor_box2d[1]), 1.0) if tr.anchor_box2d else 1.0
                ghost_sig = (mask_px < GHOST_MASK_RATIO * aarea) or \
                            (border_touch(box2d, W, H) >= 2) or \
                            (tr.frames_since_evi > int(2.0 * self.fps))

            # 필터 강도는 r2 값으로 되돌린다. r3에서 (2,98)+mad3.0으로 완화했더니
            # 얇은 부속물은 살았지만 배경·로봇팔이 함께 들어와 크기가 부풀었다
            # (charger 6x6x2 -> 8x28x30cm, oreo head 100% -> 67%).
            # 부속물 복원은 same_depth_components가 담당하므로 여기서 완화할 필요가 없다.
            pts = filter_points(backproject(depth, K, mk_a, depth_scale=dscale,
                                            valid_range=(0.1, 5.0)),
                                percentile=(10, 90), foreground_mad=1.5)
            pts = cluster_depth(pts, gap=0.06)
            b = fit_box3d(pts, pct=1.0)

            rec = dict(base, src=src, det_score=round(float(score), 3), phrase=phrase,
                       n_merged=ncomp, box2d=box2d, mask_px=mask_px,
                       dmed=None if dmed is None else round(dmed, 4),
                       occluder_frac=round(occ_frac, 3), ghost_sig=bool(ghost_sig),
                       evidence=src if is_evi else "prop")

            if b is None:
                rec.update(accepted=False, reason="no_points")
                cb = tr.coast(None) if tr.confirmed else None
                tr.reject("no_points", had_obs=True)
                if cb is not None:
                    c, s = tr.ema_b(*cb)
                    rec.update(accepted_amodal=True, coasting=True,
                               center_amodal=list(map(float, c)), size_amodal=list(map(float, s)),
                               conf=round(float(np.exp(-tr.frames_since_evi / 8.0)), 3))
                results.append(rec)
                continue

            if is_evi:
                tr.note_obs(self.fidx, b.center, src, score)
            # 단일 시점은 물체 뒷면을 못 보므로 깊이 방향 두께가 구조적으로 결손된다.
            # 방식 B에만 거울 보정을 적용한다(A의 '관측된 것만' 정의를 지키기 위해).
            size_mir = mirror_extend_z(b, pts)
            ok, why, diag = tr.gate(b, mk_a, box2d, K, (W, H), src, mask_px, ghost_sig)
            if ok and not tr.confirmed and src != "det":
                ok, why = False, "unconfirmed"

            rec.update(n_points=int(b.n_points), raw_size=list(map(float, b.size)),
                       size_mirror=list(map(float, size_mir)), occluded=bool(occ_frac >= 0.25), **diag)
            if occ_frac >= 0.25:
                tr.n_occluded += 1

            if ok:
                upd = tr.update_ok(src, occ_frac, diag, b.n_points, (W, H))
                cB, sB = tr.amodal(b, size_mir, occ_frac)
                tr.accept(b, box2d, dmed, size_mir, upd, src, is_evi)
                cBe, sBe = tr.ema_b(cB, sB)
                rec.update(accepted=True, reason="", updated_history=bool(upd),
                           center=list(map(float, tr.ema_c)), size=list(map(float, tr.ema_s)),
                           center_amodal=list(map(float, cBe)), size_amodal=list(map(float, sBe)),
                           accepted_amodal=True, coasting=False,
                           conf=1.0 if is_evi else round(float(np.exp(-tr.frames_since_evi / 8.0)), 3))
            else:
                # 정체성 실패면 그 후보 위치를 방식 B가 이어받지 않는다
                cb = tr.coast(None if is_identity_fail(why) else np.array(b.center)) if tr.confirmed else None
                tr.reject(why, had_obs=True)
                rec.update(accepted=False, reason=why)
                if cb is not None:
                    c, s = tr.ema_b(*cb)
                    rec.update(accepted_amodal=True, coasting=True,
                               center_amodal=list(map(float, c)), size_amodal=list(map(float, s)),
                               conf=round(float(np.exp(-tr.frames_since_evi / 8.0)), 3))
            results.append(rec)
        return results

    def draw(self, vis, results, K, mode="A"):
        from geometry import Box3D
        pal = {"det": (0, 220, 120), "weak": (0, 200, 255), "prop": (255, 160, 40),
               "none": (255, 160, 40)}
        for r in results:
            if mode == "B":
                if not r.get("accepted_amodal"):
                    continue
                c, s = np.array(r["center_amodal"]), np.array(r["size_amodal"])
                conf = float(r.get("conf", 1.0))
                est = r.get("coasting") or r["src"] == "prop"
                col = (60, 120, 255) if est else ((80, 200, 255) if r.get("occluded") else pal[r["src"]])
                th = 1 if conf < 0.6 else 2
                tag = " (추정)" if est else (" (가림보정)" if r.get("occluded") else "")
            else:
                if not r.get("accepted"):
                    continue
                c, s = np.array(r["center"]), np.array(r["size"])
                col = pal[r["src"]]
                th = 1 if r["src"] == "prop" else 2
                tag = " (전파)" if r["src"] == "prop" else ""
            b = Box3D(center=c, size=s, min_xyz=c - s / 2, max_xyz=c + s / 2,
                      n_points=int(r.get("n_points", 0)))
            draw_box3d(vis, b, K, color=col, thickness=th,
                       label=f"{r['label']} {s[0]*100:.0f}x{s[1]*100:.0f}x{s[2]*100:.0f}cm{tag}")
        return vis

    def stats(self, n_frames):
        out = {}
        for k, tr in self.tracks.items():
            ms, sr = tr.med_size(), tr.size_ref()
            out[tr.label] = dict(
                is_target=tr.is_target, confirmed=tr.confirmed,
                accepted=tr.n_acc, rejected=tr.n_rej,
                coverage=round(tr.n_acc / max(n_frames, 1), 3),
                # 실검출 근거로만 승인된 비율 — 전파 승인을 제외한다.
                # (라운드2 stats는 83%가 유령인 유닛을 0.98로 보고했다)
                coverage_det=round(tr.n_det_acc / max(n_frames, 1), 3),
                prop_accepted=tr.n_prop_acc, ghost_blocked=tr.n_ghost_block,
                coasting_frames=tr.n_coast,
                coverage_B=round((tr.n_acc + tr.n_coast) / max(n_frames, 1), 3),
                occluded_frames=tr.n_occluded, n_history=len(tr.sizes),
                det_score_median=round(float(np.median(tr.det_scores)), 3) if tr.det_scores else None,
                size_median=None if ms is None else [round(float(x), 4) for x in ms],
                size_ref=None if sr is None else [round(float(x), 4) for x in sr])
        return out
