"""V2 공통 모듈 — 검출 끊김·박스 튐 문제를 잡기 위한 추적 + 3중 검증 게이트.

기존 문제 (A_track 진단 결과):
  - 프레임 독립 검출: 로봇이 물체를 옮길 때 검출이 끊김 (오레오 52%)
  - 단순 박스 전파: 물체는 움직이는데 박스는 고정 -> 배경을 마스크로 잡아 박스 폭발
    (30cm 초과 폭발 51건 중 50건이 박스 전파 프레임)

V2 설계:
  1) 검출 우선. 놓치면 저임계 재검출(이전 위치 근처만 인정) -> SAM 마스크 전파 순.
  2) 마스크 전파 시 mask -> tight box 로 박스를 매 프레임 갱신 (움직임을 따라감).
  3) 결과가 어떻든 3중 게이트 통과 못 하면 박스를 그리지 않음 (틀린 박스 > 빈 프레임).
     G1 크기: 각 변이 이력 중앙값의 2.5배 초과 또는 부피 4배 초과 -> 기각
     G2 이동: 3D 중심이 직전 대비 0.25m 초과 점프 -> 기각
     G3 마스크: 박스 대비 채움비 10~98% 벗어남 / 전파 시 depth 중앙값 60~160% 벗어남 -> 기각
  4) 렌더링은 EMA 스무딩(중심·크기) -> 화면 떨림 제거.
"""
import numpy as np
import cv2

from geometry import (Intrinsics, backproject, filter_points, fit_box3d,
                      draw_box3d, align_mask_to_depth, largest_component, cluster_depth)

# 조작 대상 물체의 상식적 상한(m). 사람이 한 손으로 드는 물체라 이보다 크면 배경 혼입이다.
# plate처럼 넓은 물체를 위해 라벨별 예외를 둔다.
MAX_SIZE_DEFAULT = 0.35
MAX_SIZE_BY_LABEL = {"plate": 0.45, "doll": 0.45, "container": 0.60, "person": 2.2,
                     "desk": 2.0, "table": 2.0, "monitor": 0.8, "whiteboard": 1.5,
                     "chair": 1.2, "laptop": 0.6, "bag": 0.6, "box": 0.6}


def iou2d(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def inflate(box, ratio, W, H):
    x1, y1, x2, y2 = box
    w, h = (x2 - x1) * ratio / 2, (y2 - y1) * ratio / 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [max(0, cx - w), max(0, cy - h), min(W - 1, cx + w), min(H - 1, cy + h)]


def clip_mask_to_box(mask, box, ratio=1.15):
    """마스크를 검출 박스 주변으로 제한 — 케이블 등 길게 이어진 배경 유입 차단"""
    H, W = mask.shape
    x1, y1, x2, y2 = [int(v) for v in inflate(box, ratio, W, H)]
    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


class Track:
    """라벨 1개의 시간축 상태. 게이트 통과분만 이력에 반영한다."""
    MAX_MISS = 25          # 이 프레임 수 이상 실패하면 트랙 리셋(신규 고신뢰 검출 대기)

    def __init__(self, label):
        self.label = label
        self.sizes = []            # 최근 승인된 size 목록 (최대 30)
        self.center = None         # 마지막 승인 3D 중심
        self.box2d = None          # 마지막 승인 2D 박스
        self.depth_med = None      # 마지막 승인 전경 depth 중앙값
        self.ema_c = None
        self.ema_s = None
        self.miss = 0
        self.n_acc = self.n_rej = 0

    def med_size(self):
        return np.median(np.array(self.sizes), axis=0) if self.sizes else None

    def max_size(self):
        for k, v in MAX_SIZE_BY_LABEL.items():
            if k in self.label:
                return v
        return MAX_SIZE_DEFAULT

    def gate(self, b, mask, box2d, propagated, dmed):
        """4중 게이트. 반환 (통과여부, 사유)"""
        if np.max(b.size) > self.max_size():        # 절대 상한 — 이력 없이도 즉시 기각
            return False, f"absmax>{self.max_size():.2f}m"
        x1, y1, x2, y2 = box2d
        barea = max((x2 - x1) * (y2 - y1), 1.0)
        fill = mask.sum() / barea
        if not (0.10 <= fill <= 0.98):
            return False, f"fill={fill:.2f}"
        ms = self.med_size()
        if ms is not None and len(self.sizes) >= 3:
            if np.any(np.array(b.size) > 2.5 * ms + 0.05):
                return False, "size>2.5x"
            if np.prod(b.size) > 4.0 * np.prod(ms) + 1e-6:
                return False, "vol>4x"
        if self.center is not None and self.miss < 5:
            if np.linalg.norm(np.array(b.center) - self.center) > 0.25:
                return False, "jump>0.25m"
        if propagated and self.depth_med is not None and dmed is not None:
            if not (0.6 * self.depth_med <= dmed <= 1.6 * self.depth_med):
                return False, "depth_shift"
        return True, ""

    def accept(self, b, box2d, dmed):
        self.sizes.append(np.array(b.size)); self.sizes = self.sizes[-30:]
        self.center = np.array(b.center)
        self.box2d = list(box2d)
        if dmed is not None:
            self.depth_med = dmed
        a = 0.45
        self.ema_c = self.center if self.ema_c is None else (1 - a) * self.ema_c + a * self.center
        self.ema_s = np.array(b.size) if self.ema_s is None else (1 - a) * self.ema_s + a * np.array(b.size)
        self.miss = 0; self.n_acc += 1

    def reject(self):
        self.miss += 1; self.n_rej += 1
        if self.miss > self.MAX_MISS:      # 오래 놓치면 과거 위치를 믿지 않음
            self.box2d = None; self.center = None; self.depth_med = None


class EpisodeTracker:
    """에피소드 1개 처리기. targets = {짧은키: 표시라벨} (프롬프트 토큰 매칭용)"""

    def __init__(self, targets, det, seg, prompt, low_thr=0.20):
        self.targets = targets
        self.det, self.seg, self.prompt = det, seg, prompt
        self.low_thr = low_thr
        self.tracks = {k: Track(v) for k, v in targets.items()}

    def _match(self, key, boxes, phrases, scores):
        """검출 결과에서 key 토큰이 포함된 후보 중 최적 1개 (score + 이전 IoU 가중)"""
        tr = self.tracks[key]
        best, bs = None, -1
        for b, p, s in zip(boxes, phrases, scores):
            if key not in str(p):
                continue
            bonus = 0.3 * iou2d(b, tr.box2d) if tr.box2d is not None else 0.0
            if s + bonus > bs:
                bs, best = s + bonus, list(map(float, b))
        return best

    def step(self, img, depth, K, dscale, align_dx=0):
        """1프레임 처리. 반환: [{key,label,box2d,center,size,src,accepted,reason}...]"""
        H, W = img.shape[:2]
        boxes, phrases, scores = self.det(img, self.prompt)
        results = []
        pend = {}                                    # key -> (box2d, src)
        missing = []
        for key in self.targets:
            b = self._match(key, boxes, phrases, scores)
            if b is not None:
                pend[key] = (b, "det")
            else:
                missing.append(key)
        # 저임계 재검출 (놓친 라벨이 있고 이전 위치를 알 때만 1회)
        if missing and any(self.tracks[k].box2d is not None for k in missing):
            old = self.det.box_thr
            self.det.box_thr = self.low_thr
            b2, p2, s2 = self.det(img, self.prompt)
            self.det.box_thr = old
            for key in list(missing):
                tr = self.tracks[key]
                if tr.box2d is None:
                    continue
                cand = None; bi = 0.1
                for b, p in zip(b2, p2):
                    if key in str(p) and iou2d(b, tr.box2d) > bi:
                        bi = iou2d(b, tr.box2d); cand = list(map(float, b))
                if cand is not None:
                    pend[key] = (cand, "redet"); missing.remove(key)
        # SAM 전파: 이전 박스를 12% 키워 프롬프트로, 마스크 -> tight box 갱신
        for key in missing:
            tr = self.tracks[key]
            if tr.box2d is not None:
                pend[key] = (inflate(tr.box2d, 1.12, W, H), "prop")

        if not pend:
            return results
        keys = list(pend.keys())
        bxs = np.array([pend[k][0] for k in keys], dtype=np.float32)
        masks = self.seg(img, bxs)

        for k, m in zip(keys, masks):
            box2d, src = pend[k]
            tr = self.tracks[k]
            mk = clip_mask_to_box(m.astype(bool), box2d, 1.15)
            mk = largest_component(mk)               # 케이블·팔·그림자 등 딸린 덩어리 제거
            if src == "prop" and mk.sum() > 50:      # 전파면 mask 기준으로 박스 재산출
                ys, xs = np.where(mk)
                box2d = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            mk = cv2.erode(mk.astype(np.uint8), np.ones((5, 5), np.uint8), 1).astype(bool)
            mk = align_mask_to_depth(mk, dx=align_dx)
            pts = filter_points(backproject(depth, K, mk, depth_scale=dscale, valid_range=(0.1, 5.0)),
                                percentile=(10, 90), foreground_mad=1.5)
            pts = cluster_depth(pts, gap=0.06)       # 깊이로 분리된 덩어리 중 주 덩어리만
            b = fit_box3d(pts)
            if b is None:
                tr.reject()
                results.append(dict(key=k, label=tr.label, src=src, accepted=False, reason="no_points"))
                continue
            dmed = float(np.median(depth[mk])) * dscale if mk.sum() > 0 else None
            ok, why = tr.gate(b, mk, box2d, src == "prop", dmed)
            if ok:
                tr.accept(b, box2d, dmed)
                results.append(dict(key=k, label=tr.label, src=src, accepted=True, reason="",
                                    box2d=box2d, center=list(map(float, tr.ema_c)),
                                    size=list(map(float, tr.ema_s)), raw_size=list(map(float, b.size)),
                                    n_points=int(b.n_points), _b=b))
            else:
                tr.reject()
                results.append(dict(key=k, label=tr.label, src=src, accepted=False, reason=why))
        return results

    def draw(self, vis, results, K):
        from geometry import Box3D
        pal = {"det": (0, 220, 120), "redet": (0, 200, 255), "prop": (255, 160, 40)}
        for r in results:
            if not r.get("accepted"):
                continue
            raw = r.pop("_b")
            # 렌더는 EMA 스무딩된 중심·크기로 박스 재구성
            c, s = np.array(r["center"]), np.array(r["size"])
            b = Box3D(center=c, size=s, min_xyz=c - s / 2, max_xyz=c + s / 2,
                      n_points=raw.n_points)
            draw_box3d(vis, b, K, color=pal[r["src"]], thickness=2,
                       label=f"{r['label']} {s[0]*100:.0f}x{s[1]*100:.0f}x{s[2]*100:.0f}cm")
        return vis

    def stats(self, n_frames):
        out = {}
        for k, tr in self.tracks.items():
            ms = tr.med_size()
            out[tr.label] = dict(
                accepted=tr.n_acc, rejected=tr.n_rej,
                coverage=round(tr.n_acc / max(n_frames, 1), 3),
                size_median=None if ms is None else [round(float(x), 4) for x in ms])
        return out
