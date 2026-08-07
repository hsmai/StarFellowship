"""보고서용 모듈 그림 렌더링 — SAM 2.1 마스크 / Back-projection 점군.

frames.json에 저장된 box2d를 SAM에 그대로 넣어 **파이프라인과 동일한 마스크**를 재현한다.
(GroundingDINO 재실행 불필요 → GPU 몇 분)

패널 3종을 유닛/프레임마다 저장한다:
  <key>_mask.png : RGB + 마스크 반투명 오버레이 + 윤곽선   (SAM 그림용)
  <key>_bev.png  : 마스크 픽셀을 3D로 역투영한 점군의 top-view (Back-projection 그림용)
  <key>_meta.json: 크기·점수 등 캡션에 쓸 수치

라벨은 전부 영문/숫자 (cv2가 한글을 못 그림). 한글 캡션은 로컬 합성 단계에서 붙인다.
"""
import os, sys, json
import numpy as np
import cv2

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import Intrinsics
import run_review as RV

R6 = f"{ROOT}/review/r6"
OUT = f"{ROOT}/probe5"
DEV = "cuda:0"

# (키, 유닛경로, 라벨, 프레임번호, 설명)
CASES = [
    # SAM 워스트 — 같은 물체/같은 에피소드인데 마스크가 붕괴한 프레임 (전부 실검출 프레임)
    ("sam_towel",      "he/deformable_ep4838",                    "towel",       62, "best"),
    ("sam_towel_bad",  "he/deformable_ep4838",                    "towel",       92, "worst cand"),
    ("sam_towel_bad2", "he/deformable_ep4838",                    "towel",       89, "worst cand"),
    ("sam_doll_bad",   "brainco/PickDoll_ep5/cam_right_wrist",    "doll",       115, "worst cand"),
    ("sam_duster_bad", "he/Tool_use_ep8198",                      "duster",      32, "worst cand"),
]


def log(m):
    print(m, flush=True)


def get_seq(unit):
    """유닛 경로에서 (프레임 시퀀스, 임시디렉토리) 복원. run_review와 동일 파라미터."""
    parts = unit.rstrip("/").split("/")
    if unit.startswith("brainco/"):
        task, ep = parts[1].split("_ep")
        files, tmp = RV.frames_brainco(task, int(ep), parts[2])
        return [(cv2.imread(f), None) for f in files], tmp, False
    ep = int(parts[1].split("_ep")[1])
    return RV.frames_he(ep), None, True


def rec_at(unit, label, fi):
    d = json.load(open(f"{R6}/{unit}/frames.json"))
    for x in d[fi]["r"]:
        if x["label"] == label:
            return x
    return None


def draw_mask_panel(img, mask, box, label, score):
    """RGB 위에 마스크를 반투명으로 얹고 윤곽선을 그린다."""
    vis = img.copy()
    ov = vis.copy()
    ov[mask] = (0, 235, 255)                       # 노랑(BGR) — 배경과 구분
    vis = cv2.addWeighted(ov, 0.45, vis, 0.55, 0)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 140, 255), 2)
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
    px = int(mask.sum())
    ar = px / (img.shape[0] * img.shape[1]) * 100
    txt = f"{label}  mask {px:,}px ({ar:.1f}% of frame)"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return vis, px, ar


def backproject(mask, depth, K, dscale, adx):
    """마스크 픽셀 → 카메라 좌표계 3D 점군."""
    H, W = depth.shape
    m = mask
    if adx:                                        # HE의 RGB-depth 정렬 보정
        m = np.roll(mask, adx, axis=1)
    ys, xs = np.nonzero(m[:H, :W])
    if len(xs) == 0:
        return np.empty((0, 3))
    z = depth[ys, xs].astype(np.float64) * dscale
    ok = np.isfinite(z) & (z > 0.05) & (z < 8.0)
    xs, ys, z = xs[ok], ys[ok], z[ok]
    X = (xs - K.cx) / K.fx * z
    Y = (ys - K.cy) / K.fy * z
    return np.stack([X, Y, z], 1)


def draw_bev_panel(pts, size, wpx=560, hpx=430):
    """점군을 위에서 본 그림(X-Z 평면). 깊이 방향 두께가 한눈에 보이게 한다."""
    pan = np.full((hpx, wpx, 3), 22, np.uint8)
    if len(pts) < 5:
        return pan, None
    # 이상치 제거 후 표시 범위 결정 (파이프라인 필터와 동일한 백분위 기준)
    lo, hi = np.percentile(pts, [1, 99], axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    q = pts[keep] if keep.sum() > 5 else pts
    cx, cz = q[:, 0].mean(), q[:, 2].mean()
    span = max(np.ptp(q[:, 0]), np.ptp(q[:, 2]), 0.04) * 1.9
    s = min(wpx, hpx) / span
    u = ((q[:, 0] - cx) * s + wpx / 2).astype(int)
    v = ((q[:, 2] - cz) * s + hpx / 2).astype(int)
    ok = (u >= 0) & (u < wpx) & (v >= 0) & (v < hpx)
    for a, b in zip(u[ok], v[ok]):
        cv2.circle(pan, (a, b), 1, (120, 230, 120), -1)
    # 눈금 (10cm)
    g = int(0.10 * s)
    if g > 12:
        for k in range(-8, 9):
            cv2.line(pan, (int(wpx/2+k*g), 0), (int(wpx/2+k*g), hpx), (48, 48, 48), 1)
            cv2.line(pan, (0, int(hpx/2+k*g)), (wpx, int(hpx/2+k*g)), (48, 48, 48), 1)
    # 점군의 실제 폭/깊이
    ex, ez = np.ptp(q[:, 0]), np.ptp(q[:, 2])
    bw, bh = int(ex * s), int(ez * s)
    cv2.rectangle(pan, (int(wpx/2-bw/2), int(hpx/2-bh/2)),
                  (int(wpx/2+bw/2), int(hpx/2+bh/2)), (90, 200, 255), 2)
    cv2.putText(pan, "TOP VIEW (X-Z)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(pan, f"width {ex*100:.1f}cm", (10, hpx - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (150, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(pan, f"depth {ez*100:.1f}cm  <- camera axis", (10, hpx - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 200, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(pan, (wpx - 40, hpx - 70), (wpx - 40, hpx - 25), (200, 200, 200), 2, tipLength=0.3)
    cv2.putText(pan, "cam", (wpx - 62, hpx - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    return pan, (float(ex), float(ez))


def main():
    import shutil
    from models_wrap import Segmenter, DepthEstimator
    os.makedirs(OUT, exist_ok=True)
    seg = Segmenter(DEV)
    dep = DepthEstimator(DEV)
    log("모델 로드 완료")

    by_unit = {}
    for key, unit, label, fi, desc in CASES:
        by_unit.setdefault(unit, []).append((key, label, fi, desc))

    meta = {}
    for unit, items in by_unit.items():
        log(f"--- {unit} ({len(items)}건)")
        seq, tmp, is_he = get_seq(unit)
        for key, label, fi, desc in items:
            if fi >= len(seq):
                log(f"  !! {key}: 프레임 {fi} 없음 (총 {len(seq)})"); continue
            r = rec_at(unit, label, fi)
            if r is None or "box2d" not in r:
                log(f"  !! {key}: 기록 없음"); continue
            img = seq[fi][0]
            H, W = img.shape[:2]
            box = np.array([r["box2d"]], dtype=np.float32)
            masks = seg(img, box)
            if len(masks) == 0:
                log(f"  !! {key}: 마스크 없음"); continue
            mask = masks[0]
            vis, px, ar = draw_mask_panel(img, mask, r["box2d"], label, r.get("det_score", 0))
            cv2.imwrite(f"{OUT}/{key}_mask.png", vis)

            # depth: HE는 실측, Brainco는 UniDepthV2
            if is_he:
                d = seq[fi][1]
                if d.ndim == 1:
                    d = d.reshape(d.size // W, W)
                if d.shape != (H, W):
                    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
                K = Intrinsics.from_fov(W, H, 70.0); dscale, adx = 1e-3, -20
            else:
                d, Kp = dep(img)
                K = (Intrinsics(float(Kp[0, 0]), float(Kp[1, 1]), float(Kp[0, 2]), float(Kp[1, 2]))
                     if Kp is not None else Intrinsics.from_fov(W, H, 70.0))
                dscale, adx = 1.0, 0
            pts = backproject(mask, d, K, dscale, adx)
            bev, ext = draw_bev_panel(pts, r.get("size"))
            cv2.imwrite(f"{OUT}/{key}_bev.png", bev)
            meta[key] = dict(unit=unit, label=label, frame=fi, desc=desc,
                             mask_px=px, area_pct=round(ar, 2), n_points=int(len(pts)),
                             size_cm=[round(v * 100, 1) for v in r.get("size", [])],
                             raw_cm=[round(v * 100, 1) for v in r.get("raw_size", [])],
                             bev_wd_cm=[round(ext[0] * 100, 1), round(ext[1] * 100, 1)] if ext else None,
                             det_score=round(float(r.get("det_score", 0)), 3),
                             dmed=round(float(r.get("dmed", 0)), 3))
            log(f"  {key}: mask {px}px ({ar:.1f}%) pts={len(pts)} "
                f"size={meta[key]['size_cm']} bev(w,d)={meta[key]['bev_wd_cm']}")
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    json.dump(meta, open(f"{OUT}/meta.json", "w"), ensure_ascii=False, indent=1)
    log(f"완료 → {OUT}")


if __name__ == "__main__":
    main()
