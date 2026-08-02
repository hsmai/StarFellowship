"""Task 3 메인 실행기 — STEP 2/3/4 전 과정.
체크포인트 방식: 각 단계 결과를 즉시 저장하고 .done 마커를 남겨,
중간에 끊겨도 재실행 시 완료 단계를 건너뛴다.

사용법:
  python run_pipeline.py smoke          # 모델 로딩 + VRAM 측정
  python run_pipeline.py frame          # STEP2: 프레임 1장 관통
  python run_pipeline.py brainco        # STEP3: Brainco 에피소드
  python run_pipeline.py he             # STEP4: HE + depth 정확도 검증
  python run_pipeline.py all            # 전부 순차
"""
import os, sys, json, time, traceback
import numpy as np
import cv2

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import (Intrinsics, backproject, filter_points, fit_box3d,
                      draw_box3d, align_mask_to_depth, project_points)

MK, RES, LOG = f"{ROOT}/markers", f"{ROOT}/results", f"{ROOT}/logs"
for d in (MK, RES, LOG):
    os.makedirs(d, exist_ok=True)

DATA = "/data2/humanoid_dataset_isangmin"
BRAINCO = f"{DATA}/G1_Brainco_GraspOreo_Dataset"
HE = f"{DATA}/humanoid-everyday"

# 대상 (STEP 0.5에서 선정)
BRAINCO_EP, BRAINCO_PROMPT = 5, "oreo snack package . plate ."
HE_EP, HE_PROMPT = 3800, "pink toy . orange bowl ."


def log(msg, f="run.log"):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(f"{LOG}/{f}", "a") as fp:
        fp.write(line + "\n")


def done(tag): return os.path.exists(f"{MK}/{tag}.done")
def mark(tag, info=None):
    with open(f"{MK}/{tag}.done", "w") as f:
        f.write(json.dumps(info or {}, ensure_ascii=False, indent=1, default=str))
    log(f"  -> {tag} 완료")


# ---------------------------------------------------------------- 데이터 로더
def load_brainco_frame(ep=BRAINCO_EP, t_rel=0.5):
    """Brainco: 통합 mp4에서 에피소드 구간을 찾아 프레임 추출"""
    import glob, pandas as pd, subprocess
    epm = pd.read_parquet(sorted(glob.glob(f"{BRAINCO}/meta/episodes/chunk-000/*.parquet"))[0])
    r = epm[epm.episode_index == ep].iloc[0]
    key = "videos/observation.images.cam_left_high"
    fi = int(r[f"{key}/file_index"])
    t0, t1 = float(r[f"{key}/from_timestamp"]), float(r[f"{key}/to_timestamp"])
    mp4 = f"{BRAINCO}/videos/observation.images.cam_left_high/chunk-000/file-{fi:03d}.mp4"
    ts = t0 + (t1 - t0) * t_rel
    tmp = "/tmp/_bc.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(ts), "-i", mp4,
                    "-frames:v", "1", tmp], check=True)
    return cv2.imread(tmp), dict(ep=ep, t0=t0, t1=t1, mp4=mp4, dur=t1 - t0)


def load_he_frame(ep=HE_EP, idx=None):
    """HE: 에피소드별 파일. RGB + 실측 depth 동시 반환"""
    import pandas as pd
    ch = ep // 1000
    df = pd.read_parquet(f"{HE}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")
    i = idx if idx is not None else len(df) // 2
    depth = np.stack(df["observation.depth.egocentric"].iloc[i]).astype(np.float32)
    mp4 = f"{HE}/videos/chunk-{ch:03d}/egocentric/episode_{ep:06d}.mp4"
    cap = cv2.VideoCapture(mp4); cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read(); cap.release()
    return img, depth, dict(ep=ep, frame=i, n=len(df), mp4=mp4)


# ---------------------------------------------------------------- STEP 1.5 스모크
def smoke():
    if done("2_smoke"): log("[smoke] 이미 완료 — 건너뜀"); return True
    import torch
    from models_wrap import Detector, Segmenter, DepthEstimator, IntrinsicsEstimator
    info = {"gpu": torch.cuda.get_device_name(0), "n_gpu": torch.cuda.device_count()}
    log(f"[smoke] {info['gpu']} x{info['n_gpu']}")
    img, _ = load_brainco_frame()
    log(f"[smoke] 테스트 프레임 {img.shape}")

    for name, fn in [
        ("GroundingDINO", lambda: Detector("cuda:0")),
        ("SAM2.1", lambda: Segmenter("cuda:0")),
        ("UniDepthV2", lambda: DepthEstimator("cuda:1")),
        ("WildCamera", lambda: IntrinsicsEstimator("cuda:1")),
    ]:
        try:
            torch.cuda.empty_cache(); t = time.time()
            m = fn()
            dev = 0 if "cuda:0" in str(getattr(m, "device", "cuda:0")) else 1
            v = torch.cuda.memory_allocated(dev) / 1e9
            info[name] = {"ok": True, "load_s": round(time.time() - t, 1), "vram_GB": round(v, 2)}
            log(f"[smoke] {name:15s} 로드 OK  {info[name]['load_s']}s  VRAM {info[name]['vram_GB']}GB")
            del m
        except Exception as e:
            info[name] = {"ok": False, "err": str(e)[:200]}
            log(f"[smoke] {name:15s} 실패: {str(e)[:150]}")
    mark("2_smoke", info)
    return any(v.get("ok") for k, v in info.items() if isinstance(v, dict))


# ---------------------------------------------------------------- 공통 처리
def process_frame(img, prompt, det, seg, depth_map=None, dep=None, wc=None,
                  align_dx=0, tag="frame"):
    """1~5단계를 한 프레임에 적용. depth_map이 주어지면 3단계(추정)를 건너뛴다."""
    out = {"tag": tag}
    boxes, phrases, logits = det(img, prompt)
    out["n_det"] = len(boxes); out["phrases"] = list(phrases)
    log(f"  [1] 검출 {len(boxes)}개: {list(phrases)}")
    if len(boxes) == 0:
        return out, None, None, None

    masks = seg(img, boxes)
    out["n_mask"] = len(masks)
    log(f"  [2] 마스크 {len(masks)}개 (픽셀 {[int(m.sum()) for m in masks]})")

    K_est = None
    if depth_map is None:
        depth_map, K_pred = dep(img)
        out["depth_src"] = "UniDepthV2"
        out["depth_range"] = [float(np.nanmin(depth_map)), float(np.nanmax(depth_map))]
        K_est = K_pred
        depth_scale = 1.0                       # UniDepth는 이미 meter
        log(f"  [3] depth 추정 {depth_map.shape} {out['depth_range'][0]:.2f}~{out['depth_range'][1]:.2f}m")
    else:
        out["depth_src"] = "GT(실측)"; depth_scale = 1e-3
        log(f"  [3] 실측 depth 사용 (mm)")

    if K_est is None and wc is not None:
        K_est = wc(img)
    H, W = img.shape[:2]
    if K_est is not None:
        K = Intrinsics(float(K_est[0, 0]), float(K_est[1, 1]), float(K_est[0, 2]), float(K_est[1, 2]))
        out["K_src"] = "model"
    else:
        K = Intrinsics.from_fov(W, H, 70.0); out["K_src"] = "FOV70가정"
    out["K"] = [K.fx, K.fy, K.cx, K.cy]
    log(f"  [3] intrinsics({out['K_src']}) fx={K.fx:.1f} fy={K.fy:.1f}")

    vis = img.copy(); boxes3d = []
    palette = [(0, 220, 120), (0, 165, 255), (255, 120, 60), (200, 80, 220)]
    for i, (m, ph) in enumerate(zip(masks, phrases)):
        mk = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8), 1).astype(bool)
        mk = align_mask_to_depth(mk, dx=align_dx)
        pts = filter_points(backproject(depth_map, K, mk, depth_scale=depth_scale,
                                        valid_range=(0.1, 5.0)),
                            percentile=(10, 90), foreground_mad=2.0)
        b = fit_box3d(pts)
        if b is None:
            log(f"  [4-5] {ph}: 점군 부족"); continue
        c = palette[i % len(palette)]
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cont, -1, c, 1)
        draw_box3d(vis, b, K, color=c, thickness=2,
                   label=f"{ph} {b.size[0]*100:.0f}x{b.size[1]*100:.0f}x{b.size[2]*100:.0f}cm")
        boxes3d.append({"label": ph, "center": b.center.tolist(), "size": b.size.tolist(),
                        "n_points": int(b.n_points)})
        log(f"  [4-5] {ph}: {b.summary()}")
    out["boxes3d"] = boxes3d
    return out, vis, depth_map, K


# ---------------------------------------------------------------- STEP 2
def step_frame():
    if done("3_frame"): log("[STEP2] 이미 완료 — 건너뜀"); return
    from models_wrap import Detector, Segmenter, DepthEstimator, IntrinsicsEstimator
    log("[STEP2] 프레임 1장 관통 (Brainco + HE)")
    det, seg = Detector("cuda:0"), Segmenter("cuda:0")
    dep, wc = DepthEstimator("cuda:1"), IntrinsicsEstimator("cuda:1")
    res = {}
    img, meta = load_brainco_frame()
    o, vis, _, _ = process_frame(img, BRAINCO_PROMPT, det, seg, None, dep, wc, tag="brainco")
    if vis is not None: cv2.imwrite(f"{RES}/step2_brainco_frame.png", vis)
    res["brainco"] = o
    img, depth, meta = load_he_frame()
    o, vis, _, _ = process_frame(img, HE_PROMPT, det, seg, depth, dep, wc, align_dx=-20, tag="he_gt")
    if vis is not None: cv2.imwrite(f"{RES}/step2_he_frame.png", vis)
    res["he"] = o
    mark("3_frame", res)


# ---------------------------------------------------------------- STEP 3
def step_brainco(max_frames=None, stride=2):
    if done("4_brainco"): log("[STEP3] 이미 완료 — 건너뜀"); return
    from models_wrap import Detector, Segmenter, DepthEstimator, IntrinsicsEstimator
    import subprocess, glob, pandas as pd
    log("[STEP3] Brainco 에피소드 전체 실행")
    det, seg = Detector("cuda:0"), Segmenter("cuda:0")
    dep, wc = DepthEstimator("cuda:1"), IntrinsicsEstimator("cuda:1")

    _, meta = load_brainco_frame()
    t0, t1, mp4 = meta["t0"], meta["t1"], meta["mp4"]
    seq = "/tmp/bc_seq"; os.makedirs(seq, exist_ok=True)
    os.system(f"rm -f {seq}/*.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0), "-t", str(t1 - t0),
                    "-i", mp4, f"{seq}/f%05d.png"], check=True)
    files = sorted(glob.glob(f"{seq}/f*.png"))[::stride]
    if max_frames: files = files[:max_frames]
    log(f"  프레임 {len(files)}장 (stride={stride})")

    first = cv2.imread(files[0]); H, W = first.shape[:2]
    vw = cv2.VideoWriter(f"{RES}/step3_brainco_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30 // stride, (W, H))
    recs = []
    for i, f in enumerate(files):
        img = cv2.imread(f)
        try:
            o, vis, _, _ = process_frame(img, BRAINCO_PROMPT, det, seg, None, dep, wc, tag=f"bc{i}")
            vw.write(vis if vis is not None else img)
            recs.append(o)
        except Exception as e:
            log(f"  프레임 {i} 실패: {str(e)[:100]}"); vw.write(img)
        if i % 20 == 0:
            log(f"  진행 {i}/{len(files)}")
            json.dump(recs, open(f"{RES}/step3_brainco_partial.json", "w"), ensure_ascii=False, default=str)
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {RES}/step3_brainco_raw.mp4 -c:v libx264 -crf 23 "
              f"-pix_fmt yuv420p {RES}/step3_brainco.mp4 && rm -f {RES}/step3_brainco_raw.mp4")
    json.dump(recs, open(f"{RES}/step3_brainco.json", "w"), ensure_ascii=False, default=str)
    mark("4_brainco", {"n_frames": len(files), "n_ok": sum(1 for r in recs if r.get("boxes3d"))})


# ---------------------------------------------------------------- STEP 4
def step_he():
    if done("5_he"): log("[STEP4] 이미 완료 — 건너뜀"); return
    from models_wrap import Detector, Segmenter, DepthEstimator, IntrinsicsEstimator
    import pandas as pd
    log("[STEP4] HE — 추정 depth vs 실측 depth 검증")
    det, seg = Detector("cuda:0"), Segmenter("cuda:0")
    dep, wc = DepthEstimator("cuda:1"), IntrinsicsEstimator("cuda:1")

    ch = HE_EP // 1000
    df = pd.read_parquet(f"{HE}/data/chunk-{ch:03d}/episode_{HE_EP:06d}.parquet")
    idxs = np.linspace(10, len(df) - 10, 12).astype(int)
    cmp_rows, box_pairs = [], []
    for k, i in enumerate(idxs):
        img, gt, meta = load_he_frame(HE_EP, int(i))
        gt_m = gt.astype(np.float32) * 1e-3
        pred, _ = dep(img)
        valid = (gt_m > 0.1) & (gt_m < 5) & np.isfinite(pred)
        if valid.sum() < 1000: continue
        a, b = gt_m[valid], pred[valid]
        scale = float(np.median(a / np.clip(b, 1e-6, None)))
        cmp_rows.append(dict(frame=int(i), n=int(valid.sum()),
                             mae=float(np.abs(a - b).mean()),
                             rel=float((np.abs(a - b) / a).mean()),
                             scale=scale,
                             mae_scaled=float(np.abs(a - b * scale).mean())))
        # 같은 프레임에서 GT/추정 각각으로 3D 박스
        o_gt, vis_gt, _, _ = process_frame(img, HE_PROMPT, det, seg, gt, dep, wc, align_dx=-20, tag=f"he_gt{k}")
        o_pr, vis_pr, _, _ = process_frame(img, HE_PROMPT, det, seg, None, dep, wc, tag=f"he_pred{k}")
        box_pairs.append({"frame": int(i), "gt": o_gt.get("boxes3d"), "pred": o_pr.get("boxes3d")})
        if k == len(idxs) // 2 and vis_gt is not None and vis_pr is not None:
            cv2.imwrite(f"{RES}/step4_he_gt.png", vis_gt)
            cv2.imwrite(f"{RES}/step4_he_pred.png", vis_pr)
            cv2.imwrite(f"{RES}/step4_he_compare.png", np.hstack([vis_gt, vis_pr]))
        log(f"  프레임 {i}: MAE {cmp_rows[-1]['mae']:.3f}m, 상대오차 {cmp_rows[-1]['rel']*100:.1f}%, "
            f"스케일 {scale:.3f}, 보정후 MAE {cmp_rows[-1]['mae_scaled']:.3f}m")
        json.dump({"depth": cmp_rows, "boxes": box_pairs},
                  open(f"{RES}/step4_he_partial.json", "w"), ensure_ascii=False, default=str)
    summ = {"depth": cmp_rows, "boxes": box_pairs,
            "mean_mae": float(np.mean([r["mae"] for r in cmp_rows])) if cmp_rows else None,
            "mean_rel": float(np.mean([r["rel"] for r in cmp_rows])) if cmp_rows else None,
            "mean_scale": float(np.mean([r["scale"] for r in cmp_rows])) if cmp_rows else None}
    json.dump(summ, open(f"{RES}/step4_he.json", "w"), ensure_ascii=False, default=str)
    log(f"[STEP4] 평균 MAE {summ['mean_mae']}m, 상대오차 {summ['mean_rel']}, 스케일 {summ['mean_scale']}")
    mark("5_he", {k: summ[k] for k in ("mean_mae", "mean_rel", "mean_scale")})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    try:
        if cmd in ("smoke", "all"): smoke()
        if cmd in ("frame", "all"): step_frame()
        if cmd in ("brainco", "all"): step_brainco()
        if cmd in ("he", "all"): step_he()
        log(f"=== {cmd} 종료 ===")
    except Exception:
        log("!! 예외:\n" + traceback.format_exc())
        sys.exit(1)
