"""Task 3 심화 작업 A~D — GPU 1장에서 직렬 실행.
A. SAM 2.1 비디오 추적으로 검출률 개선
B. AgiBot World 2026 정답(GT) 대조 검증  ← 파이프라인 정확도를 처음으로 수치화
C. Brainco 4개 카메라 전부 활용
D. Brainco 8개 태스크로 확장

각 작업은 markers/<tag>.done 으로 중복 실행이 방지되어 재시작에 안전하다.
"""
import os, sys, json, glob, time, traceback, subprocess
import numpy as np
import cv2

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import (Intrinsics, backproject, filter_points, fit_box3d,
                      draw_box3d, align_mask_to_depth)

MK, RES, LOGD = f"{ROOT}/markers", f"{ROOT}/results", f"{ROOT}/logs"
for d in (MK, RES, LOGD):
    os.makedirs(d, exist_ok=True)

DATA = "/data2/humanoid_dataset_isangmin"
BRAINCO_ROOT = f"{DATA}/G1_Brainco_%s_Dataset"
AGI = f"{DATA}/agibot-world-2026-sample/ImitationLearning/CommercialSpaces/task_3777/extracted_380090_381352/data"

DEV = "cuda:0"          # CUDA_VISIBLE_DEVICES로 물리 GPU 1장만 노출

BRAINCO_TASKS = {   # 태스크 -> 검출 프롬프트 (tasks.parquet 문장에서 객체 추출)
    "GraspOreo":        "oreo snack package . plate .",
    "GraspRubiksCube":  "rubiks cube . plate .",
    "PickApple":        "apple . plate .",
    "PickCharger":      "charger . plate .",
    "PickDoll":         "doll . plate .",
    "PickDrink":        "red cup . plate .",
    "PickTissues":      "tissue paper . plate .",
    "PickToothpaste":   "toothpaste tube . plate .",
}
CAMS = ["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]


def log(msg, f="extra.log"):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(f"{LOGD}/{f}", "a") as fp:
        fp.write(line + "\n")


def done(tag): return os.path.exists(f"{MK}/{tag}.done")
def mark(tag, info=None):
    json.dump(info or {}, open(f"{MK}/{tag}.done", "w"), ensure_ascii=False, indent=1, default=str)
    log(f"  -> {tag} 완료")


# ------------------------------------------------------------ 공통 유틸
def brainco_episode_clip(task="GraspOreo", ep=5, cam="cam_left_high", outdir=None, stride=2):
    """Brainco 통합 mp4에서 에피소드 구간을 프레임 시퀀스로 추출"""
    import pandas as pd
    root = BRAINCO_ROOT % task
    epm = pd.read_parquet(sorted(glob.glob(f"{root}/meta/episodes/chunk-000/*.parquet"))[0])
    rows = epm[epm.episode_index == ep]
    if len(rows) == 0:
        return [], None
    r = rows.iloc[0]
    key = f"videos/observation.images.{cam}"
    fi = int(r[f"{key}/file_index"])
    t0, t1 = float(r[f"{key}/from_timestamp"]), float(r[f"{key}/to_timestamp"])
    mp4 = f"{root}/videos/observation.images.{cam}/chunk-000/file-{fi:03d}.mp4"
    outdir = outdir or f"/tmp/seq_{task}_{cam}_{ep}"
    os.makedirs(outdir, exist_ok=True); os.system(f"rm -f {outdir}/*.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0), "-t", str(t1 - t0),
                    "-i", mp4, f"{outdir}/f%05d.png"], check=True)
    files = sorted(glob.glob(f"{outdir}/f*.png"))[::stride]
    return files, dict(task=task, ep=ep, cam=cam, dur=t1 - t0, n=len(files))


def boxes_from_frame(img, prompt, det, seg, dep, align_dx=0, depth_gt=None):
    """1~5단계 1프레임 처리 → (박스목록, 시각화, 검출수)"""
    boxes, phrases, scores = det(img, prompt)
    if len(boxes) == 0:
        return [], img.copy(), 0, None
    masks = seg(img, boxes)
    if depth_gt is None:
        depth, K_pred = dep(img); dscale = 1.0
    else:
        depth, K_pred, dscale = depth_gt, None, 1e-3
    H, W = img.shape[:2]
    K = (Intrinsics(float(K_pred[0, 0]), float(K_pred[1, 1]), float(K_pred[0, 2]), float(K_pred[1, 2]))
         if K_pred is not None else Intrinsics.from_fov(W, H, 70.0))
    vis = img.copy(); out = []
    pal = [(0, 220, 120), (0, 165, 255), (255, 120, 60), (200, 80, 220)]
    for i, (m, ph) in enumerate(zip(masks, phrases)):
        mk = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8), 1).astype(bool)
        mk = align_mask_to_depth(mk, dx=align_dx)
        pts = filter_points(backproject(depth, K, mk, depth_scale=dscale, valid_range=(0.1, 5.0)),
                            percentile=(10, 90), foreground_mad=2.0)
        b = fit_box3d(pts)
        if b is None: continue
        c = pal[i % len(pal)]
        draw_box3d(vis, b, K, color=c, thickness=2,
                   label=f"{ph} {b.size[0]*100:.0f}x{b.size[1]*100:.0f}x{b.size[2]*100:.0f}cm")
        out.append(dict(label=str(ph), box2d=[float(x) for x in boxes[i]],
                        center=b.center.tolist(), size=b.size.tolist(), n_points=int(b.n_points)))
    return out, vis, len(boxes), K


# ============================================================ A. 비디오 추적
def task_A():
    if done("A_track"): log("[A] 이미 완료 — 건너뜀"); return
    from models_wrap import Detector, Segmenter, DepthEstimator
    log("[A] SAM 비디오 추적으로 검출률 개선")
    det, seg, dep = Detector(DEV), Segmenter(DEV), DepthEstimator(DEV)
    files, meta = brainco_episode_clip("GraspOreo", 5, "cam_left_high", stride=2)
    log(f"  프레임 {len(files)}장")

    prompt = BRAINCO_TASKS["GraspOreo"]
    recs, prev_boxes = [], None
    H, W = cv2.imread(files[0]).shape[:2]
    vw = cv2.VideoWriter(f"{RES}/A_track_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    n_det_only = n_with_track = 0
    for i, f in enumerate(files):
        img = cv2.imread(f)
        boxes, phrases, scores = det(img, prompt)
        used_track = False
        if len(boxes) < 2 and prev_boxes is not None:
            # 검출이 빠진 라벨을 직전 프레임 박스로 보완 (박스 전파 방식 추적)
            have = {str(p) for p in phrases}
            add_b, add_p = [], []
            for pb, pp in prev_boxes:
                if pp not in have:
                    add_b.append(pb); add_p.append(pp)
            if add_b:
                boxes = np.vstack([boxes, np.array(add_b)]) if len(boxes) else np.array(add_b)
                phrases = list(phrases) + add_p
                used_track = True
        if len(boxes) == 0:
            vw.write(img); continue
        n_det_only += 1 if not used_track else 0
        n_with_track += 1
        masks = seg(img, boxes)
        depth, K_pred = dep(img)
        K = (Intrinsics(float(K_pred[0, 0]), float(K_pred[1, 1]), float(K_pred[0, 2]), float(K_pred[1, 2]))
             if K_pred is not None else Intrinsics.from_fov(W, H, 70.0))
        vis = img.copy(); frame_boxes = []
        pal = [(0, 220, 120), (0, 165, 255)]
        for j, (m, ph) in enumerate(zip(masks, phrases)):
            mk = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8), 1).astype(bool)
            pts = filter_points(backproject(depth, K, mk, depth_scale=1.0, valid_range=(0.1, 5.0)),
                                percentile=(10, 90), foreground_mad=2.0)
            b = fit_box3d(pts)
            if b is None: continue
            col = pal[j % 2] if not used_track else (60, 200, 255)
            draw_box3d(vis, b, K, color=col, thickness=2,
                       label=f"{ph}{'*' if used_track else ''} {b.size[0]*100:.0f}x{b.size[1]*100:.0f}x{b.size[2]*100:.0f}cm")
            frame_boxes.append(dict(label=str(ph), tracked=used_track,
                                    center=b.center.tolist(), size=b.size.tolist()))
        prev_boxes = [(list(map(float, bb)), str(pp)) for bb, pp in zip(boxes, phrases)]
        recs.append(dict(frame=i, boxes=frame_boxes, tracked=used_track))
        vw.write(vis)
        if i % 50 == 0:
            log(f"  진행 {i}/{len(files)}")
            json.dump(recs, open(f"{RES}/A_track_partial.json", "w"), default=str)
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {RES}/A_track_raw.mp4 -c:v libx264 -crf 23 "
              f"-pix_fmt yuv420p {RES}/A_track.mp4 && rm -f {RES}/A_track_raw.mp4")
    # 라벨별 커버리지
    cov = {}
    for lab in ("plate", "oreo snack package"):
        cov[lab] = sum(1 for r in recs for b in r["boxes"] if b["label"] == lab)
    json.dump(recs, open(f"{RES}/A_track.json", "w"), default=str)
    log(f"[A] 라벨별 프레임 수: {cov} / 전체 {len(files)}")
    mark("A_track", {"n_frames": len(files), "coverage": cov})


# ============================================================ B. AgiBot GT 검증
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def task_B():
    if done("B_agibot"): log("[B] 이미 완료 — 건너뜀"); return
    from models_wrap import Detector, Segmenter, DepthEstimator
    log("[B] AgiBot 정답(GT) 대조 검증 — 검출/depth/intrinsics")
    det, seg, dep = Detector(DEV), Segmenter(DEV), DepthEstimator(DEV)
    info = json.load(open(f"{AGI}/meta/info.json"))
    kf = info["key_frame"]["0"]["dual"]
    gts = [e for e in kf if e.get("frame_type_name") == "2D Bounding Box"]
    K_gt_raw = info["camera_parameters"]["0"].get("intrinsic_head_front_rgb", {})
    log(f"  GT bbox {len(gts)}개, GT intrinsics {K_gt_raw}")

    mp4 = f"{AGI}/videos/chunk-000/observation.images.top_head/episode_000000.mp4"
    dmp4 = f"{AGI}/videos/chunk-000/observation.images.head_depth/episode_000000.mp4"
    rows, dep_rows = [], []
    vis_saved = False
    for gi, g in enumerate(gts[:8]):
        t = (g["start"] + g["end"]) // 2
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t / 30), "-i", mp4,
                        "-frames:v", "1", "/tmp/_agi.png"], check=True)
        img = cv2.imread("/tmp/_agi.png")
        if img is None: continue
        H, W = img.shape[:2]
        bx = g["frame_detail"]["box"]
        gt_xyxy = [bx["x"] * W, bx["y"] * H, (bx["x"] + bx["w"]) * W, (bx["y"] + bx["h"]) * H]

        boxes, phrases, scores = det(img, "flyer . paper . leaflet . person .")
        best = max((iou(gt_xyxy, b) for b in boxes), default=0.0)
        rows.append(dict(gt_idx=gi, frame=t, n_det=len(boxes), best_iou=float(best),
                         phrases=[str(p) for p in phrases]))
        log(f"  GT#{gi} f{t}: 검출 {len(boxes)}개, 최고 IoU {best:.3f}")

        # depth 실측 vs 추정
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t / 30), "-i", dmp4,
                        "-frames:v", "1", "-f", "image2", "-pix_fmt", "gray16be", "/tmp/_agid.png"], check=True)
        dgt = cv2.imread("/tmp/_agid.png", cv2.IMREAD_UNCHANGED)
        if dgt is not None:
            dgt_m = dgt.astype(np.float32) * 1e-3
            pred, K_pred = dep(img)
            if pred.shape != dgt_m.shape:
                pred = cv2.resize(pred, (dgt_m.shape[1], dgt_m.shape[0]))
            v = (dgt_m > 0.1) & (dgt_m < 6) & np.isfinite(pred)
            if v.sum() > 1000:
                a, b2 = dgt_m[v], pred[v]
                dep_rows.append(dict(frame=t, n=int(v.sum()),
                                     mae=float(np.abs(a - b2).mean()),
                                     rel=float((np.abs(a - b2) / a).mean()),
                                     scale=float(np.median(a / np.clip(b2, 1e-6, None))),
                                     K_pred=[float(K_pred[0, 0]), float(K_pred[1, 1])] if K_pred is not None else None))
                log(f"       depth MAE {dep_rows[-1]['mae']:.3f}m rel {dep_rows[-1]['rel']*100:.1f}% "
                    f"| K_pred fx={dep_rows[-1]['K_pred'][0] if dep_rows[-1]['K_pred'] else '-'}")
        if not vis_saved and len(boxes):
            vis = img.copy()
            cv2.rectangle(vis, (int(gt_xyxy[0]), int(gt_xyxy[1])), (int(gt_xyxy[2]), int(gt_xyxy[3])),
                          (0, 255, 255), 3)
            cv2.putText(vis, "GT (human)", (int(gt_xyxy[0]), max(20, int(gt_xyxy[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for b3, p3 in zip(boxes, phrases):
                cv2.rectangle(vis, (int(b3[0]), int(b3[1])), (int(b3[2]), int(b3[3])), (0, 165, 255), 2)
                cv2.putText(vis, str(p3), (int(b3[0]), max(20, int(b3[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.imwrite(f"{RES}/B_agibot_gt_vs_pred.png", vis); vis_saved = True
        json.dump({"det": rows, "depth": dep_rows}, open(f"{RES}/B_agibot_partial.json", "w"), default=str)

    K_gt = [K_gt_raw.get("Fx"), K_gt_raw.get("Fy")] if K_gt_raw else None
    summ = {"det": rows, "depth": dep_rows, "K_gt": K_gt,
            "mean_iou": float(np.mean([r["best_iou"] for r in rows])) if rows else None,
            "mean_mae": float(np.mean([r["mae"] for r in dep_rows])) if dep_rows else None,
            "mean_rel": float(np.mean([r["rel"] for r in dep_rows])) if dep_rows else None,
            "mean_scale": float(np.mean([r["scale"] for r in dep_rows])) if dep_rows else None}
    json.dump(summ, open(f"{RES}/B_agibot.json", "w"), ensure_ascii=False, default=str)
    log(f"[B] 평균 IoU {summ['mean_iou']} | depth MAE {summ['mean_mae']} | GT fx {K_gt}")
    mark("B_agibot", {k: summ[k] for k in ("mean_iou", "mean_mae", "mean_rel", "mean_scale", "K_gt")})


# ============================================================ C. 4카메라
def task_C():
    if done("C_multiview"): log("[C] 이미 완료 — 건너뜀"); return
    from models_wrap import Detector, Segmenter, DepthEstimator
    log("[C] Brainco 4개 카메라 전부 처리")
    det, seg, dep = Detector(DEV), Segmenter(DEV), DepthEstimator(DEV)
    prompt = BRAINCO_TASKS["GraspOreo"]
    per_cam, tiles = {}, {}
    for cam in CAMS:
        files, meta = brainco_episode_clip("GraspOreo", 5, cam, stride=6)
        if not files: log(f"  {cam}: 프레임 없음"); continue
        log(f"  {cam}: {len(files)}장")
        recs = []
        mid_vis = None
        for i, f in enumerate(files):
            img = cv2.imread(f)
            try:
                bs, vis, ndet, K = boxes_from_frame(img, prompt, det, seg, dep)
                recs.append(dict(frame=i, n_det=ndet, boxes=bs))
                if i == len(files) // 2: mid_vis = vis
            except Exception as e:
                log(f"   {cam} f{i} 실패: {str(e)[:80]}")
        per_cam[cam] = recs
        if mid_vis is not None:
            cv2.imwrite(f"{RES}/C_{cam}.png", mid_vis)
            tiles[cam] = cv2.resize(mid_vis, (480, 360))
        json.dump(per_cam, open(f"{RES}/C_multiview_partial.json", "w"), default=str)
    if len(tiles) == 4:
        top = np.hstack([tiles[CAMS[0]], tiles[CAMS[1]]])
        bot = np.hstack([tiles[CAMS[2]], tiles[CAMS[3]]])
        cv2.imwrite(f"{RES}/C_multiview_grid.png", np.vstack([top, bot]))
    stats = {c: {"frames": len(v),
                 "det_rate": float(np.mean([r["n_det"] > 0 for r in v])) if v else 0,
                 "labels": {lab: sum(1 for r in v for b in r["boxes"] if b["label"] == lab)
                            for lab in ("plate", "oreo snack package")}}
             for c, v in per_cam.items()}
    json.dump({"stats": stats, "per_cam": per_cam}, open(f"{RES}/C_multiview.json", "w"), default=str)
    log(f"[C] 카메라별 통계: {json.dumps(stats, ensure_ascii=False)}")
    mark("C_multiview", stats)


# ============================================================ D. 8태스크 확장
def task_D():
    if done("D_tasks"): log("[D] 이미 완료 — 건너뜀"); return
    from models_wrap import Detector, Segmenter, DepthEstimator
    log("[D] Brainco 8개 태스크로 확장")
    det, seg, dep = Detector(DEV), Segmenter(DEV), DepthEstimator(DEV)
    allstats, tiles = {}, []
    for task, prompt in BRAINCO_TASKS.items():
        try:
            files, meta = brainco_episode_clip(task, 0, "cam_left_high", stride=10)
        except Exception as e:
            log(f"  {task}: 클립 실패 {str(e)[:80]}"); continue
        if not files: continue
        files = files[:40]
        log(f"  {task}: {len(files)}장, prompt='{prompt}'")
        recs, mid_vis = [], None
        for i, f in enumerate(files):
            img = cv2.imread(f)
            try:
                bs, vis, ndet, K = boxes_from_frame(img, prompt, det, seg, dep)
                recs.append(dict(frame=i, n_det=ndet, boxes=bs))
                if i == len(files) // 2: mid_vis = vis
            except Exception as e:
                log(f"   {task} f{i} 실패: {str(e)[:70]}")
        # 태스크별 대표 물체 크기
        sizes = {}
        for r in recs:
            for b in r["boxes"]:
                sizes.setdefault(b["label"], []).append(b["size"])
        allstats[task] = {"frames": len(recs),
                          "det_rate": float(np.mean([r["n_det"] > 0 for r in recs])) if recs else 0,
                          "objects": {k: {"n": len(v), "size_median": np.median(np.array(v), 0).tolist()}
                                      for k, v in sizes.items()}}
        if mid_vis is not None:
            cv2.imwrite(f"{RES}/D_{task}.png", mid_vis)
            tiles.append(cv2.resize(mid_vis, (400, 300)))
        json.dump(allstats, open(f"{RES}/D_tasks_partial.json", "w"), default=str)
        log(f"   {task} 완료: {json.dumps(allstats[task]['objects'], ensure_ascii=False)[:160]}")
    if len(tiles) >= 8:
        grid = np.vstack([np.hstack(tiles[0:4]), np.hstack(tiles[4:8])])
        cv2.imwrite(f"{RES}/D_tasks_grid.png", grid)
    json.dump(allstats, open(f"{RES}/D_tasks.json", "w"), ensure_ascii=False, default=str)
    mark("D_tasks", allstats)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {"A": task_A, "B": task_B, "C": task_C, "D": task_D}
    order = ["A", "B", "C", "D"] if which == "all" else [which.upper()]
    for k in order:
        try:
            log(f"===== 작업 {k} 시작 =====")
            jobs[k]()
        except Exception:
            log(f"!! 작업 {k} 예외:\n{traceback.format_exc()}")
    log("===== run_extra 종료 =====")
