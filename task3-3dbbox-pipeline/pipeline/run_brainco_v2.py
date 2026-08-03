"""V2 Brainco — 8개 태스크 전 에피소드 지원 러너 (GPU 1장).

기본 실행(인자 없음): 태스크당 대표 에피소드 1개씩 전 프레임 3D box 추출
  -> 오버레이 mp4 + 대표 png 2장(정지/이동 구간) + json
개별 실행: python run_brainco_v2.py <Task> <ep>   (모든 에피소드에 동일 적용 가능)

개선점 (run_extra.py 대비):
  - track3d.EpisodeTracker: 검출 -> 저임계 재검출 -> 마스크 전파(박스 매 프레임 갱신)
  - 3중 게이트로 튀는 박스 기각 (틀린 박스는 그리지 않음)
  - EMA 스무딩, 마스크의 박스 밖 유입 차단(케이블 문제)
  - doll/tissue 프롬프트 교체
"""
import os, sys, json, glob, time, subprocess
import numpy as np
import cv2
import torch

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import Intrinsics
from track3d import EpisodeTracker

MK, RES, LOGD = f"{ROOT}/markers", f"{ROOT}/results_v2", f"{ROOT}/logs"
for d in (MK, RES, LOGD):
    os.makedirs(d, exist_ok=True)
DATA = "/data2/humanoid_dataset_isangmin"
DEV = "cuda:0"

# 태스크 -> (프롬프트, {매칭키: 표시라벨})  ※ doll/tissue는 D에서 검출 실패해 프롬프트 교체
TASKS = {
    "GraspOreo":       ("oreo snack package . plate .", {"oreo": "oreo", "plate": "plate"}),
    "GraspRubiksCube": ("rubiks cube . plate .",        {"cube": "rubiks cube", "plate": "plate"}),
    "PickApple":       ("apple . plate .",              {"apple": "apple", "plate": "plate"}),
    "PickCharger":     ("white charger . plate .",      {"charger": "charger", "plate": "plate"}),
    "PickDoll":        ("stuffed animal toy . plate .", {"toy": "doll", "plate": "plate"}),
    "PickDrink":       ("red cup . plate .",            {"cup": "red cup", "plate": "plate"}),
    "PickTissues":     ("pack of wet wipes . plate .",  {"wipes": "tissue pack", "plate": "plate"}),
    "PickToothpaste":  ("toothpaste tube . plate .",    {"toothpaste": "toothpaste", "plate": "plate"}),
}
REP_EP = 5      # 대표 에피소드 (없으면 첫 에피소드로 대체)


def log(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(f"{LOGD}/bc_v2.log", "a") as f:
        f.write(line + "\n")


def episode_frames(task, ep, stride=2):
    """통합 mp4에서 에피소드 구간 프레임 추출 (기존 run_extra와 동일 방식)"""
    import pandas as pd
    root = f"{DATA}/G1_Brainco_{task}_Dataset"
    epm = pd.read_parquet(sorted(glob.glob(f"{root}/meta/episodes/chunk-000/*.parquet"))[0])
    rows = epm[epm.episode_index == ep]
    if len(rows) == 0:
        ep = int(epm.episode_index.iloc[0])
        rows = epm[epm.episode_index == ep]
    r = rows.iloc[0]
    key = "videos/observation.images.cam_left_high"
    fi = int(r[f"{key}/file_index"])
    t0, t1 = float(r[f"{key}/from_timestamp"]), float(r[f"{key}/to_timestamp"])
    mp4 = f"{root}/videos/observation.images.cam_left_high/chunk-000/file-{fi:03d}.mp4"
    outdir = f"/tmp/v2_{task}_{ep}"
    os.makedirs(outdir, exist_ok=True); os.system(f"rm -f {outdir}/*.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0), "-t", str(t1 - t0),
                    "-i", mp4, f"{outdir}/f%05d.png"], check=True)
    return sorted(glob.glob(f"{outdir}/f*.png"))[::stride], ep, outdir


def run_episode(task, ep, det, seg, dep, stride=2, video=True):
    prompt, targets = TASKS[task]
    files, ep, tmpdir = episode_frames(task, ep, stride)
    log(f"[{task}] ep{ep}: {len(files)}프레임, prompt='{prompt}'")
    trk = EpisodeTracker(targets, det, seg, prompt)
    H, W = cv2.imread(files[0]).shape[:2]
    vw = cv2.VideoWriter(f"{RES}/BC_{task}_ep{ep}.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H)) if video else None
    recs, snap_mid, snap_carry = [], None, None
    for i, f in enumerate(files):
        img = cv2.imread(f)
        depth, K_pred = dep(img)
        K = (Intrinsics(float(K_pred[0, 0]), float(K_pred[1, 1]), float(K_pred[0, 2]), float(K_pred[1, 2]))
             if K_pred is not None else Intrinsics.from_fov(W, H, 70.0))
        rs = trk.step(img, depth, K, 1.0)
        vis = trk.draw(img.copy(), rs, K)
        if vw: vw.write(vis)
        recs.append(dict(frame=i, results=[{k: v for k, v in r.items() if k != "_b"} for r in rs]))
        if i == len(files) // 2:
            snap_mid = vis.copy()
        obj_key = [k for k in targets if k != "plate"][0]
        if snap_carry is None and 0.35 < i / max(len(files), 1) < 0.75:
            for r in rs:
                if r["key"] == obj_key and r.get("accepted") and r["src"] in ("redet", "prop"):
                    snap_carry = vis.copy()
        if i % 60 == 0:
            log(f"  {task} {i}/{len(files)}")
    if vw: vw.release()
    if snap_mid is not None:
        cv2.imwrite(f"{RES}/BC_{task}_ep{ep}_mid.png", snap_mid)
    if snap_carry is not None:
        cv2.imwrite(f"{RES}/BC_{task}_ep{ep}_carry.png", snap_carry)
    st = trk.stats(len(files))
    json.dump(recs, open(f"{RES}/BC_{task}_ep{ep}.json", "w"), ensure_ascii=False, default=str)
    os.system(f"rm -rf {tmpdir}")
    return dict(task=task, ep=ep, n_frames=len(files), stats=st)


def main():
    from models_wrap import Detector, Segmenter, DepthEstimator
    t_load = time.time()
    det, seg, dep = Detector(DEV), Segmenter(DEV), DepthEstimator(DEV)
    load_s = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()
    if len(sys.argv) >= 2:                       # 개별 에피소드 모드
        s = run_episode(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else REP_EP, det, seg, dep)
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return
    summary, t0 = {}, time.time()
    for task in TASKS:
        tag = f"V2_BC_{task}"
        if os.path.exists(f"{MK}/{tag}.done"):
            log(f"[{task}] 이미 완료 — 건너뜀")
            summary[task] = json.load(open(f"{MK}/{tag}.done"))
            continue
        s = run_episode(task, REP_EP, det, seg, dep)
        json.dump(s, open(f"{MK}/{tag}.done", "w"), ensure_ascii=False, default=str)
        summary[task] = s
        log(f"[{task}] 완료: {json.dumps(s['stats'], ensure_ascii=False)}")
    peak = torch.cuda.max_memory_allocated() / 1e9
    wall = time.time() - t0
    nfr = sum(s["n_frames"] for s in summary.values())
    summary["_resource"] = dict(model_load_s=round(load_s, 1), peak_vram_GB=round(peak, 2),
                                wall_s=round(wall, 1), n_frames=nfr,
                                s_per_frame=round(wall / max(nfr, 1), 2))
    json.dump(summary, open(f"{RES}/BC_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
    log(f"전체 완료 — peak VRAM {peak:.2f}GB, {wall:.0f}s, {nfr}프레임")


if __name__ == "__main__":
    main()
