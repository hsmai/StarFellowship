"""V2 Humanoid Everyday — 7개 카테고리 전 에피소드 지원 러너 (GPU 1장).

기본 실행: 카테고리별 대표 에피소드 1개씩 전 프레임 3D box 추출
  -> 오버레이 mp4 + 대표 png + json
개별 실행: python run_he_v2.py <ep> "<prompt>" "<key=label,key=label>"

HE 특성:
  - 실측 depth(mm, parquet 내장)가 있어 3단계 추정을 건너뛰고 GT depth로 역투영
    (파이프라인 그림의 3단계는 depth '확보' 단계 — 실측이 있으면 그것이 상위 호환)
  - intrinsics 메타 부재 -> FOV 70° 가정 (STEP 0.5에서 테이블 평면 잔차 5.3mm로 검증)
  - RGB-depth 정렬 오차 dx=-20px 보정 (STEP 0.5 실측)
  - 검출·마스크는 Brainco와 동일 GPU 파이프라인, track3d 게이트 동일 적용
"""
import os, sys, json, time
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
HE = "/data2/humanoid_dataset_isangmin/humanoid-everyday"
DEV = "cuda:0"
ALIGN_DX = -20

# 카테고리 -> (대표 ep, 프롬프트, {매칭키: 표시라벨})
# ep 선정: 카테고리별 태스크 중 객체가 명확한 것, g1, 길이 250~650 (meta 전수 조회로 선정)
CATS = {
    "Basic":       (3800, "pink toy . orange bowl .",  {"toy": "pink toy", "bowl": "orange bowl"}),
    "Articulated": (285,  "laptop .",                  {"laptop": "laptop"}),
    "deformable":  (4838, "towel .",                   {"towel": "towel"}),
    "HRI":         (5598, "flower . person .",         {"flower": "flower", "person": "person"}),
    "Locomanip":   (7120, "bottle . box container .",  {"bottle": "bottle", "container": "container"}),
    "Precision":   (7918, "flower . vase .",           {"flower": "flower", "vase": "vase"}),
    "Tool_use":    (8198, "cleaning duster brush .",   {"duster": "duster", "brush": "duster"}),
}


def log(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(f"{LOGD}/he_v2.log", "a") as f:
        f.write(line + "\n")


def load_episode(ep, stride=2):
    """RGB 프레임(mp4 순차 읽기) + 실측 depth(parquet) 동기 로드"""
    import pyarrow.parquet as pq
    ch = ep // 1000
    tbl = pq.read_table(f"{HE}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet",
                        columns=["observation.depth.egocentric"])
    darr = tbl["observation.depth.egocentric"].to_pylist()
    cap = cv2.VideoCapture(f"{HE}/videos/chunk-{ch:03d}/egocentric/episode_{ep:06d}.mp4")
    frames = []
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % stride == 0 and i < len(darr):
            d = np.array(darr[i], dtype=np.float32)
            frames.append((img, d))
        i += 1
    cap.release()
    return frames


def run_episode(cat, ep, prompt, targets, det, seg, stride=2):
    frames = load_episode(ep, stride)
    if not frames:
        log(f"[{cat}] ep{ep}: 프레임 로드 실패!")
        return None
    log(f"[{cat}] ep{ep}: {len(frames)}프레임, prompt='{prompt}'")
    trk = EpisodeTracker(targets, det, seg, prompt)
    img0, d0 = frames[0]
    H, W = img0.shape[:2]
    if d0.ndim == 1:                                    # 평탄화 저장 대응
        side = d0.size // W
        d0 = d0.reshape(side, W)
    dH, dW = d0.shape
    K = Intrinsics.from_fov(dW, dH, 70.0)
    vw = cv2.VideoWriter(f"{RES}/HE_{cat}_ep{ep}.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    recs, snap_mid, snap_carry = [], None, None
    for i, (img, depth) in enumerate(frames):
        if depth.ndim == 1:
            depth = depth.reshape(dH, dW)
        if (dH, dW) != (H, W):                          # depth 해상도가 다르면 RGB에 맞춤
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            Kf = Intrinsics.from_fov(W, H, 70.0)
        else:
            Kf = K
        rs = trk.step(img, depth, Kf, 1e-3, align_dx=ALIGN_DX)   # mm -> m
        vis = trk.draw(img.copy(), rs, Kf)
        vw.write(vis)
        recs.append(dict(frame=i, results=[{k: v for k, v in r.items() if k != "_b"} for r in rs]))
        if i == len(frames) // 2:
            snap_mid = vis.copy()
        if snap_carry is None and any(
                r.get("accepted") and r["src"] in ("redet", "prop") for r in rs):
            snap_carry = vis.copy()
        if i % 60 == 0:
            log(f"  {cat} {i}/{len(frames)}")
    vw.release()
    if snap_mid is not None:
        cv2.imwrite(f"{RES}/HE_{cat}_ep{ep}_mid.png", snap_mid)
    if snap_carry is not None:
        cv2.imwrite(f"{RES}/HE_{cat}_ep{ep}_carry.png", snap_carry)
    st = trk.stats(len(frames))
    json.dump(recs, open(f"{RES}/HE_{cat}_ep{ep}.json", "w"), ensure_ascii=False, default=str)
    return dict(cat=cat, ep=ep, n_frames=len(frames), stats=st)


def main():
    from models_wrap import Detector, Segmenter
    t_load = time.time()
    det, seg = Detector(DEV), Segmenter(DEV)      # HE는 depth 모델 불필요 (실측 사용)
    load_s = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()
    if len(sys.argv) >= 3:                        # 개별 모드
        ep = int(sys.argv[1]); prompt = sys.argv[2]
        targets = dict(kv.split("=") for kv in sys.argv[3].split(",")) if len(sys.argv) > 3 \
            else {prompt.split()[0]: prompt.split()[0]}
        s = run_episode("custom", ep, prompt, targets, det, seg)
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return
    summary, t0 = {}, time.time()
    for cat, (ep, prompt, targets) in CATS.items():
        tag = f"V2_HE_{cat}"
        if os.path.exists(f"{MK}/{tag}.done"):
            log(f"[{cat}] 이미 완료 — 건너뜀")
            summary[cat] = json.load(open(f"{MK}/{tag}.done"))
            continue
        s = run_episode(cat, ep, prompt, targets, det, seg)
        if s is None:
            continue
        json.dump(s, open(f"{MK}/{tag}.done", "w"), ensure_ascii=False, default=str)
        summary[cat] = s
        log(f"[{cat}] 완료: {json.dumps(s['stats'], ensure_ascii=False)}")
    peak = torch.cuda.max_memory_allocated() / 1e9
    wall = time.time() - t0
    nfr = sum(s["n_frames"] for s in summary.values() if isinstance(s, dict) and "n_frames" in s)
    summary["_resource"] = dict(model_load_s=round(load_s, 1), peak_vram_GB=round(peak, 2),
                                wall_s=round(wall, 1), n_frames=nfr,
                                s_per_frame=round(wall / max(nfr, 1), 2))
    json.dump(summary, open(f"{RES}/HE_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
    log(f"전체 완료 — peak VRAM {peak:.2f}GB, {wall:.0f}s, {nfr}프레임")


if __name__ == "__main__":
    main()
