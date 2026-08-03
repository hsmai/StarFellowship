"""전 데이터셋 3D BBox 추출 — 작업 큐 방식 (GPU 여러 장이 같은 큐를 나눠 처리).

설계 원칙
  - 작업 단위 = 에피소드 1개. Brainco 1,598개 + HE 8,949개 = 총 10,547개.
  - claim: `mkdir`의 원자성으로 락 -> 두 워커가 같은 에피소드를 잡지 않는다.
  - done:  결과 JSON이 기록되면 완료. 재시작 시 done된 것은 건너뛴다.
  - stale: 워커가 죽어 claim만 남으면 STALE_SEC 후 회수해 다른 워커가 처리한다.
  => GPU가 끊겨도 진행분이 남고, 재실행하면 끊긴 지점부터 이어서 진행된다.

사용법
  python queue_runner.py build                 # 큐 생성 (1회)
  python queue_runner.py work <worker_id>      # 워커 실행 (GPU당 1개)
  python queue_runner.py status                # 진행률
"""
import os, sys, json, glob, time, gzip, shutil, subprocess, traceback
import numpy as np
import cv2

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import Intrinsics
from track3d import EpisodeTracker

V2 = f"{ROOT}/v2"
QUEUE = f"{V2}/queue.jsonl"
CLAIMS, DONE, OUT, REP, LOGD = f"{V2}/claims", f"{V2}/done", f"{V2}/out", f"{V2}/rep", f"{V2}/logs"
for d in (V2, CLAIMS, DONE, OUT, REP, LOGD):
    os.makedirs(d, exist_ok=True)

DATA = "/data2/humanoid_dataset_isangmin"
HE = f"{DATA}/humanoid-everyday"
DEV = "cuda:0"                      # CUDA_VISIBLE_DEVICES로 물리 GPU 1장만 노출
STALE_SEC = 3600                    # 1시간 넘게 진행 없는 claim은 회수

BC_STRIDE, BC_MAX = 5, 200          # 30fps -> 6fps 샘플 (영상이 부드럽게 보이도록)
HE_STRIDE, HE_MAX = 5, 90

# 에피소드는 같은 태스크의 반복 시도라 전수 처리는 중복이 크다.
# 태스크별로 균등 샘플링해 '모든 태스크 종류'를 커버하는 것이 파이프라인 검증의 핵심.
# fps를 3->6으로 올린 만큼 에피소드 수를 줄여 총 처리량을 8시간 안에 맞춘다.
BC_PER_TASK = 12                    # 8태스크 x 12 = 96 에피소드
HE_PER_TASK = 3                     # 246태스크 x 3 = 약 738 에피소드
SAMPLE_SEED = 20260803

BRAINCO_TASKS = {
    "GraspOreo":       ("oreo snack package . plate .", {"oreo": "oreo", "plate": "plate"}),
    "GraspRubiksCube": ("rubiks cube . plate .",        {"cube": "rubiks cube", "plate": "plate"}),
    "PickApple":       ("apple . plate .",              {"apple": "apple", "plate": "plate"}),
    "PickCharger":     ("white charger . plate .",      {"charger": "charger", "plate": "plate"}),
    "PickDoll":        ("stuffed animal toy . plate .", {"toy": "doll", "plate": "plate"}),
    "PickDrink":       ("red cup . plate .",            {"cup": "red cup", "plate": "plate"}),
    "PickTissues":     ("pack of wet wipes . plate .",  {"wipes": "tissue pack", "plate": "plate"}),
    "PickToothpaste":  ("toothpaste tube . plate .",    {"toothpaste": "toothpaste", "plate": "plate"}),
}
# 육안 검증용 대표 에피소드 (영상+이미지 저장). 태스크/카테고리당 1개.
REP_BC_EP = 5
REP_HE = {}          # build 시 카테고리별 첫 에피소드로 채움


def log(msg, wid="main"):
    line = f"[{time.strftime('%F %T')}][{wid}] {msg}"
    print(line, flush=True)
    with open(f"{LOGD}/{wid}.log", "a") as f:
        f.write(line + "\n")


# ============================================================ 큐 생성
def build_queue():
    """태스크별 균등 샘플링으로 큐 생성. 대표 에피소드는 반드시 포함한다."""
    import pandas as pd
    import random
    from prompts_he import build as build_he_prompts
    rng = random.Random(SAMPLE_SEED)
    jobs = []

    # --- Brainco: 태스크당 BC_PER_TASK개 (대표 ep5는 항상 포함)
    for task, (prompt, targets) in BRAINCO_TASKS.items():
        root = f"{DATA}/G1_Brainco_{task}_Dataset"
        f = sorted(glob.glob(f"{root}/meta/episodes/chunk-000/*.parquet"))
        if not f:
            continue
        all_eps = sorted(int(x) for x in pd.read_parquet(f[0])["episode_index"])
        rep_ep = REP_BC_EP if REP_BC_EP in all_eps else all_eps[0]
        pool = [e for e in all_eps if e != rep_ep]
        pick = [rep_ep] + sorted(rng.sample(pool, min(BC_PER_TASK - 1, len(pool))))
        for ep in pick:
            jobs.append(dict(id=f"BC_{task}_{ep:04d}", ds="bc", task=task, ep=ep,
                             prompt=prompt, targets=targets, rep=(ep == rep_ep),
                             n_total=len(all_eps)))

    # --- HE: 태스크당 HE_PER_TASK개 (카테고리 대표 1개는 항상 포함)
    tb = build_he_prompts(HE)
    eps = [json.loads(l) for l in open(f"{HE}/meta/episodes.jsonl")]
    byti = {}
    for e in eps:
        ep = e["episode_index"]
        ch = ep // 1000
        if not os.path.exists(f"{HE}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet"):
            continue
        byti.setdefault(e["tasks"][0], []).append(e)
    seen_cat = set()
    for ti, elist in sorted(byti.items()):
        info = tb.get(ti)
        if info is None:
            continue
        cat = info["category"]
        # 카테고리 대표: 길이가 적당한 첫 에피소드
        rep_e = None
        if cat not in seen_cat:
            cand = [e for e in elist if 250 <= e["length"] <= 650]
            if cand:
                rep_e = cand[0]
                seen_cat.add(cat)
        pool = [e for e in elist if rep_e is None or e["episode_index"] != rep_e["episode_index"]]
        k = HE_PER_TASK - (1 if rep_e else 0)
        pick = ([rep_e] if rep_e else []) + rng.sample(pool, min(k, len(pool)))
        for e in pick:
            ep = e["episode_index"]
            jobs.append(dict(id=f"HE_{ep:05d}", ds="he", task=info["task"], cat=cat, ep=ep,
                             prompt=info["prompt"], targets=info["targets"],
                             rep=(rep_e is not None and ep == rep_e["episode_index"]),
                             n_total=len(elist)))

    with open(QUEUE, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    nb = sum(1 for j in jobs if j["ds"] == "bc")
    nt_bc = len(set(j["task"] for j in jobs if j["ds"] == "bc"))
    nt_he = len(set(j["task"] for j in jobs if j["ds"] == "he"))
    log(f"큐 생성: 총 {len(jobs)} (Brainco {nb}/{nt_bc}태스크, HE {len(jobs)-nb}/{nt_he}태스크), "
        f"대표 {sum(1 for j in jobs if j['rep'])}")
    return jobs


def load_queue():
    return [json.loads(l) for l in open(QUEUE)]


# ============================================================ claim / done
def is_done(jid):
    return os.path.exists(f"{DONE}/{jid}.json")


def try_claim(jid):
    """mkdir 원자성으로 락. 이미 있으면 stale 여부 확인 후 회수."""
    p = f"{CLAIMS}/{jid}"
    try:
        os.mkdir(p)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(p) > STALE_SEC:
                shutil.rmtree(p, ignore_errors=True)
                os.mkdir(p)
                return True
        except Exception:
            pass
        return False


def heartbeat(jid):
    try:
        os.utime(f"{CLAIMS}/{jid}", None)
    except Exception:
        pass


def release(jid, ok):
    shutil.rmtree(f"{CLAIMS}/{jid}", ignore_errors=True) if not ok else None


# ============================================================ 프레임 로더
def frames_brainco(task, ep, stride=BC_STRIDE, cap_n=BC_MAX):
    import pandas as pd
    root = f"{DATA}/G1_Brainco_{task}_Dataset"
    epm = pd.read_parquet(sorted(glob.glob(f"{root}/meta/episodes/chunk-000/*.parquet"))[0])
    rows = epm[epm.episode_index == ep]
    if len(rows) == 0:
        return [], None
    r = rows.iloc[0]
    key = "videos/observation.images.cam_left_high"
    fi = int(r[f"{key}/file_index"])
    t0, t1 = float(r[f"{key}/from_timestamp"]), float(r[f"{key}/to_timestamp"])
    mp4 = f"{root}/videos/observation.images.cam_left_high/chunk-000/file-{fi:03d}.mp4"
    fps_out = 30.0 / stride
    tmp = f"/tmp/q_{task}_{ep}_{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    os.system(f"rm -f {tmp}/*.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0), "-t", str(t1 - t0),
                    "-i", mp4, "-vf", f"fps={fps_out}", f"{tmp}/f%05d.png"], check=True)
    files = sorted(glob.glob(f"{tmp}/f*.png"))
    if cap_n and len(files) > cap_n:                 # 균등 솎기
        idx = np.linspace(0, len(files) - 1, cap_n).astype(int)
        files = [files[i] for i in idx]
    return files, tmp


def frames_he(ep, stride=HE_STRIDE, cap_n=HE_MAX):
    """RGB(mp4) + 실측 depth(parquet) 동기 로드. 반환 [(img, depth)...]"""
    import pyarrow.parquet as pq
    ch = ep // 1000
    tbl = pq.read_table(f"{HE}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet",
                        columns=["observation.depth.egocentric"])
    n = tbl.num_rows
    want = list(range(0, n, stride))
    if cap_n and len(want) > cap_n:
        want = [want[i] for i in np.linspace(0, len(want) - 1, cap_n).astype(int)]
    wset = set(want)
    darr = tbl["observation.depth.egocentric"]
    cap = cv2.VideoCapture(f"{HE}/videos/chunk-{ch:03d}/egocentric/episode_{ep:06d}.mp4")
    out, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in wset and i < n:
            d = np.array(darr[i].as_py(), dtype=np.float32)
            out.append((img, d))
        i += 1
    cap.release()
    return out


# ============================================================ 에피소드 처리
def process_job(job, models, wid):
    det, seg, dep = models
    ds, ep = job["ds"], job["ep"]
    targets, prompt = job["targets"], job["prompt"]
    trk = EpisodeTracker(targets, det, seg, prompt)
    recs = []
    rep = job.get("rep", False)
    vw = None
    tmpdir = None
    snap_mid = snap_carry = None

    if ds == "bc":
        files, tmpdir = frames_brainco(job["task"], ep)
        if not files:
            return None
        seq = [(cv2.imread(f), None) for f in files]
    else:
        seq = frames_he(ep)
        if not seq:
            return None

    H, W = seq[0][0].shape[:2]
    if rep:
        tag = f"BC_{job['task']}" if ds == "bc" else f"HE_{job.get('cat','x')}"
        fps_out = 30.0 / (BC_STRIDE if ds == "bc" else HE_STRIDE)   # 원본 속도로 재생
        vw = cv2.VideoWriter(f"{REP}/{tag}_ep{ep}.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_out, (W, H))

    for i, (img, draw_depth) in enumerate(seq):
        if ds == "bc":                                  # 3단계: depth 추정
            depth, K_pred = dep(img)
            K = (Intrinsics(float(K_pred[0, 0]), float(K_pred[1, 1]),
                            float(K_pred[0, 2]), float(K_pred[1, 2]))
                 if K_pred is not None else Intrinsics.from_fov(W, H, 70.0))
            dscale, adx = 1.0, 0
        else:                                           # 3단계: 실측 depth
            depth = draw_depth
            if depth.ndim == 1:
                depth = depth.reshape(depth.size // W, W)
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            K = Intrinsics.from_fov(W, H, 70.0)
            dscale, adx = 1e-3, -20
        rs = trk.step(img, depth, K, dscale, align_dx=adx)
        recs.append(dict(f=i, r=[{k: v for k, v in x.items() if k != "_b"} for x in rs]))
        if rep:
            vis = trk.draw(img.copy(), rs, K)
            vw.write(vis)
            if i == len(seq) // 2:
                snap_mid = vis.copy()
            if snap_carry is None and any(x.get("accepted") and x["src"] in ("redet", "prop") for x in rs):
                snap_carry = vis.copy()
        if i % 20 == 0:
            heartbeat(job["id"])

    if vw:
        vw.release()
        tag = f"BC_{job['task']}" if ds == "bc" else f"HE_{job.get('cat','x')}"
        if snap_mid is not None:
            cv2.imwrite(f"{REP}/{tag}_ep{ep}_mid.png", snap_mid)
        if snap_carry is not None:
            cv2.imwrite(f"{REP}/{tag}_ep{ep}_carry.png", snap_carry)
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    st = trk.stats(len(seq))
    with gzip.open(f"{OUT}/{job['id']}.json.gz", "wt") as f:      # 프레임별 상세는 압축 보관
        json.dump(recs, f, ensure_ascii=False, default=str)
    return dict(id=job["id"], ds=ds, task=job["task"], ep=ep, n_frames=len(seq),
                prompt=prompt, stats=st)


# ============================================================ 워커
def work(wid, rep_only=False):
    import torch
    from models_wrap import Detector, Segmenter, DepthEstimator
    t0 = time.time()
    det, seg = Detector(DEV), Segmenter(DEV)
    dep = DepthEstimator(DEV)
    log(f"모델 로드 {time.time()-t0:.0f}s", wid)
    torch.cuda.reset_peak_memory_stats()
    models = (det, seg, dep)

    jobs = load_queue()
    # 대표 에피소드를 먼저 처리해 육안 검증을 빨리 할 수 있게 한다
    jobs.sort(key=lambda j: (not j.get("rep", False), j["id"]))
    if rep_only:                       # 육안 검증 단계: 대표 15건만 처리하고 멈춘다
        jobs = [j for j in jobs if j.get("rep")]
        log(f"대표 전용 모드 — {len(jobs)}건", wid)
    n_ok = n_fail = n_frames = 0
    tstart = time.time()
    for j in jobs:
        jid = j["id"]
        if is_done(jid) or not try_claim(jid):
            continue
        try:
            r = process_job(j, models, wid)
            if r is None:
                raise RuntimeError("프레임 로드 실패")
            json.dump(r, open(f"{DONE}/{jid}.json", "w"), ensure_ascii=False, default=str)
            n_ok += 1
            n_frames += r["n_frames"]
            if n_ok % 10 == 0:
                el = time.time() - tstart
                log(f"{n_ok}건 완료 ({n_frames}프레임, {el/max(n_frames,1):.2f}s/frame, "
                    f"peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB)", wid)
        except Exception as e:
            n_fail += 1
            log(f"실패 {jid}: {str(e)[:160]}", wid)
            with open(f"{LOGD}/fail_{wid}.log", "a") as f:
                f.write(f"{jid}\n{traceback.format_exc()}\n")
            release(jid, False)
    el = time.time() - tstart
    log(f"워커 종료 — 성공 {n_ok} 실패 {n_fail}, {n_frames}프레임, "
        f"{el/max(n_frames,1):.2f}s/frame, peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB", wid)


def status():
    jobs = load_queue()
    d = len(glob.glob(f"{DONE}/*.json"))
    c = len(glob.glob(f"{CLAIMS}/*"))
    bc = len(glob.glob(f"{DONE}/BC_*.json"))
    he = len(glob.glob(f"{DONE}/HE_*.json"))
    print(f"전체 {len(jobs)} | 완료 {d} ({100*d/max(len(jobs),1):.1f}%) "
          f"[BC {bc} / HE {he}] | 진행중 {c-d if c>d else 0}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "build":
        build_queue()
    elif cmd == "work":
        work(sys.argv[2] if len(sys.argv) > 2 else "w0", rep_only=("--rep" in sys.argv))
    else:
        status()
