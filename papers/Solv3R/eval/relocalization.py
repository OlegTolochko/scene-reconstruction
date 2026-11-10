"""Map-based relocalization (paper Table 2): 7-Scenes and Cambridge Landmarks.

Per scene, a map is built from the training frames (7-Scenes: every 5th; Cambridge: every 1st-4th, at most ~400 views)
and solved once with 32 joint calls. Only one key / value cache of the solved map is kept. Each query (7-Scenes: every
5th test frame; Cambridge: every test frame) is encoded on its own and refined with 16 calls against the cached keys /
values of a fixed view set: the reference view, 64 evenly spaced map views and its 10 nearest map views by NetVLAD
(VGG16-Pitts30K, as in Reloc3r and hloc). The map states are never updated. The absolute query pose comes from the
similarity transform that aligns the predicted cameras of the 10 retrieved views with their ground truth.
Reported per dataset: means over scenes of the median translation / rotation error and of the recall (7-Scenes:
5 cm / 5 deg, Cambridge: 25 cm / 2 deg), the cache size per map view and the query time (encoding + calls + camera
readout, image loading excluded; query-weighted mean of the per-scene medians).

Data:
  <data-root>/7scenes/<scene>/{TrainSplit.txt, TestSplit.txt, seq-XX/frame-XXXXXX.{color.png, pose.txt}}
  <data-root>/cambridge/<Scene>/{dataset_train.txt, dataset_test.txt, seqN/frameXXXXX.png}
NetVLAD weights (555 MB) are downloaded once into the torch hub directory.

usage:
  python eval/relocalization.py --data-root /path/to/data --out results/relocalization
  python eval/relocalization.py --data-root /path/to/data --out results/relocalization --datasets 7scenes --scenes heads
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solv3r.images import load_images  # noqa: E402
from solv3r.model import Context, Solv3R, autocast, cameras_from_pose_enc  # noqa: E402
from solv3r.network.block import _attn_kv  # noqa: E402

SEVEN_SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
CAMBRIDGE = {"GreatCourt": 4, "KingsCollege": 4, "OldHospital": 3, "ShopFacade": 1, "StMarysChurch": 4}  # map stride
DATASETS = {"7scenes": dict(scenes=SEVEN_SCENES, thr=(5.0, 5.0), q_every=5),
            "cambridge": dict(scenes=tuple(CAMBRIDGE), thr=(25.0, 2.0), q_every=1)}
MAP_CALLS, QUERY_CALLS, BLOCK, N_COVER, TOP_K = 32, 16, 32, 64, 10
DEV = torch.device("cuda")


# ---------------------------------------------------------------------- data
def _quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def seven_scenes_frames(root, scene, split):
    """[(image path, camera-to-world 4x4)]; the rotations are projected onto SO(3) (the stored ones are scaled by up
    to 0.1 %)."""
    d = os.path.join(root, "7scenes", scene)
    f = os.path.join(d, f"{split}Split.txt")
    if not os.path.isfile(f):
        raise FileNotFoundError(f"7-Scenes {scene}: {f} not found (expected <data-root>/7scenes/<scene>/...)")
    out = []
    for line in open(f).read().split():
        seq = os.path.join(d, f"seq-{int(line.lower().replace('sequence', '')):02d}")
        ids = sorted(int(n[6:12]) for n in os.listdir(seq) if n.endswith(".color.png"))
        for i in ids:
            P = np.loadtxt(os.path.join(seq, f"frame-{i:06d}.pose.txt"))
            P[:3, :3] = _project_so3(P[:3, :3])
            out.append((os.path.join(seq, f"frame-{i:06d}.color.png"), P))
    return out


def cambridge_frames(root, scene, split):
    """[(image path, camera-to-world 4x4)] from dataset_{train,test}.txt (camera centre, world-to-camera quaternion)."""
    d = os.path.join(root, "cambridge", scene)
    f = os.path.join(d, f"dataset_{split.lower()}.txt")
    if not os.path.isfile(f):
        raise FileNotFoundError(f"Cambridge {scene}: {f} not found (expected <data-root>/cambridge/<Scene>/...)")
    out = []
    for line in open(f).read().split("\n")[3:]:
        t = line.split()
        if len(t) == 8:
            v = np.array([float(x) for x in t[1:]])
            P = np.eye(4)
            P[:3, :3] = _quat_to_R(v[3:]).T
            P[:3, 3] = v[:3]
            out.append((os.path.join(d, t[0]), P))
    return out


# ---------------------------------------------------------------------- NetVLAD retrieval
class NetVLAD(nn.Module):
    """VGG16-NetVLAD-Pitts30K with PCA whitening (4096-d), the MATLAB weights converted as in hloc
    (https://github.com/cvg/Hierarchical-Localization, Apache-2.0)."""

    URL = "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat"

    def __init__(self):
        super().__init__()
        try:
            import torchvision
            from scipy.io import loadmat
        except ImportError as e:
            raise ImportError(f"NetVLAD retrieval needs torchvision and scipy ({e}); pip install torchvision scipy") from e
        path = os.path.join(torch.hub.get_dir(), "netvlad", "VGG16-NetVLAD-Pitts30K.mat")
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"downloading the NetVLAD weights (555 MB) to {path}", flush=True)
            torch.hub.download_url_to_file(self.URL, path)
        mat = loadmat(path, struct_as_record=False, squeeze_me=True)
        self.backbone = nn.Sequential(*list(torchvision.models.vgg16().features.children())[:-2])
        for layer, ml in zip(self.backbone.children(), mat["net"].layers):
            if isinstance(layer, nn.Conv2d):
                layer.weight = nn.Parameter(torch.tensor(ml.weights[0]).float().permute(3, 2, 0, 1))
                layer.bias = nn.Parameter(torch.tensor(ml.weights[1]).float())
        self.score = nn.Conv1d(512, 64, 1, bias=False)
        self.score.weight = nn.Parameter(torch.tensor(mat["net"].layers[30].weights[0]).float().permute(1, 0)[..., None])
        self.centers = nn.Parameter(-torch.tensor(mat["net"].layers[30].weights[1]).float())          # (512, 64)
        w, b = mat["net"].layers[33].weights
        self.whiten = nn.Linear(512 * 64, 4096)
        self.whiten.weight = nn.Parameter(torch.tensor(w).float().squeeze().permute(1, 0))
        self.whiten.bias = nn.Parameter(torch.tensor(b.squeeze()).float())
        self.register_buffer("mean", torch.tensor(mat["net"].meta.normalization.averageImage[0, 0]).float().view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, image):
        """image (1, 3, H, W) in [0, 1] -> (1, 4096) descriptor"""
        x = self.backbone((image * 255).clamp(0, 255) - self.mean).flatten(2)
        x = F.normalize(x, dim=1)
        a = F.softmax(self.score(x), dim=1)
        d = (a[:, None] * (x[:, :, None] - self.centers[None, :, :, None])).sum(-1)
        d = F.normalize(F.normalize(d, dim=1).flatten(1), dim=1)
        return F.normalize(self.whiten(d), dim=1)


def _netvlad_image(path, long_side):
    """The retrieval input as in Reloc3r: OpenCV channel order, min-max normalised; Cambridge at a 1024-pixel long side."""
    import cv2

    im = cv2.imread(path)
    if long_side and max(im.shape[:2]) > long_side:
        s = long_side / max(im.shape[:2])
        im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    x = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)[None].to(DEV)
    return (x - x.min()) / (x.max() - x.min())


def retrieve(net, map_paths, query_paths, long_side, k=TOP_K):
    """Top-k map views of every query by descriptor similarity -> (Q, k) map indices."""
    D = torch.cat([net(_netvlad_image(p, long_side)) for p in map_paths])
    Q = torch.cat([net(_netvlad_image(p, long_side)) for p in query_paths])
    return torch.topk(Q @ D.T, k, dim=1).indices.cpu().tolist()


# ---------------------------------------------------------------------- solved map with one K/V cache
def _attn_over_map(attn, x, K, V, own, P, chunk=128):
    """Global attention of the tokens x (1, n, C) of the map views `own` (a range) over the keys / values K, V
    (1, H, N * P, D) of all other map views plus their own: flash attention over chunks of `chunk` views, merged exactly
    by log-sum-exp in fp32."""
    B, n, C = x.shape
    q, k, v = attn.qkv(x).reshape(B, n, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
    bf = torch.bfloat16
    q, k, v = attn.q_norm(q).to(bf), attn.k_norm(k).to(bf), v.to(bf)
    parts = []
    for c0 in range(0, K.shape[2] // P, chunk):
        c1 = min(K.shape[2] // P, c0 + chunk)
        keep = [r for r in ((c0, min(c1, own.start)), (max(c0, own.stop), c1)) if r[1] > r[0]]
        if keep:
            parts.append([torch.cat([T[:, :, a * P:b * P] for a, b in keep], 2) for T in (K, V)])
    parts.append([k, v])
    out = m = w = None
    for kc, vc in parts:
        o, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, kc, vc, 0.0, False, False)[:2]
        o = o.float()
        if out is None:
            out, m, w = o, lse, torch.ones_like(lse)
            continue
        mn = torch.maximum(m, lse)
        a_, b_ = torch.exp(m - mn), torch.exp(lse - mn)
        out, w, m = out * a_[..., None] + o * b_[..., None], w * a_ + b_, mn
    o = (out / w[..., None]).to(bf)
    return attn.proj(o.transpose(1, 2).reshape(B, n, C))


class MapSolver:
    """Joint calls on a map of N views with the states on the host and the global keys / values (N * P tokens) on the
    GPU. A call runs block by block, every block attending to the keys / values of all views, then one state scale over
    all views: the same update as Solv3R.step on all N views at once (in a fixed summation order)."""

    def __init__(self, model, ctx_host, pos):
        self.net, self.blk = model.net, model.net.block
        self.pos1 = pos[:1]                                                     # RoPE positions are the same for every view
        self.x0 = ctx_host.x0                                                   # (N, P, C) host
        self.N, self.P, self.C = self.x0.shape
        self.H, self.W = ctx_host.H, ctx_host.W
        att = self.blk.global_attn.attn
        D = self.C // att.num_heads
        self.K = torch.empty(1, att.num_heads, self.N * self.P, D, dtype=torch.bfloat16, device=DEV)
        self.V = torch.empty_like(self.K)
        self.z = self.x0.clone()                                                # start state = encoding
        self.energy = None
        self._write_kv(range(0, self.N), self.z)

    def _pos(self, n):
        return self.pos1.expand(n, -1, -1)

    def _write_kv(self, views, z_src, chunk=16):
        views = list(views)
        for a in range(0, len(views), chunk):
            v = views[a:a + chunk]
            y = self.blk._frame(z_src[v[0]:v[-1] + 1].to(DEV), self.x0[v[0]:v[-1] + 1].to(DEV), self._pos(len(v)))
            k, vv = _attn_kv(self.blk.global_attn.attn, self.blk.global_attn.norm1(y.reshape(1, -1, self.C)))
            self.K[:, :, v[0] * self.P:(v[-1] + 1) * self.P] = k
            self.V[:, :, v[0] * self.P:(v[-1] + 1) * self.P] = vv

    @torch.no_grad()
    def call(self, block=BLOCK):
        blk, attn = self.blk, self.blk.global_attn.attn
        ynew, energy, total = None, torch.zeros(self.N, device=DEV), torch.zeros((), device=DEV)
        for a in range(0, self.N, block):
            b = min(self.N, a + block) - a
            y = blk._frame(self.z[a:a + b].to(DEV), self.x0[a:a + b].to(DEV), self._pos(b)).reshape(1, b * self.P, self.C)
            y = blk.global_attn(y, blk.g_global, attn_fn=lambda x: _attn_over_map(
                attn, x, self.K, self.V, range(a, a + b), self.P)).reshape(b, self.P, self.C)
            if ynew is None:
                ynew = torch.empty(self.N, self.P, self.C, dtype=y.dtype)
            ynew[a:a + b].copy_(y)
            energy[a:a + b] = y.float().pow(2).sum((1, 2))
            total = total + energy[a:a + b].sum()
        rms = (total / (self.N * self.P * self.C)).clamp_min(1e-12).sqrt()
        scale = (blk.target_rms[3] / rms).to(ynew.dtype)
        for a in range(0, self.N, 16):
            self.z[a:a + 16].copy_((ynew[a:a + 16].to(DEV) * scale).float())
        self._write_kv(range(0, self.N), self.z)
        self.energy = energy                                    # per-view state energy before scaling (last call)

    @torch.no_grad()
    def cameras(self, group=64, chunk=32):
        """Camera-to-world (N, 4, 4) of the map views (camera tokens of the ray head, then the camera head)."""
        pe = []
        for g in range(0, self.N, group):
            tok = []
            for a in range(g, min(self.N, g + group), chunk):
                z = self.z[a:min(a + chunk, g + group)].to(DEV)
                tok.append(self.net.ray_head.tokens(z, self._pos(len(z)))[:, 0])
            tok = torch.cat(tok)
            pe.append(self.net.camera_head(tok))
        pe = torch.cat(pe)
        return cameras_from_pose_enc(pe, (self.H, self.W))[0].double().numpy()

    def cache_of(self, views):
        """The frozen-map cache seen by a query: keys / values of these views and their share of the state scale."""
        v = torch.as_tensor(views, device=DEV)
        tok = (v[:, None] * self.P + torch.arange(self.P, device=DEV)[None]).reshape(-1)
        return dict(k=self.K.index_select(2, tok), v=self.V.index_select(2, tok),
                    sum_sq=self.energy[v].sum(), numel=len(views) * self.P * self.C)


# ---------------------------------------------------------------------- pose
def absolute_pose(P_ref, G_ref, P_q):
    """Similarity that maps the predicted reference cameras onto their ground truth, applied to the query (c2w)."""
    R = _project_so3(sum(g[:3, :3] @ p[:3, :3].T for p, g in zip(P_ref, G_ref)))
    X, Y = (R @ P_ref[:, :3, 3].T).T, G_ref[:, :3, 3]
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    s = max((Xc * Yc).sum() / max((Xc ** 2).sum(), 1e-12), 1e-6)
    Q = np.eye(4)
    Q[:3, :3] = R @ P_q[:3, :3]
    Q[:3, 3] = s * R @ P_q[:3, 3] + Y.mean(0) - s * X.mean(0)
    return Q


def pose_error(P, G):
    """translation error (cm, poses in metres), rotation error (deg)"""
    c = (np.trace(P[:3, :3].T @ G[:3, :3]) - 1) / 2
    return float(np.linalg.norm(P[:3, 3] - G[:3, 3]) * 100), float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _sync():
    torch.cuda.synchronize()
    return time.perf_counter()


# ---------------------------------------------------------------------- one scene
@torch.no_grad()
def run_scene(model, net, root, dataset, scene, out_dir):
    frames = seven_scenes_frames if dataset == "7scenes" else cambridge_frames
    train, test = frames(root, scene, "Train"), frames(root, scene, "Test")
    stride = 5 if dataset == "7scenes" else CAMBRIDGE[scene]
    mi = list(range(0, len(train), stride))
    qs = list(range(0, len(test), DATASETS[dataset]["q_every"]))
    G_map = np.stack([train[i][1] for i in mi])
    N = len(mi)

    ret_file = os.path.join(out_dir, "retrieval", f"{dataset}_{scene}.json")
    if os.path.isfile(ret_file):
        top = json.load(open(ret_file))["top10"]
    else:
        top = retrieve(net, [train[i][0] for i in mi], [test[q][0] for q in qs], 1024 if dataset == "cambridge" else 0)
        os.makedirs(os.path.dirname(ret_file), exist_ok=True)
        json.dump(dict(map=[train[i][0] for i in mi], queries=[test[q][0] for q in qs], top10=top), open(ret_file, "w"))
    cover = list(dict.fromkeys([0] + np.linspace(0, N - 1, N_COVER).round().astype(int).tolist()))

    torch.cuda.reset_peak_memory_stats()
    t0 = _sync()
    images = torch.cat([load_images([train[i][0] for i in mi[a:a + 64]]) for a in range(0, N, 64)])
    H, W = images.shape[-2:]
    x0 = []
    for a in range(0, N, 32):
        c = model.encode(images[a:a + 32], reference_first=(a == 0))
        x0.append(c.x0.cpu())
        pos = c.pos
    del images
    with autocast():
        S = MapSolver(model, Context(torch.cat(x0), None, H, W), pos)
        del x0
        for _ in range(MAP_CALLS):
            S.call()
        P_map = S.cameras()
    build_s = _sync() - t0
    S.z = S.x0 = None
    torch.cuda.empty_cache()
    mb_per_view = (S.K.numel() + S.V.numel()) * S.K.element_size() / 2 ** 20 / N
    print(f"  {scene}: map of {N} views solved in {build_s:.0f} s, cache {mb_per_view:.2f} MB / view, {len(qs)} queries",
          flush=True)

    errors, ms = [], []
    for q, t10 in zip(qs, top):
        im = load_images([test[q][0]], size=(H, W))
        t0 = _sync()
        ctx = model.encode(im, reference_first=False)
        cache = S.cache_of(list(dict.fromkeys(cover + t10)))
        z = model.start(ctx)
        with autocast():
            for _ in range(QUERY_CALLS):
                z = model.step_cached(z, ctx, cache)
            pe = model.decode_cameras(z, ctx)
        P_q = cameras_from_pose_enc(pe, (H, W))[0][0].double().numpy()
        ms.append((_sync() - t0) * 1000)
        errors.append(pose_error(absolute_pose(P_map[t10], G_map[t10], P_q), test[q][1]))
    e = np.array(errors)
    thr = DATASETS[dataset]["thr"]
    res = dict(scene=scene, map_views=N, queries=len(qs), t_cm=float(np.median(e[:, 0])), r_deg=float(np.median(e[:, 1])),
               recall=float(np.mean((e[:, 0] < thr[0]) & (e[:, 1] < thr[1]))), ms=float(np.median(ms)),
               mb_per_view=mb_per_view, map_build_s=build_s, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
               per_query=[[q, round(a, 3), round(b, 4)] for q, (a, b) in zip(qs, errors)])
    print(f"  {scene}: {res['t_cm']:.2f} cm {res['r_deg']:.2f} deg recall {100 * res['recall']:.1f}  "
          f"{res['ms']:.0f} ms / query  peak {res['peak_gb']:.1f} GB", flush=True)
    del S, cache
    torch.cuda.empty_cache()
    return res


def summarize(rows, dataset):
    w = np.array([r["queries"] for r in rows], dtype=float)
    s = dict(t_cm=np.mean([r["t_cm"] for r in rows]), r_deg=np.mean([r["r_deg"] for r in rows]),
             recall=100 * np.mean([r["recall"] for r in rows]), ms=float((w * [r["ms"] for r in rows]).sum() / w.sum()),
             mb_per_view=np.mean([r["mb_per_view"] for r in rows]), scenes=len(rows))
    thr = DATASETS[dataset]["thr"]
    print(f"{dataset:10s} {s['t_cm']:.2f} cm / {s['r_deg']:.2f} deg  recall@{thr[0]:g}cm,{thr[1]:g}deg {s['recall']:.1f}  "
          f"{s['mb_per_view']:.2f} MB / view  {s['ms']:.0f} ms / query  ({len(rows)} scenes)", flush=True)
    return {k: float(v) for k, v in s.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="a .safetensors file or a Hugging Face repo id (default: checkpoints/ if present, else the released weights)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--scenes", nargs="+", default=None, help="a subset of the scenes of the chosen datasets")
    a = ap.parse_args()
    datasets = a.datasets.split(",")
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        ap.error(f"unknown --datasets {','.join(unknown)} (choose from {','.join(DATASETS)})")
    if a.scenes:
        known = {s for d in datasets for s in DATASETS[d]["scenes"]}
        if set(a.scenes) - known:
            ap.error(f"unknown --scenes {' '.join(sorted(set(a.scenes) - known))} (choose from {' '.join(sorted(known))})")
    if not torch.cuda.is_available():
        sys.exit("relocalization.py needs a CUDA GPU")
    os.makedirs(a.out, exist_ok=True)

    model = Solv3R.from_pretrained(a.checkpoint)
    net = NetVLAD().eval().to(DEV)
    summary = {}
    for ds in datasets:
        print(f"== {ds}", flush=True)
        rows = []
        for sc in DATASETS[ds]["scenes"]:
            if a.scenes and sc not in a.scenes:
                continue
            rows.append(run_scene(model, net, a.data_root, ds, sc, a.out))
            json.dump(rows[-1], open(os.path.join(a.out, f"{ds}_{sc}.json"), "w"))
        if rows:
            summary[ds] = summarize(rows, ds)
    for ds, s in summary.items():
        print(f"{ds:10s} {s['t_cm']:.2f} cm / {s['r_deg']:.2f} deg  recall {s['recall']:.1f}  "
              f"{s['mb_per_view']:.2f} MB / view  {s['ms']:.0f} ms / query")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
