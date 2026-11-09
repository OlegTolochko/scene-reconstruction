"""Saving reconstructions: cameras and depth as .npz, coloured point clouds as .ply and as a browser viewer (.html)."""
import base64
import json
import os
import warnings

import numpy as np
import torch


def write_ply(path, xyz, rgb):
    """xyz (N, 3) float, rgb (N, 3) uint8 -> binary little-endian PLY."""
    xyz, rgb = np.asarray(xyz, np.float32), np.asarray(rgb, np.uint8)
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.empty(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = xyz, rgb
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


_VIEWER = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>__TITLE__</title>
<style>html,body{margin:0;height:100%;overflow:hidden;background:#0b0d10;color:#d9dde2;font:13px system-ui,sans-serif}
#info{position:absolute;left:12px;top:10px;pointer-events:none}#ui{position:absolute;right:12px;top:10px;display:flex;gap:14px}</style>
</head><body><div id="info"></div>
<div id="ui"><label>point size <input id="size" type="range" min="1" max="40" value="8"></label>
<label><input id="cams" type="checkbox" checked> cameras</label></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const D = __DATA__;
const raw = atob(D.points), bytes = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
const q = new Uint16Array(bytes.buffer, 0, D.n * 3), pos = new Float32Array(D.n * 3);
for (let i = 0; i < D.n * 3; i++) { const k = i % 3; pos[i] = D.lo[k] + (q[i] / 65535) * (D.hi[k] - D.lo[k]); }
const geo = new THREE.BufferGeometry();
geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
geo.setAttribute("color", new THREE.BufferAttribute(new Uint8Array(bytes.buffer, D.n * 6, D.n * 3), 3, true));
const mat = new THREE.PointsMaterial({ size: 0.008, vertexColors: true, sizeAttenuation: true });
const scene = new THREE.Scene(), renderer = new THREE.WebGLRenderer();
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); document.body.appendChild(renderer.domElement);
scene.add(new THREE.Points(geo, mat));
const path = new THREE.Line(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(D.cams.flat()), 3)),
                            new THREE.LineBasicMaterial({ color: 0x6ea7e3 }));
scene.add(path);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100), controls = new THREE.OrbitControls(camera, renderer.domElement);
camera.position.set(0, 0.9, 2.4); controls.update();
const render = () => renderer.render(scene, camera);
const resize = () => { renderer.setSize(innerWidth, innerHeight); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); render(); };
controls.addEventListener("change", render); addEventListener("resize", resize);
document.getElementById("size").oninput = (e) => { mat.size = e.target.value / 1000; render(); };
document.getElementById("cams").onchange = (e) => { path.visible = e.target.checked; render(); };
document.getElementById("info").textContent = `${D.title}: ${D.n.toLocaleString()} of ${D.total.toLocaleString()} points. Drag to rotate, scroll to zoom, right-drag to pan.`;
resize();
</script></body></html>
"""


def write_html(path, xyz, rgb, c2w=None, max_points=300000, title="Solv3R"):
    """A point cloud as one self-contained .html file that opens in a browser (three.js is loaded from a CDN).
    xyz (N, 3), rgb (N, 3) uint8, c2w (S, 4, 4) the cameras: their path is drawn and they set the start view.
    Clouds larger than max_points are subsampled."""
    xyz, rgb = np.asarray(xyz, np.float64), np.asarray(rgb, np.uint8)
    total = len(xyz)
    if total > max_points:
        idx = np.sort(np.random.default_rng(0).choice(total, max_points, replace=False))
        xyz, rgb = xyz[idx], rgb[idx]
    c = np.median(xyz, 0)
    r = max(float(np.percentile(np.linalg.norm(xyz - c, axis=1), 80)), 1e-9)
    up, fwd = (np.array([0.0, -1, 0]), np.array([0.0, 0, 1])) if c2w is None else (-c2w[:, :3, 1].mean(0), c2w[:, :3, 2].mean(0))
    up = up / np.linalg.norm(up)
    fwd = fwd - (fwd @ up) * up
    fwd = fwd / np.linalg.norm(fwd) if np.linalg.norm(fwd) > 1e-6 else np.cross(up, [1.0, 0, 0])
    R = np.stack([np.cross(fwd, up), up, -fwd])                        # viewer frame: y up, the capture looks along -z
    V = (xyz - c) @ R.T / r
    keep = np.linalg.norm(V, axis=1) < 6                               # far stray points would stretch the quantization box
    V, rgb = V[keep], rgb[keep]
    lo, hi = V.min(0), V.max(0)
    q = np.round((V - lo) / np.maximum(hi - lo, 1e-9) * 65535).astype("<u2")
    cams = [] if c2w is None else ((c2w[:, :3, 3] - c) @ R.T / r).round(4).tolist()
    data = dict(title=title, n=len(V), total=total, lo=lo.tolist(), hi=hi.tolist(), cams=cams,
                points=base64.b64encode(q.tobytes() + np.ascontiguousarray(rgb).tobytes()).decode())
    with open(path, "w") as f:
        f.write(_VIEWER.replace("__TITLE__", title).replace("__DATA__", json.dumps(data)))


def unproject(depth, c2w, K):
    """depth (S, H, W), c2w (S, 4, 4), K (S, 3, 3) -> world points (S, H, W, 3)."""
    S, H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    x = (u[None] - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None] * depth
    y = (v[None] - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None] * depth
    cam = np.stack([x, y, depth], -1)
    return np.einsum("sij,shwj->shwi", c2w[:, :3, :3], cam) + c2w[:, None, None, :3, 3]


def save_solve(prefix, images, pred, names=None, keep=0.5, sky=False, html=False):
    """Saves a decoded solve (Solv3R.decode) as <prefix>.npz (c2w, K, depth, depth_conf) and <prefix>.ply.
    sky=True leaves the sky out of the point cloud (sky_mask). html=True also writes the viewer <prefix>.html."""
    from .model import cameras_from_pose_enc

    hw = tuple(pred["depth"].shape[1:3])
    c2w, K = cameras_from_pose_enc(pred["pose_enc"], hw)
    depth = pred["depth"][..., 0].float().cpu().numpy()
    conf = pred["depth_conf"].float().cpu().numpy()
    np.savez_compressed(f"{prefix}.npz", c2w=c2w.numpy(), K=K.numpy(), depth=depth.astype(np.float16),
                        depth_conf=conf.astype(np.float16), **({"names": np.array(names)} if names else {}))
    valid = ~sky_mask(images) if sky else None
    xyz, rgb = confident_points(pred["world_points"].float().cpu().numpy(), conf, images, keep, valid)
    write_ply(f"{prefix}.ply", xyz, rgb)
    if html:
        write_html(f"{prefix}.html", xyz, rgb, c2w.numpy(), title=os.path.basename(prefix))


def sky_mask(images):
    """(S, H, W) bool, True on sky, for images (S, 3, H, W) in [0, 1]: the sky segmentation network of the DVLT demo
    (onnxruntime; the model, 176 MB, is downloaded to ./skyseg.onnx on first use)."""
    from .network.skyseg import segment_sky

    with warnings.catch_warnings():                       # the CPU build of onnxruntime has no CUDA provider to offer
        warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider'")
        return segment_sky(torch.as_tensor(images).float().cpu())


def confident_points(points, conf, images, keep=0.5, valid=None):
    """Points (S, H, W, 3) of the most confident `keep` fraction of each view, with their colours from images
    (S, 3, H, W) in [0, 1].  valid (S, H, W) bool: only these pixels are used (e.g. ~sky_mask(images))."""
    points, conf = np.asarray(points), np.asarray(conf)
    rgb = (torch.as_tensor(images).permute(0, 2, 3, 1).clamp(0, 1) * 255).round().byte().numpy()
    xyz, col = [], []
    for v in range(len(points)):
        if valid is None:
            m = np.isfinite(points[v]).all(-1) & (conf[v] >= np.quantile(conf[v], 1 - keep))
        else:
            ok = np.isfinite(points[v]).all(-1) & valid[v]
            m = ok & (conf[v] >= np.quantile(conf[v][ok], 1 - keep)) if ok.any() else ok
        xyz.append(points[v][m])
        col.append(rgb[v][m])
    return np.concatenate(xyz), np.concatenate(col)
