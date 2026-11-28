<h2 align="center">Solv3R: 3D Looped Transformer as a Multi-view Solver</h2>

<blockquote>
  <p>
    <a href="https://jgkwak95.github.io/">Jeong-gi Kwak</a> · <a href="https://theialab.ca/">Andrea Tagliasacchi</a> · Sho Kagami · Yuki Ono · <a href="https://www.cs.ubc.ca/~kmyi/">Kwang Moo Yi</a><br>
    <sub>University of British Columbia · Wayve Labs · Simon Fraser University<br>
    University of Toronto · Sony Semiconductor Solutions Corporation · Sony Corporation</sub>
  </p>
</blockquote>

<p align="center">
  <a href="https://arxiv.org/abs/2611.02857"><img src="https://img.shields.io/badge/arXiv-2611.02857-b31b1b.svg" alt="arXiv"></a>
  <a href="https://ubc-vision.github.io/solv3r/"><img src="https://img.shields.io/badge/Project-Page-1f72b1.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/jgkwak/solv3r"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/papers/2611.02857"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-orange" alt="Hugging Face Paper"></a>
</p>

[![One Solv3R solver for offline, streaming, and any-order reconstruction](assets/teaser.gif)](https://ubc-vision.github.io/solv3r/)

## News

- **[2026-11-05]** Inference and evaluation code and the pretrained weights are released.
- **[2026-11-02]** [Solv3R][paper] is now available on arXiv! Code and models will be released within a week.

[paper]: https://arxiv.org/abs/2611.02857

## TODO

- [x] Release inference code and examples.
- [x] Release pretrained models.
- [x] Release evaluation code for the offline benchmark (Table 1).
- [x] Release evaluation code for the streaming benchmarks (Tables 3 and 8).
- [x] Release evaluation code for map-based relocalization (Table 2).
- [ ] Release solver property examples.
- [ ] Release demo.

## Installation

```bash
git clone https://github.com/ubc-vision/solv3r.git && cd solv3r
conda create -n solv3r python=3.11 -y && conda activate solv3r
pip install -e .
```

The pretrained model is on [🤗 Hugging Face](https://huggingface.co/jgkwak/solv3r).
`Solv3R.from_pretrained()` downloads it to the Hugging Face cache on first use, or uses `checkpoints/solv3r.safetensors` if you place it there.

## Quick start

```python
from solv3r import Solv3R, cameras_from_pose_enc, list_images, load_images

model = Solv3R.from_pretrained()                       # the released weights
images = load_images(list_images("path/to/images"))   # long side 504, sides multiples of 14, no crop
ctx = model.encode(images)                             # the first image defines the world frame
z = model.solve(ctx, calls=16)                         # 16 calls of the looped block
pred = model.decode(z, ctx)                            # depth, depth_conf, rays, world_points, pose_enc
c2w, K = cameras_from_pose_enc(pred["pose_enc"], (ctx.H, ctx.W))
```

The state `z` can be used again at any point:

```python
z = model.solve(ctx, calls=8, z=z)                     # continue solving
new_images = load_images(list_images("path/to/more_images"), size=(ctx.H, ctx.W))
z, ctx = model.add_views(z, ctx, new_images)           # new views join the ongoing solve
z = model.solve(ctx, calls=8, z=z)                     # all views are updated, the old ones too
```

## Examples

<p align="center">
  <img src="assets/examples/offline_park.webp" width="49%" alt="Offline reconstruction of 39 drone photos of a park">
  <img src="assets/examples/streaming_desk.webp" width="49%" alt="Streaming reconstruction of a desk video">
</p>

Each script takes a folder of images or a video, and writes the cameras and depth maps (`.npz`), a point cloud (`.ply`)
and a viewer that opens in the browser (`.html`).

```bash
python examples/offline.py     --images path/to/images --out outputs/offline      # all images at once
python examples/incremental.py --images path/to/images --out outputs/incremental  # groups of images join the solve
python examples/any_order.py   --images path/to/images --out outputs/any_order    # groups join in a random order
python examples/streaming.py   --frames path/to/video.mp4 --out outputs/streaming # video frames, one at a time
```

Or try them on the samples in `examples/data`, the inputs of the animations above:

```bash
python examples/offline.py   --images examples/data/park --out outputs/offline --sky-mask
python examples/streaming.py --frames examples/data/desk --out outputs/streaming
```

`--adaptive 0.02` stops the offline solve once the state settles, `--config L4` or `S4` selects a lighter streaming
setting from paper Table 8, and `--sky-mask` leaves the sky out of outdoor point clouds.
A result can also be rendered as a short video (`--aerial` for drone photos):

```bash
python examples/render_video.py --input outputs/offline/solve --out outputs/offline/solve.mp4 --aerial
```

## Evaluation

The offline and streaming benchmarks follow existing protocols without changes: the offline benchmark of DVLT and
the streaming camera-pose evaluation of CUT3R / TTT3R.

### Offline benchmark

DTU, ETH3D, 7-Scenes, ScanNet++ and nuScenes, prepared with the scripts of [DVLT](https://github.com/nv-tlabs/dvlt)
as `<data-root>/test/<dataset>`.

```bash
pip install -e ".[eval]"   # the offline benchmark runs inside DVLT's evaluation pipeline
python eval/offline.py --data-root /path/to/benchmark --out results/offline                    # 16 calls
python eval/offline.py --data-root /path/to/benchmark --out results/offline --adaptive 0.02   # adaptive
python eval/offline.py --data-root /path/to/benchmark --out results/offline --datasets eth3d
```

Expected results (Table 1 of the paper):

<table>
  <thead>
    <tr><th rowspan="2">Dataset</th><th colspan="4">16 calls</th><th colspan="4">adaptive (0.02)</th></tr>
    <tr><th>AUC@3 ↑</th><th>AUC@30 ↑</th><th>Rel. L2 ↓</th><th>Inlier ↑</th><th>AUC@3 ↑</th><th>AUC@30 ↑</th><th>Rel. L2 ↓</th><th>Inlier ↑</th></tr>
  </thead>
  <tbody>
    <tr><td>DTU</td><td>84.5</td><td>99.0</td><td>.010</td><td>96.9</td><td>85.5</td><td>99.1</td><td>.010</td><td>97.0</td></tr>
    <tr><td>ETH3D</td><td>65.4</td><td>94.8</td><td>.029</td><td>74.3</td><td>70.4</td><td>96.3</td><td>.027</td><td>77.9</td></tr>
    <tr><td>7-Scenes</td><td>13.8</td><td>81.8</td><td>.035</td><td>74.4</td><td>14.2</td><td>81.9</td><td>.034</td><td>75.8</td></tr>
    <tr><td>nuScenes</td><td>41.5</td><td>85.3</td><td>.059</td><td>59.9</td><td>41.5</td><td>85.3</td><td>.059</td><td>60.4</td></tr>
    <tr><td>ScanNet++</td><td>80.7</td><td>98.2</td><td>.016</td><td>92.5</td><td>81.2</td><td>98.2</td><td>.016</td><td>92.6</td></tr>
  </tbody>
</table>

AUC@3 and AUC@30 are camera pose accuracy, Rel. L2 and Inlier are point-map error and inlier ratio (%).

### Streaming camera poses

Sintel, TUM-dynamics and ScanNet as in [CUT3R](https://github.com/CUT3R/CUT3R) and
[TTT3R](https://github.com/Inception3D/TTT3R), prepared with TTT3R `datasets_preprocess/long_prepare_{tum,scannet}.py`:

```
<data-root>/sintel/training/{final,camdata_left}/<seq>
<data-root>/long_tum_s1/<seq>/{rgb_1000, groundtruth_1000.txt}
<data-root>/long_scannet_s3/<scene>/{color_1000, pose_1000.txt}
```

```bash
python eval/streaming.py --data-root /path/to/data --task all --config L8 --out results/streaming
```

Expected results (Tables 3 and 8 of the paper):

<table>
  <thead>
    <tr><th rowspan="2">Config</th><th colspan="3">ATE ↓</th></tr>
    <tr><th>Sintel</th><th>TUM-dynamics</th><th>ScanNet</th></tr>
  </thead>
  <tbody>
    <tr><td>L8 (default)</td><td>0.157</td><td>0.015</td><td>0.077</td></tr>
    <tr><td>L4</td><td>0.163</td><td>0.017</td><td>0.091</td></tr>
    <tr><td>S4</td><td>0.179</td><td>0.017</td><td>0.180</td></tr>
  </tbody>
</table>

### Map-based relocalization

[7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and
[Cambridge Landmarks](https://www.repository.cam.ac.uk/handle/1810/251342) as released:

```
<data-root>/7scenes/<scene>/{TrainSplit.txt, TestSplit.txt, seq-XX/frame-XXXXXX.{color.png, pose.txt}}
<data-root>/cambridge/<Scene>/{dataset_train.txt, dataset_test.txt, seqN/frameXXXXX.png}
```

```bash
python eval/relocalization.py --data-root /path/to/data --out results/relocalization
```

Expected results (Table 2 of the paper):

<table>
  <thead>
    <tr><th rowspan="2">Cache (MB / view)</th><th colspan="3">7-Scenes</th><th colspan="3">Cambridge</th></tr>
    <tr><th>Trans. (cm) ↓</th><th>Rot. (°) ↓</th><th>5cm, 5° ↑</th><th>Trans. (cm) ↓</th><th>Rot. (°) ↓</th><th>25cm, 2° ↑</th></tr>
  </thead>
  <tbody>
    <tr><td>2.86</td><td>3.58</td><td>0.95</td><td>72.8</td><td>63.5</td><td>0.74</td><td>24.9</td></tr>
  </tbody>
</table>

Median translation and rotation errors and localization recall (%), averaged over scenes.

## Solver properties

<p align="center">
  <a href="https://ubc-vision.github.io/solv3r/#subsets"><img src="assets/examples/ballroom_subsets.jpg" width="70%" alt="128 photos of a ballroom solved 8 random photos at a time"></a>
</p>

Examples of the solver properties (stability over deep calls, correction, continuation and stochastic subset updates)
are coming soon! The image above links to the video on the project page.

## Citation

```bibtex
@article{kwak2026solv3r,
  title   = {Solv3R: 3D Looped Transformer as a Multi-view Solver},
  author  = {Kwak, Jeong-gi and Tagliasacchi, Andrea and Kagami, Sho and Ono, Yuki and Yi, Kwang Moo},
  journal = {arXiv preprint arXiv:2611.02857},
  year    = {2026}
}
```

### License

The code is released under the [Apache License 2.0](LICENSE).
The model weights are built on [DVLT](https://github.com/nv-tlabs/dvlt) and are released under the
[NVIDIA License](LICENSES/NVIDIA-LICENSE.txt) (non-commercial, research and evaluation use only).
