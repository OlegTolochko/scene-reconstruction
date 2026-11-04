# COLMAP

COLMAP structure from motion on the ScanNet++ evaluation inputs, as a pose baseline next to the
ScanNet++ poses (COLMAP on all DSLR images, aligned to the laser scan) and VGGT-Omega.

COLMAP gets the same N undistorted DSLR views as the other methods (`scripts/select_views_scannetpp.py`)
and none of their poses or intrinsics: one shared PINHOLE camera estimated by COLMAP, SIFT on the CPU,
exhaustive matching, incremental mapping. Uses `pycolmap` from the root env (`uv sync`).

## Run

```bash
uv run scripts/select_views_scannetpp.py 8 32 64 96
uv run experiments/colmap/run.py 8 32 64 96      # outputs/colmap/<N>views/<scene>/
```

On the cluster, split the scenes over jobs with `--shard I/K` (every K-th scene, starting at the I-th)
and pass the allocated CPUs, e.g. `--threads "$SLURM_CPUS_PER_TASK"`. Run `select_views_scannetpp.py`
once before submitting, not in each job: parallel jobs would write the same files.

Finished scenes are skipped, so resubmitting continues where a task stopped.

CPU SIFT needs about 1 GB of RAM per thread at the 1752 x 1168 input size, and by default COLMAP uses
every core of the machine. Pass `--threads 1` on a machine with little RAM. A 32-view scene takes
about one to two minutes with one thread.

## Outputs

```text
outputs/colmap/<N>views/<scene>/
├── images/          symlinks to the selected views
├── database.db      features and matches
├── sparse/<k>/      every model COLMAP built (binary)
├── colmap/          the model with the most registered images, as text
│   ├── cameras.txt
│   ├── images.txt   poses by original DSLR file name; COLMAP's arbitrary frame and scale
│   └── points3D.txt sparse points with tracks
└── summary.json     registered images (and which were not), points, reprojection error, run time
```

Scenes that already have `summary.json` are skipped; delete it to rerun a scene. Images COLMAP could
not register are missing from `images.txt` and listed in `summary.json`.
