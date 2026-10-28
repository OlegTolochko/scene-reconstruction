# Settings

There is no configuration-file loader yet. Edit these locations before running:

| Setting | Location |
| --- | --- |
| Scene order and sanity-check selection | `reconstruction/run_baseline.py` |
| Output root, resolution, shape checkpoints | `common.py` |
| Reference paths, source frame, chunk size | `preparation/prepare_scene.py` |
| Scene subset metrics | `evaluation/evaluate.py` |
| PBR input paths, checkpoints, texture size | `reconstruction/run_pbr_baseline.py` |

Verification, diagnostics and report generation currently assume the same
3D-FRONT selection; update them together when adding another dataset. Use a fresh
output directory and record dataset revision and source scene IDs in its report.
