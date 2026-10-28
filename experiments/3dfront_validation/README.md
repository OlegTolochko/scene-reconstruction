# 3D-FRONT validation and reconstruction

Run from the repository root. Install the [environments](environment/README.md)
and obtain access to `facebook/dinov3-vitl16-pretrain-lvd1689m` before inference.

## Pilot

```bash
conda activate 3dfront-validation
python data_download/3dfront/download_rooms.py --pilot
python -m experiments.3dfront_validation.preparation.validate_room
python -m experiments.3dfront_validation.visualization.export_inspection
python -m experiments.3dfront_validation.preparation.prepare_genrecon_input

conda activate genrecon
hf auth login
python -m experiments.3dfront_validation.inference.run_genrecon
```

Outputs: `outputs/3dfront_validation/` (validation and reference exports),
`data/3dfront/genrecon_pilot/` (model inputs), and
`outputs/genrecon/3dfront_pilot/` (reconstruction and logs).
Use `run_genrecon --data <input-dir> --output <new-run-dir>` to override paths.

## Benchmark

Activate `genrecon`. The selection and settings are recorded in
[`configs/BENCHMARK_PROTOCOL.json`](configs/BENCHMARK_PROTOCOL.json).

```bash
python -m experiments.3dfront_validation.preparation.prepare_benchmark
python - <<'PY'
import json
from importlib import import_module
from pathlib import Path

prepare = import_module('experiments.3dfront_validation.preparation.prepare_genrecon_input').main
protocol = json.loads(Path('data/3dfront/benchmark/protocol.json').read_text())
for scene in protocol['scenes']:
    data, output = Path(scene['data']), Path(scene['input'])
    if not (output / 'input_manifest.json').exists():
        prepare(data, output, count=8, seed=42)
PY
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m experiments.3dfront_validation.preparation.validate_benchmark_inputs
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m experiments.3dfront_validation.evaluation.test_evaluation
python -m experiments.3dfront_validation.inference.run_benchmark
python -m experiments.3dfront_validation.visualization.export_benchmark_comparisons
```

Outputs: `outputs/3dfront_benchmark/`; compact reports and diagnostics:
`reports/benchmark/`. Successful inference stages are reused; inspect each
scene's `progress.json` and logs for failures. Pass `--no-export` to skip textured
previews. Regenerate reports with:

```bash
python -m experiments.3dfront_validation.evaluation.summarize_benchmark
```

[Pilot report](reports/pilot/RESULTS.md) ·
[Pilot inference report](reports/pilot/GENRECON_STATUS.md) ·
[Benchmark results](reports/benchmark/BENCHMARK_RESULTS.md) ·
[Benchmark findings](reports/benchmark/BENCHMARK_FINDINGS.md).
