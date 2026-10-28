"""Run GenRecon on the prepared pilot, then export a GLB in reference axes.

Run this with the separate `genrecon` GPU environment. Authentication is
checked before downloading checkpoints. Existing model files are reused and
interrupted checkpoint downloads resume. Inference is always rerun, so stale
outputs are never treated as evidence of a successful new run.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
import socket

from .download_checkpoint import download_checkpoint


ROOT = Path(__file__).resolve().parents[3]
GENRECON = ROOT / "papers/GenRecon"
DINO = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def run(command, cwd, log_path, environment):
    began = time.monotonic()
    memory_samples = []
    stopped = threading.Event()
    with log_path.open("w") as log:
        process = subprocess.Popen([str(x) for x in command], cwd=cwd, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        def monitor():
            while not stopped.is_set():
                try:
                    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                    for row in result.stdout.splitlines():
                        pid, mib = row.split(',')
                        if int(pid.strip()) == process.pid:
                            memory_samples.append(float(mib.strip()))
                except (ValueError, OSError, subprocess.TimeoutExpired):
                    pass
                stopped.wait(2)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
        stopped.set()
        thread.join(timeout=6)
        telemetry = {'elapsed_seconds': time.monotonic()-began, 'returncode':status,
            'sampled_peak_process_gpu_mib': max(memory_samples) if memory_samples else None,
            'gpu_memory_sample_count':len(memory_samples),
            'gpu_memory_method':'nvidia-smi own subprocess PID, sampled every 2 s; may miss brief peaks; shared GPU'}
        log_path.with_suffix('.telemetry.json').write_text(json.dumps(telemetry,indent=2)+'\n')
        if status:
            raise RuntimeError(f"Command failed ({status}); see {log_path}")
    return telemetry


def main(data, output, seed, skip_remesh=False, simplify_threshold=2_000_000, texture_size=4096, inference_only=False):
    if (output / "scene.glb").exists() or (output / "scene_reference_frame.glb").exists():
        raise SystemExit("A reconstruction already exists here; choose a new --output for another run.")
    prefix = Path(sys.prefix)
    print('Checking encoder access and CUDA environment',flush=True)
    if (prefix / "bin/nvcc").exists():
        os.environ.setdefault("CUDA_HOME", str(prefix))
        os.environ["PATH"] = str(prefix / "bin") + ":" + os.environ["PATH"]
        include = str(prefix / "targets/x86_64-linux/include")
        os.environ["CPATH"] = include + (":" + os.environ["CPATH"] if os.environ.get("CPATH") else "")
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(DINO, "config.json")
    except Exception as error:
        raise SystemExit(
            f"Cannot access required {DINO} ({type(error).__name__}). "
            "Request model access, then run `hf auth login` in the genrecon environment."
        ) from None
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA-capable GenRecon environment is required.")
    for module in ("o_voxel", "cumesh", "flex_gemm", "nvdiffrast.torch", "nvdiffrec_render", "flash_attn"):
        importlib.import_module(module)
    inputs = json.loads((data / "input_manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"ATTN_BACKEND": "flash_attn", "SPARSE_ATTN_BACKEND": "flash_attn",
                        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1", "PYTHONUNBUFFERED": "1"})
    stages = {"sparse_structure": "ss_flow_img/genrecon.json",
              "shape_slat": "slat_flow_img2shape/genrecon_512.json",
              "texture_slat": "slat_flow_imgshape2tex/genrecon_512.json"}
    paths = {}
    for stage, config in stages.items():
        folder = GENRECON / "checkpoints" / stage
        (folder / "ckpts").mkdir(parents=True, exist_ok=True)
        path = folder / "ckpts" / f"{stage}.pt"
        config_path = folder / "config.json"
        if not config_path.exists():
            config_path.symlink_to(GENRECON / "configs/gen" / config)
        paths[stage] = path
    def fetch_stage(stage):
        download_checkpoint(f"https://kaldir.vc.cit.tum.de/genrecon/{stage}.pt", paths[stage])
    with ThreadPoolExecutor(max_workers=2) as downloads:
        for _ in downloads.map(fetch_stage, stages):
            pass
    checkpoint_records = {}
    for stage, path in paths.items():
        print(f'Hashing cached checkpoint: {stage}',flush=True)
        with path.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        checkpoint_records[stage] = {"path": str(path), "bytes": path.stat().st_size, "sha256": checksum}
    command = [sys.executable, "reconstruct_scene.py", "--mode", "Scannet_iphone",
               "--path", data, "--output_path", output,
               "--ss_ckpt", paths["sparse_structure"], "--shape_ckpt", paths["shape_slat"],
               "--tex_ckpt", paths["texture_slat"], "--pipeline_config", "configs/pipelines/original.json",
               "--num_imgs_per_scene", len(inputs["views"]), "--seed", seed,
               "--min_points_per_chunk", "30", "--skip_point_cleaning", "--save_imgs",
               "--proj_batch_voxels", "2048"]
    (output / "run_manifest.json").write_text(json.dumps({
        "status": "starting", "input": inputs, "command": [str(x) for x in command],
        "checkpoints": checkpoint_records,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "host":socket.gethostname(),
        "metric_scale_verified": False}, indent=2) + "\n")
    try:
        inference_telemetry = run(command, GENRECON, output / "reconstruction.log", environment)
    except Exception as error:
        failure_path=output/'run_manifest.json'
        failed=json.loads(failure_path.read_text())
        failed.update({'status':'inference_failed','inference_completed':False,'error':str(error)})
        failure_path.write_text(json.dumps(failed,indent=2)+'\n')
        raise
    report_path = output / "run_manifest.json"
    report = json.loads(report_path.read_text())
    report.update({'status':'inference_complete','inference_completed':True,
                   'inference_telemetry':inference_telemetry})
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    if inference_only:
        print(f'Raw reconstruction saved: {output / "mesh.ply"}',flush=True)
        return
    # Use a fresh chunk cache for each export invocation.
    export_command = [sys.executable, "chunked_to_glb.py", "--inputs", output / "to_glb_inputs.pt",
         "--chunk_inputs", output / "chunk_inputs.pt", "--output_dir", output,
         "--no_chunk_cache", "--simplify_threshold", simplify_threshold,
         "--texture_size", texture_size]
    if skip_remesh:
        export_command.append("--skip_remesh")
    run(export_command, GENRECON, output / "export.log", environment)
    path = output / "scene.glb"
    if not path.exists():
        raise RuntimeError("GLB conversion returned without writing scene.glb; inspect export.log.")
    import trimesh
    scene = trimesh.load_scene(path, process=False)
    # o_voxel.postprocess.to_glb already maps z-up to glTF y-up.
    # Applying that rotation again would misalign it with the reference.
    destination = output / "scene_reference_frame.glb"
    scene.export(destination, file_type="glb")
    report_path = output / "run_manifest.json"
    report = json.loads(report_path.read_text())
    report.update({"status": "complete", "inspection_glb": str(destination),
                   "export_options": {"skip_remesh": skip_remesh,
                                      "simplify_threshold_per_chunk": simplify_threshold,
                                      "texture_size": texture_size},
                   "export_command": [str(x) for x in export_command]})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Reconstruction in reference glTF axes: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/3dfront/genrecon_pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/genrecon/3dfront_pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-remesh", action="store_true",
                        help="Skip global remeshing during GLB export; useful when that stage exceeds VRAM.")
    parser.add_argument("--simplify-threshold", type=int, default=2_000_000)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument('--inference-only', action='store_true', help='Save raw geometry and export inputs; defer textured GLB conversion.')
    args = parser.parse_args()
    main(args.data.resolve(), args.output.resolve(), args.seed, args.skip_remesh,
         args.simplify_threshold, args.texture_size, args.inference_only)
