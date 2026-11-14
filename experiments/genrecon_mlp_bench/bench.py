"""What the bench adds to GenRecon's dataset and trainer. GenRecon is not imported here (train.py combines these
mixins with GenRecon's classes).

  PairedItemsMixin   a dataset whose validation items repeat
  BenchTrainerMixin  a trainer with a repeatable validation loss, Weights & Biases logging, the comparison with
                     the baseline run of the same bench, and checkpoints that hold only the aggregator

Why a repeatable validation loss: GenRecon's draws new views for each chunk, new timesteps and new noise on every
pass and drops the conditioning of 10% of the samples. Two runs that differ by a few percent cannot be told apart
that way. Here every pass of every run evaluates the same (chunk, views, timestep, noise) tuples, so the
difference between two runs, or two steps of one run, is a paired difference.
"""

import functools
import json
import random
import uuid
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

BASELINE = "genrecon"  # the run folder (and variant) every other run of a bench is compared with
STEP = "training_step"  # the x axis of every W&B chart


def seed_all(seed: int, cuda: bool) -> None:
    seed = int(seed) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(seed)  # the CPU generator only, unlike torch.manual_seed
    if cuda:
        torch.cuda.manual_seed_all(seed)


@contextmanager
def fixed_rng(seed: int, cuda: bool):
    """Seeds Python's, NumPy's and torch's generators inside the block and restores their states after it.
    cuda=False leaves the CUDA generators alone, as a DataLoader worker has to."""
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if cuda else None
    seed_all(seed, cuda)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if cuda:
            torch.cuda.set_rng_state_all(cuda_state)


def named_seed(*parts) -> int:
    return zlib.crc32("/".join(str(p) for p in parts).encode())


def flatten(log: dict, prefix: str = "") -> dict:
    """{"mse": a, "bin_3": {"mse": b}} -> {"mse": a, "bin_3/mse": b}, as floats."""
    flat = {}
    for key, value in log.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}/"))
        else:
            flat[f"{prefix}{key}"] = float(value)
    return flat


def read_val(path) -> dict[int, dict]:
    """val.jsonl as step -> row. A resumed run can repeat steps: the last row of a step counts."""
    rows = {}
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row["step"])] = row
    return dict(sorted(rows.items()))


def percent(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def claim_run(output_dir, info: dict) -> None:
    """Writes run.json into a new run folder. An existing one must have been started by the same variant code:
    its checkpoints and curves belong to that code."""
    path = Path(output_dir) / "run.json"
    if path.exists() and json.loads(path.read_text()).get("code_hash") != info["code_hash"]:
        raise SystemExit(
            f"The code of variant {info['variant']!r} changed since {output_dir} was started, so that run cannot "
            "be continued. Register the changed class under a new name, or delete the folder to start it again."
        )
    path.write_text(json.dumps(info, indent=2))


def start_wandb(output_dir: Path, config: dict, code_dir=None):
    """The W&B run of a run folder (name = the folder, group = its parent, the bench), resumed if the folder
    already has one. Project and entity are WANDB_PROJECT and WANDB_ENTITY. Without a connection or a login the
    run is logged offline, to be uploaded later with `wandb sync`."""
    import wandb

    id_file = output_dir / "wandb_id.txt"
    run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
    kwargs = dict(name=output_dir.name, group=output_dir.parent.name, id=run_id, resume="allow", job_type="train",
                  dir=str(output_dir), config=config)
    try:
        run = wandb.init(**kwargs)
    except Exception as e:  # no connection from this node, or not logged in
        print(f"\nW&B: could not start an online run ({e}).\nLogging offline instead; upload it later from a node "
              f"with a connection: wandb sync {output_dir}/wandb/offline-run-*\n", flush=True)
        run = wandb.init(**kwargs, mode="offline")
    id_file.write_text(run_id)
    run.define_metric(STEP)
    run.define_metric("*", step_metric=STEP)
    if code_dir is not None:
        run.log_code(str(code_dir))
    return run


class WandbWriter:
    """Stands in for the trainer's TensorBoard writer and sends every scalar to W&B too, against the training
    step. GenRecon's trainer logs everything through that writer, so its curves reach W&B unchanged."""

    def __init__(self, writer, run):
        self._writer, self._run, self._last = writer, run, {}

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        self._writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        # GenRecon writes a validation value twice, when it is computed and again with that step's log interval.
        if self._last.get(tag) != global_step:
            self._last[tag] = global_step
            self._run.log({tag: float(scalar_value), STEP: int(global_step)})

    def __getattr__(self, name):
        return getattr(self._writer, name)


class PairedItemsMixin:
    """For a GenRecon dataset (before it in the MRO), whose items are random: it picks the number of views of a
    chunk and the views themselves anew on every access. After set_paired(seed, repeats), item i is always the
    same, and the dataset is `repeats` times as long: each repeat gives every chunk another fixed draw."""

    pair_seed = None
    pair_repeats = 1

    def set_paired(self, seed: int, repeats: int = 1) -> None:
        assert repeats >= 1, f"repeats must be >= 1, got {repeats}"
        self.pair_seed, self.pair_repeats = int(seed), int(repeats)

    def __len__(self):
        return super().__len__() * self.pair_repeats

    def __getitem__(self, index):
        if self.pair_seed is None:
            return super().__getitem__(index)
        repeat, base = divmod(int(index), super().__len__())
        name = self.instances[base][1]  # the chunk id, so the draw does not depend on the chunk's position
        with fixed_rng(named_seed(self.pair_seed, repeat, name), cuda=False):
            return super().__getitem__(base)


class BenchTrainerMixin:
    """For a GenRecon trainer (before it in the MRO). Arguments, all optional:

        val_seed     seed of the validation draws; the same in every run of a bench
        val_repeats  fixed draws (views, timestep, noise) per validation chunk
        seed         0 keeps GenRecon's seeding; any other value reseeds training (views, timesteps, noise, order)
        run_config   what W&B shows as the run's config
        code_dir     folder whose .py files are stored with the W&B run

    Validation is the conditional loss only (p_uncond = 0 while it runs): that is the part the aggregator
    changes. It also runs once before the first update. Each pass is appended to <output_dir>/val.jsonl as
    {"step", "samples", "mse", "bin_<k>/mse", ...}; bin k holds the timesteps in [k / 10, (k + 1) / 10).

    A run in a folder other than `genrecon` also logs vs_genrecon/val_mse_percent: its validation loss relative
    to the baseline run next to it, at the same step. Below 0 is better than the baseline.
    """

    saved_prefix = "aggregator."  # the parameters a checkpoint holds; everything that trains has to be among them

    def __init__(self, *args, val_seed: int = 0, val_repeats: int = 4, seed: int = 0, run_config=None,
                 code_dir=None, **kwargs):
        self.val_seed, self.train_seed = int(val_seed), int(seed)
        self._val_batch = None  # the index of the next validation batch while run_val_step runs
        if kwargs.get("val_dataset") is not None:
            kwargs["val_dataset"].set_paired(val_seed, val_repeats)
        if self.train_seed:
            seed_all(self.train_seed, cuda=torch.cuda.is_available())
        super().__init__(*args, **kwargs)
        if self.train_seed:
            self.data_sampler.seed = self.train_seed  # read when the first batch is drawn
        loose = [n for n, _ in getattr(self, "trainable_named_params", [])
                 if not n.split(".", 1)[1].startswith(self.saved_prefix)]
        assert not loose, f"parameters outside {self.saved_prefix} train and would not be saved: {loose[:5]}"

        out = Path(self.output_dir)
        self.val_path = out / "val.jsonl"
        self.baseline_dir = None if out.name == BASELINE else out.parent / BASELINE
        self._wandb = None
        if self.is_master:
            self._wandb = start_wandb(out, run_config or {}, code_dir)
            self.writer = WandbWriter(self.writer, self._wandb)

    # ---- repeatable validation ----

    def validation_losses(self, *args, **kwargs):
        if self._val_batch is not None:
            # Per batch, so a batch's timesteps and noise do not depend on what ran before it.
            seed_all(named_seed(self.val_seed, "batch", self._val_batch), cuda=torch.cuda.is_available())
            self._val_batch += 1
        return super().validation_losses(*args, **kwargs)

    def run_val_step(self):
        if self.val_dataloader is None:
            return None
        p_uncond = getattr(self, "p_uncond", 0.0)
        self.p_uncond, self._val_batch = 0.0, 0
        try:
            with fixed_rng(self.val_seed, cuda=torch.cuda.is_available()):  # training's random stream is untouched
                val_log = super().run_val_step()
        finally:
            self.p_uncond, self._val_batch = p_uncond, None
        if self.is_master:
            self._record_val(flatten(val_log))
        return val_log

    def _record_val(self, flat: dict) -> None:
        row = {"step": self.step, "samples": len(self.val_dataset), **flat}
        with open(self.val_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        so_far = [r["mse"] for s, r in read_val(self.val_path).items() if s <= self.step]
        self.writer.add_scalar("val/best_mse", min(so_far), self.step)
        baseline = read_val(self.baseline_dir / "val.jsonl") if self.baseline_dir else {}
        if self.step in baseline:
            self.writer.add_scalar("vs_genrecon/val_mse_percent", percent(flat["mse"], baseline[self.step]["mse"]),
                                   self.step)

    def run(self):
        if self.step == 0 and self.val_dataloader is not None:
            if self.is_master and self.val_path.exists():
                self.val_path.unlink()  # a fresh start: drop the rows of an attempt that never reached a checkpoint
            val_log = self.run_val_step()  # the released checkpoint, before any update
            if self.is_master:
                for key, value in flatten(val_log).items():
                    self.writer.add_scalar(f"val/{key}", value, 0)  # later passes are logged by GenRecon's loop
        super().run()
        if self.is_master:
            self._finish()

    def _finish(self) -> None:
        """summary.json and the W&B summary: the run's best validation loss, and how it compares with a finished
        baseline. DONE marks the run as complete for run.sbatch."""
        out, mine, summary = Path(self.output_dir), read_val(self.val_path), {}
        if mine:
            best = min(mine, key=lambda s: mine[s]["mse"])
            summary.update({"val/best_mse": mine[best]["mse"], "val/best_step": best})
            base = read_val(self.baseline_dir / "val.jsonl") if self.baseline_dir else {}
            if base and (self.baseline_dir / "DONE").exists():
                base_best = min(base, key=lambda s: base[s]["mse"])
                reached = next((s for s, r in mine.items() if s > 0 and r["mse"] <= base[base_best]["mse"]), None)
                summary.update({
                    "vs_genrecon/best_mse_percent": percent(mine[best]["mse"], base[base_best]["mse"]),
                    "vs_genrecon/genrecon_best_step": base_best,
                    "vs_genrecon/reaches_genrecon_best_at_step": reached,  # None: never
                })
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        (out / "DONE").write_text("")
        self._wandb.summary.update({k: v for k, v in summary.items() if v is not None})
        self._wandb.finish()

    # ---- checkpoints with the aggregator only ----
    # Nothing else trains, and the full model would be several GB per checkpoint.

    def _master_params_to_state_dicts(self, master_params):
        full = super()._master_params_to_state_dicts(master_params)
        return {name: {k: v for k, v in state.items() if k.startswith(self.saved_prefix)} for name, state in full.items()}

    def load(self, load_dir, step=0):
        models = list(self.models.values())
        for model in models:  # the rest of the model keeps the weights it was built with
            model.load_state_dict = functools.partial(model.load_state_dict, strict=False)
        try:
            super().load(load_dir, step)
        finally:
            for model in models:
                del model.load_state_dict
