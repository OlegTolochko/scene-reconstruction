"""config.json of a run: what was run, on what, with which settings.

A run folder is named by experiment, tag, dataset and view count. Its config.json holds the rest, and a run
refuses to continue when the config it is started with differs from the one it was started with before.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSION_FIELDS = ("date", "git_commit")  # may differ between sessions of the same run


def git_commit() -> str:
    """The checked-out commit, with "-dirty" if tracked files have uncommitted changes."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT).returncode != 0
    return commit + ("-dirty" if dirty else "")


def build(**fields) -> dict:
    return {**fields, "date": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit()}


def differences(old: dict, new: dict, prefix: str = "") -> list[str]:
    """What changed, as "key: old -> new", with nested dicts (the settings) compared key by key."""
    out = []
    for k in sorted((set(old) | set(new)) - (set(SESSION_FIELDS) if not prefix else set())):
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            out += differences(a, b, f"{prefix}{k}.")
        elif a != b:
            out.append(f"{prefix}{k}: {a!r} -> {b!r}")
    return out


def check_or_write(path: Path, config: dict):
    """Write config.json for a new run, or check that an existing run was started with the same config."""
    if path.exists():
        diffs = differences(json.loads(path.read_text()), config)
        if diffs:
            raise SystemExit(f"{path} was started with a different config:\n  " + "\n  ".join(diffs)
                             + "\nUse another --tag, or delete the run folder to start over.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
