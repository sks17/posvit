"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import posvit

from . import paths
from .config import load_config
from .evaluate import prediction_paths
from .hashing import sha256_file
from .registry import all_models

SCHEMA = "posvit.manifest/1"


# Provenance collection

def _git(*args: str) -> "str | None":
    """
    Runs a git command in the repository root.
    - Pre: `args` contains valid git command arguments.
    - Post: Returns stripped stdout when the command succeeds.
        Returns None when git is unavailable or the command fails.
    """
    try:
        r = subprocess.run(
            ["git", *args], cwd=paths.repo_root(),
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_provenance() -> "dict[str, Any]":
    """
    Collects git provenance details.
    - Pre: Repository may or may not be a git repository.
    - Post: Returns provenance dictionary with availability, commit, branch, and dirty flag.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    status = _git("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def env_provenance() -> "dict[str, Any]":
    """
    Collects runtime environment provenance details.
    - Pre: Runtime is active and project paths are available.
    - Post: Returns environment dictionary with python, platform, package version, and lockfile hash.
    """
    lock = paths.repo_root() / "env" / "requirements.lock"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "posvit_version": posvit.__version__,
        "lockfile_sha256": sha256_file(lock) if lock.is_file() else None,
    }


# Artifact collection

def _read_json(path) -> "dict[str, Any] | None":
    """
    Reads a JSON file safely.
    - Pre: `path` points to a JSON file path.
    - Post: Returns parsed JSON dictionary when available and valid.
        Returns None when file is missing or unreadable.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _status(report: "dict[str, Any] | None", key: str = "passed") -> str:
    """
    Converts report state to normalized status value.
    - Pre: `report` is None or a report dictionary.
        `key` is the boolean field name to check.
    - Post: Returns one of `absent`, `pass`, or `fail`.
    """
    if report is None:
        return "absent"
    return "pass" if report.get(key) else "fail"


def collect_models() -> "dict[str, Any]":
    """
    Collects per-model prediction and checkpoint receipts.
    - Pre: Model registry and prediction artifacts are accessible.
    - Post: Returns dictionary keyed by safe model id.
        Each item includes model id, checkpoint receipt, variant summaries, and sanity status.
    """
    out: dict[str, Any] = {}
    for spec in all_models():
        variants: dict[str, Any] = {}
        receipt = None
        for variant in ("original", "mixed_same", "mixed_rand", "mixed_next",
                        "only_fg", "only_bg_t"):
            meta = _read_json(prediction_paths(spec, variant)[1])
            if meta is None:
                continue
            variants[variant] = {
                "n_records": meta.get("n_records"),
                "accuracy": meta.get("accuracy"),
                "frac_pred_out_of_in9": meta.get("frac_pred_out_of_in9"),
                "complete": meta.get("complete", False),
            }
            receipt = receipt or meta.get("checkpoint")

        orig = variants.get("original")
        sanity = (
            {"status": "absent"} if orig is None
            else {
                "status": "pass" if (orig["accuracy"] or 0) >= 0.95 else "fail",
                "original_acc": orig["accuracy"],
                "threshold": 0.95,
            }
        )
        out[spec.safe_id] = {
            "model_id": spec.model_id,
            "checkpoint": receipt,
            "variants": variants,
            "sanity": sanity,
        }
    return out


def collect_manifest(cfg=None) -> "dict[str, Any]":
    """
    Collects all manifest fields from existing artifacts.
    - Pre: `cfg` is None or a valid config object.
    - Post: Returns a manifest dictionary with schema, provenance, artifact status, models, and metrics.
    """
    cfg = cfg or load_config()
    verify_rep = _read_json(paths.verify_report_path())
    masks_rep = _read_json(paths.results_raw() / "masks" / "_manifest.json")
    bggap = _read_json(paths.metrics_dir() / "bggap.json")

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_provenance(),
        "environment": env_provenance(),
        "config": cfg.to_dict(),
        "dataset": {
            "status": _status(verify_rep),
            "checks": {k: v.get("ok") for k, v in (verify_rep or {}).get("checks", {}).items()},
        },
        "masks": {
            "status": _status(masks_rep),
            "coverage": (masks_rep or {}).get("coverage"),
            "written": (masks_rep or {}).get("written"),
        },
        "models": collect_models(),
        "metrics": {"bggap": bggap if bggap is not None else {}},
    }


# Certification

def certify(manifest: "dict[str, Any]") -> "dict[str, Any]":
    """
    Evaluates manifest certification status.
    - Pre: `manifest` is a dictionary returned by `collect_manifest()`.
    - Post: Returns dictionary with `certified`, `blockers`, and `warnings`.
        Blockers prevent certification and warnings provide non-blocking concerns.
    """
    blockers: list[str] = []
    warnings: list[str] = []

    for name in ("dataset", "masks"):
        status = manifest[name]["status"]
        if status == "absent":
            blockers.append(f"{name}: no report on disk - the check was never run")
        elif status == "fail":
            blockers.append(f"{name}: report says failed")

    for safe_id, m in manifest["models"].items():
        ck = m.get("checkpoint")
        if ck is None:
            warnings.append(f"{safe_id}: no predictions, so no checkpoint receipt")
            continue
        if not ck.get("sha256_pinned"):
            blockers.append(
                f"{safe_id}: checkpoint hash was never pinned (sha256_pinned=false) - "
                "run `python -m posvit.checkpoints --record` and paste it into configs/models.yaml"
            )
        if m["sanity"]["status"] == "fail":
            blockers.append(f"{safe_id}: sanity gate FAILED (acc={m['sanity']['original_acc']})")
        elif m["sanity"]["status"] == "absent":
            warnings.append(f"{safe_id}: sanity gate never run")
        for variant, v in m["variants"].items():
            if not v["complete"]:
                blockers.append(f"{safe_id}/{variant}: prediction run is incomplete")

    for safe_id, rep in manifest["metrics"]["bggap"].items():
        if not rep.get("join_losses", {}).get("lossless", True):
            blockers.append(f"{safe_id}: BG-Gap join was lossy - pairing precondition broken")

    git = manifest["git"]
    if not git.get("available"):
        warnings.append("git provenance unavailable - cannot record which code ran")
    elif git.get("dirty"):
        warnings.append(
            "git working tree is dirty; the recorded commit does not describe this run"
        )
    if manifest["environment"]["lockfile_sha256"] is None:
        warnings.append("env/requirements.lock missing - the environment is unpinned")

    return {"certified": not blockers, "blockers": blockers, "warnings": warnings}


def write_manifest(manifest: "dict[str, Any]"):
    """
    Writes the manifest JSON file to disk.
    - Pre: `manifest` is a JSON-serializable dictionary.
    - Post: Writes sorted manifest JSON with trailing newline.
        Returns the output file path.
    """
    out = paths.manifest_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    man = collect_manifest()
    man["certification"] = certify(man)

    c = man["certification"]
    print(f"dataset: {man['dataset']['status']}   masks: {man['masks']['status']}")
    for safe_id, m in man["models"].items():
        pinned = (m.get("checkpoint") or {}).get("sha256_pinned")
        print(f"  {safe_id:22s} sanity={m['sanity']['status']:6s} "
              f"pinned={pinned}  variants={len(m['variants'])}")
    for w in c["warnings"]:
        print(f"WARN     {w}")
    for b in c["blockers"]:
        print(f"BLOCKER  {b}")

    print(f"\nWrote {write_manifest(man)}")
    print("CERTIFIED" if c["certified"] else "NOT CERTIFIED")
    sys.exit(0 if c["certified"] else 1)