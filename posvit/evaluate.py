"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
Run a verified model over one IN-9 variant and log per-image predictions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import torch
from . import paths
from .registry import ModelSpec
from .data import load_in9_map
from .data.loaders import build_loader, parse_fg_id
SCHEMA = "posvit.predictions/1"


def resolve_device(prefer: "str | None" = None) -> str:
    """
    Resolves the torch device string.
    - Pre: `prefer` is a string or None.
    - Post: Returns a device string for `torch.device()`.
        If `prefer` is None, it returns `cuda` when available, else `cpu`.
    """
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def prediction_paths(spec: ModelSpec, variant: str) -> "tuple[Path, Path]":
    """
    Builds output paths for one model and variant.
    - Pre: `spec` is a ModelSpec object and `variant` is a non-empty string.
    - Post: Returns `(records_path, sidecar_path)` as Path objects.
    """
    stem = f"{spec.safe_id}__{variant}"
    d = paths.predictions_dir()
    return d / f"{stem}.jsonl", d / f"{stem}.meta.json"


def read_records(path: "Path | str") -> "list[dict[str, Any]]":
    """
    Reads prediction records from a JSONL file.
    - Pre: `path` points to a UTF-8 JSONL file.
    - Post: Returns a list of record dictionaries.
        Blank lines are ignored.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_meta(path: "Path | str") -> "dict[str, Any]":
    """
    Reads a sidecar metadata file.
    - Pre: `path` points to a UTF-8 JSON file.
    - Post: Returns the parsed metadata dictionary.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_complete(spec: ModelSpec, variant: str, n_expected: "int | None" = None) -> bool:
    """
    Checks if evaluation output is complete.
    - Pre: `spec` is a ModelSpec object and `variant` is a non-empty string.
        `n_expected` is None or an integer.
    - Post: Returns True only when record and sidecar files exist,
        the sidecar parses, and `complete` is True.
        If `n_expected` is set, the record count must match it.
    """
    rec_path, meta_path = prediction_paths(spec, variant)
    if not (rec_path.is_file() and meta_path.is_file()):
        return False
    try:
        meta = read_meta(meta_path)
    except (json.JSONDecodeError, OSError):
        return False
    if not meta.get("complete"):
        return False
    return n_expected is None or meta.get("n_records") == n_expected


@torch.no_grad()
def evaluate_variant(
    model,
    spec: ModelSpec,
    variant: str,
    cfg,
    *,
    in9_map: "dict[str, int] | None" = None,
    device: "str | None" = None,
    receipt: "dict[str, Any] | None" = None,
    limit: "int | None" = None,
    force: bool = False,
) -> "dict[str, Any]":
    """
    Evaluates one variant and writes output files.
    - Pre: `model` is a loaded model in eval mode.
        `spec` is a ModelSpec object.
        `variant` is a valid variant name.
        `cfg` has valid data loader settings.
    - Post: Returns the sidecar metadata dictionary.
        Writes records to a JSONL file and metadata to a JSON file.
    """
    device = resolve_device(device)
    in9_map = in9_map if in9_map is not None else load_in9_map()

    loader, ds = build_loader(cfg, variant)
    n_expected = len(ds) if limit is None else min(limit, len(ds))

    if not force and is_complete(spec, variant, n_expected):
        _, meta_path = prediction_paths(spec, variant)
        return read_meta(meta_path)

    rec_path, meta_path = prediction_paths(spec, variant)
    rec_path.parent.mkdir(parents=True, exist_ok=True)

    recs: list[dict[str, Any]] = []
    n_out_of_in9 = 0  # Predicted class has no IN-9 mapping.
    n_fg_fallback = 0  # Filename did not match known FG-ID patterns.
    t0 = time.time()

    for imgs, targets, idxs in loader:
        top1 = model(imgs.to(device)).argmax(1).cpu().numpy()
        for p1000, t9, idx in zip(top1, targets.numpy(), idxs.numpy()):
            if limit is not None and len(recs) >= limit:
                break
            fname = Path(ds.samples[int(idx)][0]).name
            fg_id = parse_fg_id(fname)
            if fg_id is None:
                n_fg_fallback += 1
                fg_id = fname
            pred9 = in9_map.get(str(int(p1000)), -1)
            if pred9 == -1:
                n_out_of_in9 += 1
            recs.append({
                "fname": fname,
                "fg_id": fg_id,
                "label9": int(t9),
                "pred1000": int(p1000),
                "pred9": int(pred9),
            })
        if limit is not None and len(recs) >= limit:
            break

    elapsed = time.time() - t0
    n_correct = sum(1 for r in recs if r["pred9"] == r["label9"])

    with open(rec_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    meta = {
        "schema": SCHEMA,
        "records_file": rec_path.name,
        "model_id": spec.model_id,
        "safe_id": spec.safe_id,
        "variant": variant,
        "complete": True,
        "n_records": len(recs),
        "n_expected": n_expected,
        "limit": limit,
        "accuracy": round(n_correct / len(recs), 6) if recs else 0.0,
        "n_pred_out_of_in9": n_out_of_in9,
        "frac_pred_out_of_in9": round(n_out_of_in9 / len(recs), 6) if recs else 0.0,
        "n_fg_id_fallback": n_fg_fallback,
        "device": device,
        "elapsed_s": round(elapsed, 2),
        "checkpoint": receipt,
        "config": cfg.to_dict(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def sanity_check(
    model,
    spec: ModelSpec,
    cfg,
    *,
    in9_map: "dict[str, int] | None" = None,
    device: "str | None" = None,
    threshold: float = 0.95,
) -> "dict[str, Any]":
    """
    Runs a sanity check on the original variant.
    - Pre: `model` is a loaded model in eval mode.
        `spec` is a ModelSpec object.
        `cfg` has valid data loader settings.
        `threshold` is a float in [0, 1].
    - Post: Returns a dictionary with accuracy, threshold, and pass/fail status.
    """
    meta = evaluate_variant(
        model, spec, "original", cfg, in9_map=in9_map, device=device, receipt=None
    )
    return {
        "model_id": spec.model_id,
        "original_acc": meta["accuracy"],
        "threshold": threshold,
        "passed": meta["accuracy"] >= threshold,
        "frac_pred_out_of_in9": meta["frac_pred_out_of_in9"],
    }


if __name__ == "__main__":
    import argparse
    import sys

    from .checkpoints import load_checkpoint
    from .config import load_config
    from .registry import by_id, by_index
    from .seed import seed_everything

    ap = argparse.ArgumentParser(description="Evaluate one registry model over the variants.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, help="registry index, as supplied by SLURM_ARRAY_TASK_ID")
    g.add_argument("--id", type=str, help="model id, e.g. vit_s/ape")
    ap.add_argument("--variant", action="append", help="repeatable; default = plan_a + plan_b_extra")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-unpinned", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    seed_receipt = seed_everything(cfg.seed)
    spec = by_index(args.index) if args.index is not None else by_id(args.id)
    device = resolve_device(args.device)

    model, receipt = load_checkpoint(
        spec, img_size=cfg.img_size, device=device, allow_unpinned=args.allow_unpinned
    )
    receipt["seed"] = seed_receipt

    variants = args.variant or (
        list(cfg.plan_a) + [v for v in cfg.plan_b_extra if v not in cfg.plan_a]
    )
    for v in variants:
        m = evaluate_variant(
            model, spec, v, cfg, device=device, receipt=receipt,
            limit=args.limit, force=args.force,
        )
        print(
            f"{spec.safe_id:22s} {v:12s} n={m['n_records']:5d} acc={m['accuracy']:.4f} "
            f"out_of_in9={m['frac_pred_out_of_in9']:.4f} ({m['elapsed_s']}s)"
        )

    s = sanity_check(model, spec, cfg, device=device)
    print(f"G_SANITY {spec.model_id}: {s['original_acc']:.4f} "
          f"{'PASS' if s['passed'] else 'FAIL'} (>= {s['threshold']})")
    sys.exit(0 if s["passed"] else 1)