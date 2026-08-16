"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import paths
from .intervene import _set_attn_scales, evaluate_batch, fingerprint, restore, snapshot
from .registry import ModelSpec

SCHEMA = "posvit.specificity/1"


@dataclass(frozen=True)
class Control:
    name: str
    kind: str
    spatial: bool
    description: str
    levels: "tuple[float, ...]"


CONTROLS = {
    c.name: c
    for c in (
        Control("patch_noise", "hook", False, "Gaussian noise on patch-embed output "
                "(feature fidelity)", (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8)),
        Control("channel_drop", "hook", False, "zero a fraction of feature CHANNELS at every "
                "position (feature content, no spatial loss)", (0.0, 0.2, 0.4, 0.6, 0.8)),
        Control("token_drop", "hook", True, "zero a fraction of patch TOKENS (whole positions)",
                (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)),
        Control("attn_temp", "weight", True, "flatten every block's softmax temperature "
                "(routing/selectivity)", (1.0, 0.8, 0.6, 0.4, 0.2, 0.1)),
    )
}


def _seed_for(safe_id: str, control: str, strength: float) -> int:
    """
    Builds a stable seed for one control level.
    - Pre: `safe_id` and `control` are non-empty strings.
        `strength` is a numeric value.
    - Post: Returns deterministic integer seed for the input tuple.
    """
    key = f"{safe_id}|{control}|{strength:.6f}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


@contextmanager
def applied(model, snap, control: Control, strength: float, *, safe_id: str, device: str):
    """
    Applies one control inside a managed context.
    - Pre: `model` is a loaded model object.
        `snap` is a valid snapshot object.
        `control` is a valid Control object.
        `strength` is a numeric level value.
        `safe_id` is a non-empty string.
        `device` is a valid torch device string.
    - Post: Yields the model with control applied.
        Always removes hooks and restores state when the context exits.
        Raises RuntimeError when post-restore fingerprint check fails.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(_seed_for(safe_id, control.name, strength))
    handle = None
    try:
        if strength == 0.0 and control.kind == "hook":
            pass
        elif control.kind == "hook":
            def hook(_mod, _inp, out):
                if control.name == "patch_noise":
                    noise = torch.randn(out.shape, generator=gen, device=out.device, dtype=out.dtype)
                    return out + strength * noise
                if control.name == "token_drop":
                    shape = (out.shape[0], out.shape[1], 1)
                else:
                    shape = (out.shape[0], 1, out.shape[2])
                keep = (torch.rand(shape, generator=gen, device=out.device) > strength)
                return out * keep.to(out.dtype)
            handle = model.patch_embed.register_forward_hook(hook)
        else:
            _set_attn_scales(model, [s * strength for s in snap.attn_scales])
        yield model
    finally:
        if handle is not None:
            handle.remove()
        restore(model, snap)
        if fingerprint(model) != snap.fingerprint:
            raise RuntimeError(
                f"{control.name} at strength {strength}: cleanup did not restore the model, "
                "so any result computed after this point is contaminated."
            )


def sweep_control(model, spec: ModelSpec, cfg, control: Control, *, data, in9_map,
                  device: str, batch: int = 256) -> "list[dict[str, Any]]":
    """
    Evaluates one full control-strength curve.
    - Pre: `model` is a loaded model object.
        `spec` is a valid ModelSpec object.
        `cfg` has valid settings.
        `control` is a valid Control object.
        `data` contains preloaded variant tensors.
        `in9_map` is a valid ImageNet-to-IN9 mapping.
        `device` is a valid torch device string.
        `batch` is a positive integer.
    - Post: Returns list of per-strength result dictionaries.
    """
    snap = snapshot(model, spec)
    rows = []
    for s in control.levels:
        with applied(model, snap, control, s, safe_id=spec.safe_id, device=device):
            oc, oconf = evaluate_batch(model, *data["original"][:2], in9_map, cfg, device, batch)
            ms, _ = evaluate_batch(model, *data["mixed_same"][:2], in9_map, cfg, device, batch)
            mr, _ = evaluate_batch(model, *data["mixed_rand"][:2], in9_map, cfg, device, batch)
        rows.append({
            "strength": float(s),
            "original_acc": round(float(oc.mean()), 6),
            "original_conf": round(float(oconf.mean()), 6),
            "bg_gap": round(float(ms.mean() - mr.mean()), 6),
            "seed": _seed_for(spec.safe_id, control.name, s),
        })
    return rows


def monotonicity_report(strengths, values, *, n: int, tol: "float | None" = None) -> "dict[str, Any]":
    """
    Reports monotonicity of a degradation curve.
    - Pre: `strengths` and `values` are numeric sequences of equal length.
        `n` is a non-negative integer image count.
        `tol` is None or a non-negative float tolerance.
    - Post: Returns dictionary with tolerance and back-step diagnostics.
    """
    if tol is None:
        tol = 2.0 * float(np.sqrt(0.9 * 0.1 / max(n, 1)))
    order = np.argsort(strengths)[::-1] if strengths[0] >= strengths[-1] else np.argsort(strengths)
    v = np.asarray(values, float)[order]
    steps = np.diff(v)
    back = steps[steps > 0]
    return {
        "tolerance": round(tol, 6),
        "n_backsteps": int(len(back)),
        "max_backstep": round(float(back.max()), 6) if len(back) else 0.0,
        "exceeds_tolerance": bool(len(back) and back.max() > tol),
    }


def match_curves(pos_rows, ctrl_rows, *, axis: str = "original_acc", n_grid: int = 24,
                 n_images: int = 4050) -> "dict[str, Any]":
    """
    Matches positional and control curves on a common axis.
    - Pre: `pos_rows` and `ctrl_rows` are lists of row dictionaries.
        `axis` is a valid row key name.
        `n_grid` is a positive integer.
        `n_images` is a non-negative integer.
    - Post: Returns matched-curve summary dictionary with overlap,
        excess statistics, and monotonicity diagnostics.
    """

    def prepare(rows):
        """
        Prepares sorted curve arrays and monotonicity report.
        - Pre: `rows` is a list of row dictionaries.
        - Post: Returns `(x_sorted, y_sorted, monotonicity_report_dict)`.
        """
        x = np.array([r[axis] for r in rows], float)
        y = np.array([r["bg_gap"] for r in rows], float)
        s = np.array([r["strength"] for r in rows], float)
        mono = monotonicity_report(s, x, n=n_images)
        o = np.argsort(x)
        return x[o], y[o], mono

    xp, yp, mono_p = prepare(pos_rows)
    xc, yc, mono_c = prepare(ctrl_rows)

    lo, hi = max(xp.min(), xc.min()), min(xp.max(), xc.max())
    if not (hi > lo):
        return {"comparable": False, "reason": "no overlapping range",
                "monotonicity": {"positional": mono_p, "control": mono_c}}
    grid = np.linspace(lo, hi, n_grid)
    excess = np.interp(grid, xp, yp) - np.interp(grid, xc, yc)

    return {
        "comparable": True,
        "axis": axis,
        "overlap_lo": round(float(lo), 6), "overlap_hi": round(float(hi), 6),
        "excess_mean_pp": round(float(excess.mean()) * 100, 4),
        "excess_min_pp": round(float(excess.min()) * 100, 4),
        "excess_max_pp": round(float(excess.max()) * 100, 4),
        "monotonicity": {"positional": mono_p, "control": mono_c},
        "ambiguous": bool(mono_p["exceeds_tolerance"] or mono_c["exceeds_tolerance"]),
    }


def specificity_report(spec: ModelSpec, cfg, pos_rows, ctrl_rows_by_name,
                       *, n_images: int = 4050) -> "dict[str, Any]":
    """
    Builds full specificity report across all controls.
    - Pre: `spec` is a valid ModelSpec object.
        `cfg` has valid settings.
        `pos_rows` is positional sweep row list.
        `ctrl_rows_by_name` maps control names to sweep rows.
        `n_images` is a non-negative integer.
    - Post: Returns report dictionary with matched accuracy and confidence comparisons
        for each control.
    """
    out: dict[str, Any] = {
        "schema": SCHEMA, "model_id": spec.model_id, "safe_id": spec.safe_id, "controls": {},
    }
    for name, rows in ctrl_rows_by_name.items():
        c = CONTROLS[name]
        out["controls"][name] = {
            "spatial": c.spatial,
            "description": c.description,
            "negative_excess_would_falsify": not c.spatial,
            "matched_acc": match_curves(pos_rows, rows, axis="original_acc", n_images=n_images),
            "matched_conf": match_curves(pos_rows, rows, axis="original_conf", n_images=n_images),
        }
    return out


if __name__ == "__main__":
    import argparse

    from .checkpoints import load_checkpoint
    from .config import load_config
    from .data import load_in9_map
    from .data.loaders import preload_uint8
    from .evaluate import resolve_device
    from .intervene import functional_knob, load_sweep_rows
    from .registry import by_id, by_index
    from .seed import seed_everything

    ap = argparse.ArgumentParser(description="Accuracy-matched specificity controls (E1).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int)
    g.add_argument("--id", type=str)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything(cfg.seed)
    spec = by_index(args.index) if args.index is not None else by_id(args.id)
    device = resolve_device()
    model, _ = load_checkpoint(spec, img_size=cfg.img_size, device=device)
    in9_map = load_in9_map()
    data = {v: preload_uint8(cfg, v) for v in ("original", "mixed_same", "mixed_rand")}

    ctrl_rows = {}
    for name, c in CONTROLS.items():
        ctrl_rows[name] = sweep_control(
            model,
            spec,
            cfg,
            c,
            data=data,
            in9_map=in9_map,
            device=device,
            batch=args.batch,
        )
        print(f"  {name:13s} " + "  ".join(
            f"{r['strength']:.2f}:acc={r['original_acc']:.3f},gap={r['bg_gap']*100:.1f}pp"
            for r in ctrl_rows[name]
        ))

    pos_rows = load_sweep_rows(spec, functional_knob(spec))
    rep = specificity_report(
        spec,
        cfg,
        pos_rows,
        ctrl_rows,
        n_images=len(data["mixed_same"][1]),
    )
    d = paths.metrics_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"specificity__{spec.safe_id}.json").write_text(
        json.dumps({"report": rep, "control_curves": ctrl_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"\n{'control':14s} {'class':12s} {'Δ@acc':>9s} {'Δ@conf':>9s}  flags")
    for name, r in rep["controls"].items():
        a, c2 = r["matched_acc"], r["matched_conf"]
        cls = "spatial" if r["spatial"] else "NON-spatial"
        flag = "AMBIGUOUS" if a.get("ambiguous") else ""
        print(f"{name:14s} {cls:12s} {a.get('excess_mean_pp', float('nan')):+9.2f} "
              f"{c2.get('excess_mean_pp', float('nan')):+9.2f}  {flag}")