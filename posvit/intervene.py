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
from .data import load_in9_map, map_to_in9
from .data.loaders import preload_uint8
from .registry import ModelSpec

SCHEMA = "posvit.degrade/1"
KNOB_APE = "ape"
KNOB_ROPE = "rope"


# Positional tensors

def _positional_tensors(model) -> "dict[str, torch.Tensor]":
    """
    Collects positional tensors from the model.
    - Pre: `model` is a loaded model object.
    - Post: Returns a dictionary of positional tensors keyed by name.
    """
    out: dict[str, torch.Tensor] = {}
    pe = getattr(model, "pos_embed", None)
    if pe is not None:
        out["pos_embed"] = pe.data
    if getattr(model, "rope_mixed", False):
        out["freqs"] = model.freqs.data
        out["freqs_t_x"] = model.freqs_t_x.data
        out["freqs_t_y"] = model.freqs_t_y.data
    elif hasattr(model, "rope_mixed"):
        out["freqs_cis"] = model.freqs_cis
    return out


def _blocks_of(model):
    """
    Resolves the transformer block list for a bare or wrapped model.
    - Pre: `model` is a loaded model object.
    - Post: Returns the block sequence.
        Raises AttributeError when no block list can be found.
    """
    for obj in (model, getattr(model, "model", None)):
        blocks = getattr(obj, "blocks", None)
        if blocks is not None:
            return blocks
    raise AttributeError(
        f"cannot find `.blocks` on {type(model).__name__} or its `.model` attribute"
    )


def _attn_scales(model) -> "list[float]":
    """
    Reads the per-block softmax temperature.

    These are captured on every snapshot rather than only when a control intends to
    change them, so there is one snapshot shape and one restore path.
    - Pre: `model` is a loaded model object with attention blocks.
    - Post: Returns a list of per-block attention scale floats.
    """
    return [float(b.attn.scale) for b in _blocks_of(model)]


def _set_attn_scales(model, scales) -> None:
    """
    Writes the per-block softmax temperature.
    - Pre: `model` is a loaded model object.
        `scales` has one value per transformer block.
    - Post: Returns None after assigning each block's attention scale.
    """
    for b, s in zip(_blocks_of(model), scales):
        b.attn.scale = s


def fingerprint(model) -> str:
    """
    Computes a hash of positional model state.

    Attention scales are included so a leaked temperature fails the same restore check
    as a leaked positional scale.
    - Pre: `model` is a loaded model object.
    - Post: Returns a SHA-256 hex digest string for positional state.
    """
    h = hashlib.sha256()
    for name, t in sorted(_positional_tensors(model).items()):
        h.update(name.encode())
        h.update(t.detach().cpu().numpy().tobytes())
    h.update(repr([round(s, 12) for s in _attn_scales(model)]).encode())
    return h.hexdigest()


def verify_knobs(model, spec: ModelSpec) -> None:
    """
    Validates positional knob declarations.
    - Pre: `model` is a loaded model object.
        `spec` is a valid ModelSpec object.
    - Post: Returns None when model and registry declarations match.
        Raises ValueError when declarations differ.
    """
    model_ape = getattr(model, "pos_embed", None) is not None
    model_rope = hasattr(model, "rope_mixed")
    if model_ape != spec.use_ape or model_rope != spec.has_rope:
        raise ValueError(
            f"{spec.model_id}: registry and checkpoint disagree about the positional signal.\n"
            f"  registry: use_ape={spec.use_ape}, has_rope={spec.has_rope}\n"
            f"  model:    use_ape={model_ape}, has_rope={model_rope}\n"
            "configs/models.yaml is wrong, or this is not the architecture you think it is."
        )


# Snapshot and restore

@dataclass(frozen=True, eq=False)
class PositionalSnapshot:
    """
    Holds every piece of model state an intervention may mutate.

    Attention scales live here alongside the positional tensors so there is a single
    snapshot shape, rather than a base form and an extended form that a caller can
    pair with the wrong restore.
    """
    tensors: "dict[str, torch.Tensor]"
    fingerprint: str
    knobs: "tuple[str, ...]"
    attn_scales: "list[float]"


def snapshot(model, spec: ModelSpec) -> PositionalSnapshot:
    """
    Captures positional model state snapshot.
    - Pre: `model` is a loaded model object.
        `spec` is a valid ModelSpec object.
    - Post: Returns a PositionalSnapshot with cloned tensors and state fingerprint.
        Raises ValueError when the registry and checkpoint declarations differ.
    """
    verify_knobs(model, spec)
    knobs = tuple(k for k, present in ((KNOB_APE, spec.use_ape), (KNOB_ROPE, spec.has_rope)) if present)
    return PositionalSnapshot(
        tensors={k: v.clone() for k, v in _positional_tensors(model).items()},
        fingerprint=fingerprint(model),
        knobs=knobs,
        attn_scales=_attn_scales(model),
    )


def restore(model, snap: PositionalSnapshot) -> None:
    """
    Restores positional model state from snapshot.

    Parameters and buffers are written through `.data` so the registered objects
    survive; `freqs_cis` is a plain attribute and is reassigned from a clone.
    - Pre: `model` is a loaded model object.
        `snap` is a valid PositionalSnapshot object.
    - Post: Returns None after restoring all positional state from the snapshot.
    """
    for name, saved in snap.tensors.items():
        if name == "freqs_cis":
            model.freqs_cis = saved.clone()
        else:
            getattr(model, name).data.copy_(saved)
    _set_attn_scales(model, snap.attn_scales)


def set_scale(model, snap: PositionalSnapshot, knob: str, beta: float) -> None:
    """
    Sets attenuation scale for one positional knob.
    - Pre: `model` is a loaded model object.
        `snap` is a valid PositionalSnapshot object.
        `knob` is `ape` or `rope`.
        `beta` is a numeric scale value.
    - Post: Returns None after updating model positional tensors from snapshot-scaled values.
        Raises ValueError for unsupported knob or missing knob tensors.
    """
    if knob == KNOB_APE:
        if "pos_embed" not in snap.tensors:
            raise ValueError("model has no APE knob (pos_embed is None)")
        model.pos_embed.data.copy_(snap.tensors["pos_embed"] * beta)
    elif knob == KNOB_ROPE:
        if "freqs" in snap.tensors:
            model.freqs.data.copy_(snap.tensors["freqs"] * beta)
        elif "freqs_cis" in snap.tensors:
            ang = torch.angle(snap.tensors["freqs_cis"])
            model.freqs_cis = torch.polar(torch.ones_like(ang), beta * ang)
        else:
            raise ValueError("model has no RoPE knob")
    else:
        raise ValueError(f"unknown knob {knob!r}; expected {KNOB_APE!r} or {KNOB_ROPE!r}")


@contextmanager
def scaled(model, snap: PositionalSnapshot, knob: str, beta: float, *, verify: bool = True):
    """
    Applies temporary scaling and guarantees restoration.
    - Pre: `model` is a loaded model object.
        `snap` is a valid PositionalSnapshot object.
        `knob` is `ape` or `rope`.
        `beta` is a numeric scale value.
        `verify` is True or False.
    - Post: Yields the scaled model in context.
        Always restores the original positional state on exit.
        Raises RuntimeError when restoration verification fails.
    """
    restore(model, snap)
    set_scale(model, snap, knob, beta)
    try:
        yield model
    finally:
        restore(model, snap)
        if verify and fingerprint(model) != snap.fingerprint:
            raise RuntimeError(
                "positional restore did not round-trip exactly - the model is CONTAMINATED. "
                f"knob={knob} beta={beta}. Do not trust any result computed after this point."
            )


def functional_knob(spec: ModelSpec) -> str:
    """
    Returns the functional positional knob for a model.
    - Pre: `spec` is a valid ModelSpec object.
    - Post: Returns `rope` when `spec.has_rope` is True, else `ape`.
    """
    return KNOB_ROPE if spec.has_rope else KNOB_APE


def vestigial_knob(spec: ModelSpec) -> "str | None":
    """
    Returns the vestigial positional knob for hybrid models.
    - Pre: `spec` is a valid ModelSpec object.
    - Post: Returns `ape` for hybrid models and None for non-hybrid models.
    """
    if spec.has_rope and spec.use_ape:
        return KNOB_APE
    return None


# Fast evaluation path

@torch.no_grad()
def correct_mask(model, u8, labels, in9_map, cfg, device: str, batch: int = 256) -> np.ndarray:
    """
    Computes per-image IN-9 correctness mask.
    - Pre: `model` is a loaded model object.
        `u8` is preloaded uint8 image tensor.
        `labels` is an array of IN-9 labels.
        `in9_map` is a valid ImageNet-to-IN9 mapping.
        `cfg` has normalization values.
        `device` is a valid torch device string.
        `batch` is a positive integer.
    - Post: Returns a boolean numpy array with per-image correctness.
    """
    correct, _ = evaluate_batch(model, u8, labels, in9_map, cfg, device, batch)
    return correct


@torch.no_grad()
def evaluate_batch(model, u8, labels, in9_map, cfg, device: str, batch: int = 256):
    """
    Computes per-image correctness and confidence over a preloaded variant.

    Confidence is the maximum softmax probability. Two interventions can reach the same
    accuracy with very different calibration, so controls are matched on both axes.
    - Pre: `model` is a loaded model object.
        `u8` is a preloaded uint8 image tensor.
        `labels` is an array of IN-9 labels.
        `in9_map` is a valid ImageNet-to-IN9 mapping.
        `cfg` has normalization values.
        `device` is a valid torch device string.
        `batch` is a positive integer.
    - Post: Returns `(correct, confidence)` as numpy arrays of length `len(labels)`.
    """
    mean = torch.tensor(cfg.norm_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.norm_std, device=device).view(1, 3, 1, 1)
    correct = np.zeros(len(labels), dtype=bool)
    confidence = np.zeros(len(labels), dtype=np.float64)
    for i in range(0, len(labels), batch):
        x = u8[i:i + batch].to(device, non_blocking=True).float().div_(255)
        probs = model((x - mean) / std).softmax(dim=1)
        conf, pred = probs.max(dim=1)
        p9 = np.array([map_to_in9(p, in9_map) for p in pred.cpu().numpy()])
        correct[i:i + len(p9)] = p9 == labels[i:i + len(p9)]
        confidence[i:i + len(p9)] = conf.cpu().numpy()
    return correct, confidence


def load_sweep_rows(spec: ModelSpec, knob: str) -> "list[dict[str, Any]]":
    """
    Reads a positional sweep as flat rows for curve matching.

    Control curves are recorded as `(strength, accuracy, confidence, bg_gap)` rows, so
    the positional sweep is reshaped to match before the two are compared.
    - Pre: `spec` is a ModelSpec object and `knob` names a swept knob.
    - Post: Returns a list of row dictionaries ordered by descending beta.
        Raises FileNotFoundError when the sweep record is missing.
    """
    path = paths.degrade_dir() / f"{spec.safe_id}__{knob}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no sweep at {path}. Run `posvit intervene --id {spec.model_id}` first."
        )
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    rows = []
    for beta in record["betas"]:
        entry = record["results"][f"{beta:.2f}"]
        rows.append({
            "strength": float(beta),
            "original_acc": entry["original_acc"],
            "original_conf": entry.get("original_conf"),
            "bg_gap": entry["bg_gap"],
        })
    return rows


# Intervention sweep

def sweep(
    model,
    spec: ModelSpec,
    cfg,
    *,
    knob: "str | None" = None,
    levels: int = 11,
    device: str = "cpu",
    in9_map: "dict[str, int] | None" = None,
    batch: int = 256,
    write: bool = True,
) -> "dict[str, Any]":
    """
    Runs positional intervention sweep for one model.
    - Pre: `model` is a loaded model object.
        `spec` is a valid ModelSpec object.
        `cfg` has valid settings.
        `knob` is None or a valid knob string.
        `levels` and `batch` are positive integers.
        `device` is a valid torch device string.
        `in9_map` is None or valid mapping dictionary.
        `write` is True or False.
    - Post: Returns a sweep report dictionary.
        Optionally writes the report JSON file when `write` is True.
    """
    in9_map = in9_map if in9_map is not None else load_in9_map()
    knob = knob or functional_knob(spec)
    betas = [round(float(b), 4) for b in np.linspace(1.0, 0.0, levels)]

    data = {v: preload_uint8(cfg, v) for v in ("original", "mixed_same", "mixed_rand")}
    if data["mixed_same"][2] != data["mixed_rand"][2]:
        raise ValueError("MIXED-SAME / MIXED-RAND fg_id order mismatch - cannot pair by index")

    def ev(variant):
        u8, lab, _ = data[variant]
        return evaluate_batch(model, u8, lab, in9_map, cfg, device, batch)

    snap = snapshot(model, spec)
    if knob not in snap.knobs:
        raise ValueError(f"{spec.model_id} has no {knob!r} knob; available: {snap.knobs}")

    # Identity self-test. At beta=1 the intervention must leave predictions untouched.
    # Predictions rather than tensors: for axial models beta=1 round-trips through polar
    # coordinates, so the bits may differ by a rounding step while the argmax cannot.
    base, _ = ev("original")
    with scaled(model, snap, knob, 1.0):
        if not np.array_equal(ev("original")[0], base):
            raise RuntimeError(
                f"{spec.model_id}: beta=1.0 changed predictions, so the intervention is mis-wired."
            )

    out = {
        "schema": SCHEMA, "model_id": spec.model_id, "safe_id": spec.safe_id,
        "knob": knob, "knob_role": "functional" if knob == functional_knob(spec) else "vestigial",
        "betas": betas, "snapshot_fingerprint": snap.fingerprint,
        "identity_acc": round(float(base.mean()), 6),
        "fg_ids": data["mixed_same"][2], "results": {},
    }

    for beta in betas:
        with scaled(model, snap, knob, beta):
            orig, conf = ev("original")
            ms, _ = ev("mixed_same")
            mr, _ = ev("mixed_rand")
        out["results"][f"{beta:.2f}"] = {
            "original_acc": round(float(orig.mean()), 6),
            "original_conf": round(float(conf.mean()), 6),
            "bg_gap": round(float(ms.mean() - mr.mean()), 6),
            # Per-image vectors are kept so downstream analyses can resample foreground
            # identities rather than beta points when building a slope interval.
            "ms": ms.astype(np.uint8).tolist(),
            "mr": mr.astype(np.uint8).tolist(),
        }
        print(f"  {spec.safe_id} {knob} beta={beta:.2f}  "
              f"ORIG={out['results'][f'{beta:.2f}']['original_acc']:.4f}  "
              f"gap={out['results'][f'{beta:.2f}']['bg_gap'] * 100:.2f}pp", flush=True)

    if write:
        d = paths.degrade_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{spec.safe_id}__{knob}.json").write_text(json.dumps(out), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse
    import sys

    from .checkpoints import load_checkpoint
    from .config import load_config
    from .evaluate import resolve_device
    from .registry import by_id, by_index
    from .seed import seed_everything

    ap = argparse.ArgumentParser(description="Positional attenuation sweep on frozen weights.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, help="registry index, as supplied by SLURM_ARRAY_TASK_ID")
    g.add_argument("--id", type=str)
    ap.add_argument("--knob", choices=[KNOB_APE, KNOB_ROPE], default=None)
    ap.add_argument("--vestigial", action="store_true", help="sweep the UNUSED signal instead")
    ap.add_argument("--levels", type=int, default=11)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything(cfg.seed)
    spec = by_index(args.index) if args.index is not None else by_id(args.id)
    device = resolve_device()
    model, _ = load_checkpoint(spec, img_size=cfg.img_size, device=device)

    knob = args.knob or (vestigial_knob(spec) if args.vestigial else functional_knob(spec))
    if knob is None:
        print(f"{spec.model_id} has no vestigial signal (not a hybrid) - nothing to sweep.")
        sys.exit(0)
    sweep(model, spec, cfg, knob=knob, levels=args.levels, device=device, batch=args.batch)