"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from . import paths
from .intervene import functional_knob, scaled, snapshot
from .registry import ModelSpec

SCHEMA = "posvit.waterbirds/1"
CONGRUENT = (0, 3)
INCONGRUENT = (1, 2)
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
BETAS = (1.0, 0.75, 0.5, 0.25, 0.0)


def group_of(y, place):
    """
    Computes Waterbirds group index.
    - Pre: `y` and `place` are aligned numeric arrays or scalars.
    - Post: Returns group index values computed as `2*y + place`.
    """
    return 2 * np.asarray(y) + np.asarray(place)


def load_split(split: str, *, img_size: int = 224, limit: "int | None" = None):
    """
    Loads one Waterbirds split into memory.
    - Pre: `split` is one of `train`, `val`, or `test`.
        `img_size` is a positive integer.
        `limit` is None or a positive integer.
    - Post: Returns `(images_u8, labels, places)`.
        `images_u8` is uint8 NCHW tensor.
        `labels` and `places` are numpy arrays.
    """
    root = paths.waterbirds_root()
    want = SPLIT_CODE[split]
    rows = []
    with open(root / "metadata.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["split"]) == want:
                rows.append((r["img_filename"], int(r["y"]), int(r["place"])))
    if limit:
        rows = rows[:limit]

    imgs, ys, places = [], [], []
    for fname, y, place in rows:
        im = Image.open(root / fname).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
        imgs.append(np.transpose(np.asarray(im, dtype=np.uint8), (2, 0, 1)))
        ys.append(y)
        places.append(place)
    return torch.from_numpy(np.stack(imgs)), np.array(ys), np.array(places)


def spurious_gap(correct: np.ndarray, groups: np.ndarray) -> float:
    """
    Computes spurious gap between congruent and incongruent groups.
    - Pre: `correct` and `groups` are aligned arrays.
    - Post: Returns `acc(congruent) - acc(incongruent)` as float.
        Returns NaN if one side is missing.
    """
    con = np.isin(groups, CONGRUENT)
    inc = np.isin(groups, INCONGRUENT)
    if not con.any() or not inc.any():
        return float("nan")
    return float(correct[con].mean() - correct[inc].mean())


@dataclass(frozen=True, eq=False)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(X: np.ndarray) -> "Standardizer":
        """
        Fits standardization parameters.
        - Pre: `X` is a numeric 2D numpy array.
        - Post: Returns Standardizer with feature-wise mean and std.
        """
        return Standardizer(X.mean(0), X.std(0) + 1e-6)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """
        Applies stored standardization.
        - Pre: `X` is a numeric array with matching feature dimension.
        - Post: Returns standardized array.
        """
        return (X - self.mean) / self.std


def fit_linear_head(X: np.ndarray, y: np.ndarray, *, seeds: int = 5, epochs: int = 200,
                    lr: float = 1e-2, weight_decay: float = 1e-3,
                    sample_weight: "np.ndarray | None" = None):
    """
    Fits ensemble of linear classification heads.
    - Pre: `X` and `y` are aligned training arrays.
        `seeds` and `epochs` are positive integers.
        `lr` and `weight_decay` are non-negative floats.
        `sample_weight` is None or aligned numeric weights.
    - Post: Returns list of trained torch linear heads.
    """
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    w = None if sample_weight is None else torch.tensor(sample_weight, dtype=torch.float32)
    heads = []
    for s in range(seeds):
        torch.manual_seed(s)
        head = torch.nn.Linear(Xt.shape[1], 2)
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(Xt), yt, reduction="none")
            (loss if w is None else loss * w).mean().backward()
            opt.step()
        heads.append(head)
    return heads


def head_correct(heads, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes per-image correctness from ensemble heads.
    - Pre: `heads` is a non-empty list of trained heads.
        `X` and `y` are aligned arrays.
    - Post: Returns boolean numpy array of correctness flags.
    """
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = torch.stack([h(Xt) for h in heads]).mean(0)
    return (logits.argmax(1).numpy() == y)


def fit_erm_head(X, y, *, seeds):
    """
    Fits ERM measurement head.
    - Pre: `X` and `y` are aligned training arrays.
        `seeds` is a positive integer.
    - Post: Returns trained ERM head ensemble.
    """
    return fit_linear_head(X, y, seeds=seeds)


def fit_dfr_head(X, y, groups, *, seeds):
    """
    Fits DFR reference head with group-balanced weighting.
    - Pre: `X`, `y`, and `groups` are aligned arrays.
        `seeds` is a positive integer.
    - Post: Returns trained DFR head ensemble.
    """
    counts = {g: int((groups == g).sum()) for g in np.unique(groups)}
    w = np.array([1.0 / max(counts[g], 1) for g in groups], dtype=np.float64)
    return fit_linear_head(X, y, seeds=seeds, sample_weight=w * len(w) / w.sum())


@torch.no_grad()
def extract_features(model, u8, cfg, device: str, batch: int = 128) -> np.ndarray:
    """
    Extracts penultimate features from model.
    - Pre: `model` is a loaded model object.
        `u8` is uint8 NCHW tensor.
        `cfg` provides normalization values.
        `device` is a valid torch device string.
        `batch` is a positive integer.
    - Post: Returns feature matrix as numpy array.
    """
    mean = torch.tensor(cfg.norm_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.norm_std, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(u8), batch):
        x = u8[i:i + batch].to(device).float().div_(255)
        out.append(model.forward_features((x - mean) / std).cpu().numpy())
    return np.concatenate(out)


def waterbirds_sweep(model, spec: ModelSpec, cfg, *, device: str = "cpu",
                     limit: "int | None" = None, write: bool = True) -> "dict[str, Any]":
    """
    Runs Waterbirds beta sweep with frozen readout protocol.
    - Pre: `model` is a loaded model object.
        `spec` is a valid ModelSpec object.
        `cfg` has waterbirds and feature settings.
        `device` is a valid torch device string.
        `limit` is None or positive integer.
        `write` is True or False.
    - Post: Returns sweep report dictionary.
        Writes raw Waterbirds report JSON when `write` is True.
        Raises RuntimeError when baseline positive-control check fails.
    """
    knob = functional_knob(spec)
    snap = snapshot(model, spec)

    tr_u8, tr_y, tr_place = load_split("train", img_size=cfg.img_size, limit=limit)
    te_u8, te_y, te_place = load_split("test", img_size=cfg.img_size, limit=limit)
    tr_g, te_g = group_of(tr_y, tr_place), group_of(te_y, te_place)

    # --- everything below is fitted ONCE, at beta=1, and then frozen (Part 2.5) ---
    tr_feat = extract_features(model, tr_u8, cfg, device)
    std = Standardizer.fit(tr_feat)
    tr_z = std.apply(tr_feat)
    erm = fit_erm_head(tr_z, tr_y, seeds=cfg.wb_head_seeds)
    dfr = fit_dfr_head(tr_z, tr_y, tr_g, seeds=cfg.wb_head_seeds)

    base_z = std.apply(extract_features(model, te_u8, cfg, device))
    base_gap = spurious_gap(head_correct(erm, base_z, te_y), te_g)
    if base_gap < cfg.wb_min_baseline_gap:
        raise RuntimeError(
            f"{spec.model_id}: the ERM head's baseline spurious gap is {base_gap * 100:.2f} pp, "
            f"below wb_min_baseline_gap = {cfg.wb_min_baseline_gap * 100:.2f} pp. The readout "
            "does not exhibit the shortcut, so a null result under attenuation would be "
            "uninformative rather than evidence of absence."
        )

    results = {}
    for beta in BETAS:
        with scaled(model, snap, knob, beta):
            z = std.apply(extract_features(model, te_u8, cfg, device))
        erm_c, dfr_c = head_correct(erm, z, te_y), head_correct(dfr, z, te_y)
        results[f"{beta:.2f}"] = {
            "acc": round(float(erm_c.mean()), 6),
            "spurious_gap": round(spurious_gap(erm_c, te_g), 6),
            "worst_group": round(float(min(erm_c[te_g == g].mean() for g in range(4))), 6),
            "dfr_acc": round(float(dfr_c.mean()), 6),
            "dfr_spurious_gap": round(spurious_gap(dfr_c, te_g), 6),
        }

    out = {
        "schema": SCHEMA, "model_id": spec.model_id, "safe_id": spec.safe_id, "knob": knob,
        "betas": list(BETAS),
        "baseline_gap": round(base_gap, 6),
        "positive_control_passed": True,
        "acc_floor": cfg.wb_acc_floor,
        "n_train": int(len(tr_y)), "n_test": int(len(te_y)),
        "head_seeds": cfg.wb_head_seeds,
        "design_note": (
            "measurement head = ERM fitted once at beta=1 and FROZEN; standardizer "
            "frozen; only test features re-extracted per beta."
        ),
        "results": results,
    }
    if write:
        d = paths.results_raw() / "waterbirds"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{spec.safe_id}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def summarize(records: "dict[str, dict]", cfg) -> "dict[str, Any]":
    """
    Summarizes Waterbirds sweep trends across models.
    - Pre: `records` maps model ids to sweep reports.
        `cfg` provides `wb_acc_floor`.
    - Post: Returns summary dictionary with guarded trend counts and per-model details.
    """
    per = {}
    for safe_id, rec in records.items():
        usable = [
            b
            for b in rec["betas"]
            if rec["results"][f"{b:.2f}"]["acc"] >= cfg.wb_acc_floor
        ]
        gaps = [rec["results"][f"{b:.2f}"]["spurious_gap"] for b in usable]
        per[safe_id] = {
            "n_betas_used": len(usable), "n_betas_total": len(rec["betas"]),
            "betas_used": usable,
            "gap_at_1": rec["results"]["1.00"]["spurious_gap"],
            "gap_at_min_beta": gaps[-1] if gaps else None,
            "widens": bool(len(gaps) >= 2 and gaps[-1] > gaps[0]),
        }
    widened = sum(1 for v in per.values() if v["widens"])
    spread = [r["results"]["1.00"]["spurious_gap"] for r in records.values()]
    return {
        "schema": SCHEMA, "acc_floor": cfg.wb_acc_floor,
        "n_models": len(per), "n_widened": widened,
        "intact_gap_spread_pp": round((max(spread) - min(spread)) * 100, 4) if spread else None,
        "models": per,
    }