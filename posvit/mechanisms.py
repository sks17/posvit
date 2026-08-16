"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from . import paths
from .registry import ModelSpec

SCHEMA = "posvit.mechanisms/1"


# Probe protocol

@dataclass(frozen=True)
class Probe:
    name: str
    claim: str
    metric: str
    kill_criterion: str
    requires: "tuple[str, ...]" = ()

    def applicable(self, spec: ModelSpec) -> bool:
        """
        Checks if this probe applies to a model spec.
        - Pre: `spec` is a valid ModelSpec object.
        - Post: Returns True when probe requirements are met.
            Returns False when required model properties are missing.
        """
        if "rope_mixed" in self.requires and not (spec.has_rope and "mixed" in spec.key):
            return False
        if "ape" in self.requires and not spec.use_ape:
            return False
        return True


PROBES = {
    p.name: p for p in (
        Probe("registers", "high-norm artifact tokens in BACKGROUND patches predict per-image "
              "reliance", "auc",
              "AUC CI entirely below probe_auc_support -> killed"),
        Probe("attention_routing", "flattening attention temperature reproduces the positional "
              "effect at matched accuracy", "excess_pp",
              "excess CI entirely below probe_excess_support_pp -> killed"),
        Probe("bg_decodability", "attenuating position makes the BACKGROUND more decodable",
              "trend_sign", "observed sign opposite to predicted -> killed"),
        Probe("rope_high_band", "a high-|freq| rotary band is the shared substrate", "excess_pp",
              "excess CI entirely below probe_excess_support_pp at MATCHED accuracy -> killed",
              requires=("rope_mixed",)),
        Probe("attention_entropy", "attention spread predicts per-image reliance", "auc",
              "AUC CI entirely below probe_auc_support -> killed"),
    )
}


def verdict(probe: Probe, value: "dict[str, Any] | None", cfg,
            *, guards_met: bool = True) -> "dict[str, Any]":
    """
    Computes probe verdict state.
    - Pre: `probe` is a valid Probe object.
        `value` is None or a result dictionary.
        `cfg` provides probe thresholds.
        `guards_met` is True or False.
    - Post: Returns verdict dictionary with status and supporting values.
        Status is one of supported, killed, inconclusive, or not_applicable.
    """
    if value is None:
        return {"verdict": "not_applicable", "reason": "probe does not apply to this model"}
    if not guards_met:
        return {"verdict": "inconclusive", "reason": "the row's guards were not satisfied"}

    if probe.metric == "auc":
        thr = cfg.probe_auc_support
        lo, hi = value["ci_lo"], value["ci_hi"]
        v = "supported" if lo > thr else "killed" if hi < thr else "inconclusive"
        extra = {"anti_predictive": bool(hi < 0.5)}     # 0.35 is signal with the sign flipped
    elif probe.metric == "excess_pp":
        thr = cfg.probe_excess_support_pp
        lo, hi = value["ci_lo_pp"], value["ci_hi_pp"]
        v = "supported" if lo > thr else "killed" if hi < thr else "inconclusive"
        extra = {}
    else:                                               # trend_sign
        v = "supported" if value.get("supports_hypothesis") else "killed"
        extra = {"observed_sign": value.get("observed_sign")}

    return {"verdict": v, "threshold": thr if probe.metric != "trend_sign" else None,
            "value": value, **extra}


# E2 / E3 knobs

def _require_mixed(model, snap, spec: ModelSpec) -> None:
    """
    Validates RoPE-Mixed applicability for band edits.
    - Pre: `model` and `snap` are valid objects.
        `spec` is a valid ModelSpec object.
    - Post: Returns None when model supports learnable mixed rotary frequencies.
        Raises ValueError when requirement is not met.
    """
    if not (spec.has_rope and "freqs" in snap.tensors):
        raise ValueError(
            f"{spec.model_id}: no learnable rotary band, because this is an axial or "
            "APE-only model. Call `Probe.applicable(spec)` first. This raises rather than "
            "returning a status a caller can discard, which would leave the sweep measuring "
            "an unmodified model at every strength."
        )


def attenuate_rope_band(model, snap, spec: ModelSpec, *, band: str, beta: float, frac: float):
    """
    Attenuates one magnitude band of mixed rotary frequencies.
    - Pre: `model` and `snap` are valid objects.
        `spec` is a valid ModelSpec object.
        `band` is `high` or `low`.
        `beta` and `frac` are numeric values.
    - Post: Returns None after updating model frequency tensor.
        Raises ValueError when mixed-rope requirement is not met.
    """
    _require_mixed(model, snap, spec)
    f = snap.tensors["freqs"].clone()
    mag = f.abs()
    thr = torch.quantile(mag.reshape(-1), 1.0 - frac if band == "high" else frac)
    mask = mag >= thr if band == "high" else mag <= thr
    model.freqs.data.copy_(torch.where(mask, f * float(beta), f))


def zero_rope_layers(model, snap, spec: ModelSpec, *, which: str):
    """
    Zeroes rotary frequencies in early or late layer half.
    - Pre: `model` and `snap` are valid objects.
        `spec` is a valid ModelSpec object.
        `which` is `early` or `late`.
    - Post: Returns None after writing updated frequencies to model.
        Raises ValueError when mixed-rope requirement is not met.
    """
    _require_mixed(model, snap, spec)
    f = snap.tensors["freqs"].clone()
    depth = f.shape[1]
    rows = range(0, depth // 2) if which == "early" else range(depth // 2, depth)
    for li in rows:
        f[:, li, :] = 0.0
    model.freqs.data.copy_(f)


def set_ape_lowpass(model, snap, spec: ModelSpec, *, keep_rings: int, grid: int = 14):
    """
    Applies low-pass filter to APE positional embedding.
    - Pre: `model` and `snap` are valid objects.
        `spec` is a valid ModelSpec object with APE enabled.
        `keep_rings` is a non-negative integer.
        `grid` is a positive integer.
    - Post: Returns None after writing filtered positional embedding.
        Raises ValueError when model has no APE tensor.
    """
    if not spec.use_ape or "pos_embed" not in snap.tensors:
        raise ValueError(f"{spec.model_id}: no APE to low-pass")
    pe = snap.tensors["pos_embed"]
    c = pe.shape[2]
    x = pe.view(1, grid, grid, c)[0].permute(2, 0, 1).cpu().numpy()
    yy, xx = np.indices((grid, grid))
    r = np.round(np.hypot(yy - grid // 2, xx - grid // 2)).astype(int)
    keep = r <= keep_rings
    out = np.empty_like(x)
    for ci in range(c):
        F = np.fft.fftshift(np.fft.fft2(x[ci])) * keep
        out[ci] = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    t = torch.from_numpy(out).permute(1, 2, 0).reshape(1, grid * grid, c)
    model.pos_embed.data.copy_(t.to(pe.device, pe.dtype))


# Spectra

def pe_spectrum(model, spec: ModelSpec, *, grid: int = 14) -> "dict[str, Any]":
    """
    Computes positional spectrum summaries.
    - Pre: `model` is a loaded model object.
        `spec` is a valid ModelSpec object.
        `grid` is a positive integer.
    - Post: Returns dictionary with APE radial spectrum and/or RoPE magnitude histogram.
        Includes commensurability flag.
    """
    out: dict[str, Any] = {"model_id": spec.model_id, "commensurable": False}
    pe = getattr(model, "pos_embed", None)
    if pe is not None:
        x = pe.data.view(1, grid, grid, -1)[0].permute(2, 0, 1).cpu().numpy()
        power = np.abs(np.fft.fftshift(np.fft.fft2(x, axes=(1, 2)), axes=(1, 2))) ** 2
        power = power.mean(0)
        yy, xx = np.indices((grid, grid))
        r = np.round(np.hypot(yy - grid // 2, xx - grid // 2)).astype(int)
        prof = np.array([power[r == k].mean() for k in range(r.max() + 1)])
        out["ape_radial_power"] = (prof / prof.sum()).round(6).tolist()
    if getattr(model, "rope_mixed", False):
        mag = model.freqs.data.abs().reshape(-1).cpu().numpy()
        hist, edges = np.histogram(mag, bins=16, density=True)
        out["rope_freq_hist"] = hist.round(6).tolist()
        out["rope_freq_edges"] = edges.round(6).tolist()
    return out


# Register artifacts

def token_norms(patch_tokens: torch.Tensor) -> torch.Tensor:
    """
    Computes per-token L2 norms.
    - Pre: `patch_tokens` has shape `(B, P, D)`.
    - Post: Returns tensor with shape `(B, P)` of token norms.
    """
    return patch_tokens.norm(dim=-1)


def artifact_threshold(norms: torch.Tensor, k: float) -> float:
    """
    Computes robust artifact threshold from token norms.
    - Pre: `norms` is a numeric tensor.
        `k` is a numeric scale factor.
    - Post: Returns threshold as float using median and MAD statistic.
    """
    flat = norms.reshape(-1)
    med = flat.median()
    mad = (flat - med).abs().median()
    return float(med + k * 1.4826 * mad)


def artifact_fraction(patch_tokens: torch.Tensor, cfg, patch_mask: "torch.Tensor | None" = None):
    """
    Computes per-image artifact fractions.
    - Pre: `patch_tokens` has shape `(B, P, D)`.
        `cfg` provides `artifact_k`.
        `patch_mask` is None or boolean-like mask tensor of shape `(B, grid, grid)`.
    - Post: Returns dictionary with threshold metadata and artifact fractions.
        Includes foreground/background splits when patch mask is provided.
    """
    norms = token_norms(patch_tokens)
    thr = artifact_threshold(norms, cfg.artifact_k)
    art = norms > thr
    out = {
        "threshold": round(thr, 6),
        "k": cfg.artifact_k,
        # The cutoff is computed from the same norms it classifies, so this measures
        # relative tail heaviness and is invariant to a global rescaling of the norms.
        # That bounds what a null here can rule out, so it travels with the result.
        "threshold_is_relative": True,
        "artifact_frac": art.float().mean(1).cpu().numpy().tolist(),
    }
    if patch_mask is not None:
        pm = patch_mask.reshape(patch_mask.shape[0], -1).bool()
        out["fg_artifact_frac"] = (
            (art & pm).float().sum(1).div(pm.float().sum(1).clamp_min(1)).cpu().numpy().tolist()
        )
        out["bg_artifact_frac"] = (
            (art & ~pm).float().sum(1).div((~pm).float().sum(1).clamp_min(1)).cpu().numpy().tolist()
        )
    return out


# Mechanism table

def table3(rows: "dict[str, dict[str, Any]]", cfg) -> "dict[str, Any]":
    """
    Assembles mechanism summary table.
    - Pre: `rows` maps probe names to verdict dictionaries.
        `cfg` provides probe threshold values.
    - Post: Returns table dictionary with verdict counts and localization summary.
    """
    counts = {"supported": 0, "killed": 0, "inconclusive": 0, "not_applicable": 0}
    for r in rows.values():
        counts[r["verdict"]] += 1
    return {
        "schema": SCHEMA,
        "thresholds": {"auc_support": cfg.probe_auc_support,
                       "excess_support_pp": cfg.probe_excess_support_pp,
                       "artifact_k": cfg.artifact_k},
        "counts": counts,
        "resists_localization": bool(counts["killed"] >= 4 and counts["supported"] == 0),
        "probes": rows,
    }


def write_table3(table: "dict[str, Any]"):
    """
    Writes mechanism table report to disk.
    - Pre: `table` is a JSON-serializable dictionary.
    - Post: Writes `mechanisms.json` in metrics directory and returns output path.
    """
    d = paths.metrics_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = d / "mechanisms.json"
    out.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out