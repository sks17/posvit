"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import numpy as np

from . import paths

SCHEMA = "posvit.equivalence/1"


def paired_gap_difference(ms_a, mr_a, ms_b, mr_b, *, n: int = 10000, seed: int = 0):
    """
    Computes bootstrap differences of paired BG-Gap values.
    - Pre: `ms_a`, `mr_a`, `ms_b`, and `mr_b` are aligned numeric sequences.
        `n` is a positive integer.
        `seed` is an integer.
    - Post: Returns `(point, diffs)`.
        `point` is the observed gap difference as float.
        `diffs` is a bootstrap array of replicate differences.
        Raises ValueError when input lengths are not aligned.
    """
    a_ms, a_mr, b_ms, b_mr = (np.asarray(x, float) for x in (ms_a, mr_a, ms_b, mr_b))
    if not (len(a_ms) == len(a_mr) == len(b_ms) == len(b_mr)):
        raise ValueError("all four correctness vectors must be aligned on the same foregrounds")
    N = len(a_ms)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for r in range(n):
        idx = rng.integers(0, N, N)
        diffs[r] = (a_ms[idx].mean() - a_mr[idx].mean()) - (b_ms[idx].mean() - b_mr[idx].mean())
    point = float((a_ms.mean() - a_mr.mean()) - (b_ms.mean() - b_mr.mean()))
    return point, diffs


def tost(ms_a, mr_a, ms_b, mr_b, *, margin: float, n: int = 10000, seed: int = 0,
         alpha: float = 0.05) -> "dict[str, Any]":
    """
    Runs TOST equivalence test for two paired BG-Gap vectors.
    - Pre: `ms_a`, `mr_a`, `ms_b`, and `mr_b` are aligned numeric sequences.
        `margin` is a float in (0, 1).
        `n` is a positive integer.
        `seed` is an integer.
        `alpha` is a float in (0, 0.5).
    - Post: Returns dictionary with effect size, confidence interval,
        equivalence verdict, and bootstrap settings.
        Raises ValueError when `margin` is out of range.
    """
    if not (0.0 < margin < 1.0):
        raise ValueError(f"margin must be in (0, 1); got {margin!r}")
    point, diffs = paired_gap_difference(ms_a, mr_a, ms_b, mr_b, n=n, seed=seed)
    lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    return {
        "diff_pp": round(point * 100, 4),
        "ci_lo_pp": round(float(lo) * 100, 4),
        "ci_hi_pp": round(float(hi) * 100, 4),
        "ci_level": round(1 - 2 * alpha, 3),
        "margin_pp": round(margin * 100, 4),
        "equivalent": bool(lo > -margin and hi < margin),
        "se": float(diffs.std(ddof=1)),
        "n_boot": n, "seed": seed, "alpha": alpha,
    }


def mde(se: float, *, alpha: float = 0.05, power: float = 0.80) -> float:
    """
    Computes minimum detectable effect from standard error.
    - Pre: `se` is a non-negative float.
        `alpha` is a float in (0, 1).
        `power` is a float in (0, 1).
    - Post: Returns MDE as float.
    """
    from scipy.stats import norm

    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def equivalence_report(corr_by_model: "dict[str, tuple]", cfg, *, scale: str = "vit_s") -> "dict[str, Any]":
    """
    Builds pairwise equivalence report for one model scale.
    - Pre: `corr_by_model` maps `safe_id` to `(ms, mr)` aligned correctness vectors.
        `cfg` provides margin, bootstrap, and power settings.
        `scale` is a valid model scale string.
    - Post: Returns report dictionary with pairwise TOST results,
        margin sensitivity, and MDE summary.
        Raises ValueError when fewer than two models exist for `scale`.
    """
    keys = sorted(k for k in corr_by_model if k.startswith(f"{scale}__"))
    pairs = list(combinations(keys, 2))
    if not pairs:
        raise ValueError(f"need at least two {scale} models; got {keys}")

    at_margin: dict[str, Any] = {}
    ses = []
    for a, b in pairs:
        (ms_a, mr_a), (ms_b, mr_b) = corr_by_model[a], corr_by_model[b]
        r = tost(
            ms_a,
            mr_a,
            ms_b,
            mr_b,
            margin=cfg.tost_margin,
            n=cfg.bootstrap_n,
            seed=cfg.bootstrap_seed,
            alpha=cfg.power_alpha,
        )
        at_margin[f"{a}|{b}"] = r
        ses.append(r["se"])

    sensitivity = {}
    for m in cfg.tost_margins_sensitivity:
        n_eq = 0
        for a, b in pairs:
            (ms_a, mr_a), (ms_b, mr_b) = corr_by_model[a], corr_by_model[b]
            n_eq += tost(
                ms_a,
                mr_a,
                ms_b,
                mr_b,
                margin=m,
                n=cfg.bootstrap_n,
                seed=cfg.bootstrap_seed,
                alpha=cfg.power_alpha,
            )["equivalent"]
        sensitivity[f"{m * 100:.1f}pp"] = {
            "n_equivalent": n_eq,
            "n_pairs": len(pairs),
            "is_declared_margin": bool(m == cfg.tost_margin),
        }

    widest = max(at_margin.values(), key=lambda r: r["ci_hi_pp"] - r["ci_lo_pp"])
    se_max = float(max(ses))
    mde_pp = mde(se_max, alpha=cfg.power_alpha, power=cfg.power_target) * 100

    return {
        "schema": SCHEMA,
        "scale": scale,
        "declared_margin_pp": round(cfg.tost_margin * 100, 4),
        "n_pairs": len(pairs),
        "n_equivalent": sum(1 for r in at_margin.values() if r["equivalent"]),
        "all_equivalent": all(r["equivalent"] for r in at_margin.values()),
        "widest_ci_pp": [widest["ci_lo_pp"], widest["ci_hi_pp"]],
        "max_abs_diff_pp": round(max(abs(r["diff_pp"]) for r in at_margin.values()), 4),
        "mde_pp": round(mde_pp, 4),
        "mde_from": "max SE across all pairs",
        "power_target": cfg.power_target, "alpha": cfg.power_alpha,
        "mde_below_margin": bool(mde_pp < cfg.tost_margin * 100),
        "pairs": at_margin,
        "margin_sensitivity": sensitivity,
    }


def write_equivalence(report: "dict[str, Any]"):
    """
    Writes equivalence report to disk.
    - Pre: `report` is a JSON-serializable dictionary.
    - Post: Writes `equivalence.json` in metrics directory and returns output path.
    """
    d = paths.metrics_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = d / "equivalence.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out