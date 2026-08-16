"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from . import paths
from .registry import ModelSpec, all_models

SCHEMA = "posvit.doseresponse/1"
SENSITIVITY_FLOORS = (0.70, 0.80, 0.85)


def load_sweep(spec: ModelSpec, knob: str) -> "dict[str, Any]":
    """
    Loads one intervention sweep result.
    - Pre: `spec` is a valid ModelSpec object.
        `knob` is a valid knob string.
    - Post: Returns parsed sweep dictionary from disk.
        Raises FileNotFoundError when the sweep file is missing.
    """
    p = paths.degrade_dir() / f"{spec.safe_id}__{knob}.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"no sweep at {p} - run `python -m posvit.intervene --id {spec.model_id}` first"
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def apply_floor(record: "dict[str, Any]", floor: float) -> "dict[str, Any]":
    """
    Splits sweep betas by accuracy floor.
    - Pre: `record` is a valid sweep dictionary with `betas` and `results`.
        `floor` is a float in [0, 1].
    - Post: Returns dictionary with included and excluded betas and counts.
    """
    kept, dropped = [], []
    for b in record["betas"]:
        key = f"{b:.2f}"
        (kept if record["results"][key]["original_acc"] >= floor else dropped).append(b)
    return {
        "floor": floor,
        "betas_used": kept,
        "betas_excluded": dropped,
        "n_used": len(kept),
        "n_total": len(record["betas"]),
    }


def spearman_rho(x, y) -> float:
    """
    Computes Spearman rank correlation.
    - Pre: `x` and `y` are numeric sequences of equal length.
    - Post: Returns Spearman rho as float.
    """
    from scipy.stats import spearmanr

    return float(spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic)


def bootstrap_slope(
    ms: np.ndarray, mr: np.ndarray, betas, *, n: int = 10000, seed: int = 0, alpha: float = 0.05
) -> "dict[str, Any]":
    """
    Bootstraps slope confidence interval for BG-Gap curve.
    - Pre: `ms` and `mr` are numeric arrays with shape (n_betas, n_samples).
        `betas` is a numeric sequence with `n_betas` entries.
        `n` is a positive integer.
        `seed` is an integer.
        `alpha` is a float in (0, 1).
    - Post: Returns dictionary with slope, confidence interval, p-value, and bootstrap settings.
    """
    x = 1.0 - np.asarray(betas, float)
    N = ms.shape[1]
    rng = np.random.default_rng(seed)

    point = float(np.polyfit(x, ms.mean(axis=1) - mr.mean(axis=1), 1)[0])
    slopes = np.empty(n)
    for r in range(n):
        idx = rng.integers(0, N, N)
        gaps = ms[:, idx].mean(axis=1) - mr[:, idx].mean(axis=1)
        slopes[r] = np.polyfit(x, gaps, 1)[0]

    return {
        "slope": round(point, 6),
        "ci_lo": round(float(np.percentile(slopes, 100 * alpha / 2)), 6),
        "ci_hi": round(float(np.percentile(slopes, 100 * (1 - alpha / 2))), 6),
        "p_one_sided": float((int(np.sum(slopes <= 0)) + 1) / (n + 1)),
        "n_boot": n, "seed": seed, "alpha": alpha,
    }


def bh_fdr(pvals, q: float = 0.05) -> np.ndarray:
    """
    Applies Benjamini-Hochberg FDR correction.
    - Pre: `pvals` is a numeric sequence of p-values.
        `q` is a float in (0, 1).
    - Post: Returns boolean reject mask aligned with `pvals`.
    """
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    below = p[order] <= q * np.arange(1, m + 1) / m
    reject = np.zeros(m, dtype=bool)
    hits = np.nonzero(below)[0]
    if len(hits):
        reject[order[: hits.max() + 1]] = True
    return reject


def _matrices(record: "dict[str, Any]", betas) -> "tuple[np.ndarray, np.ndarray]":
    """
    Builds paired correctness matrices for selected betas.
    - Pre: `record` is a valid sweep dictionary.
        `betas` is a sequence of beta values present in `record`.
    - Post: Returns `(ms, mr)` numpy arrays with shape (n_betas, n_samples).
    """
    ms = np.array([record["results"][f"{b:.2f}"]["ms"] for b in betas], dtype=np.float64)
    mr = np.array([record["results"][f"{b:.2f}"]["mr"] for b in betas], dtype=np.float64)
    return ms, mr


def dose_response(record: "dict[str, Any]", cfg) -> "dict[str, Any]":
    """
    Computes dose-response statistics for one sweep.
    - Pre: `record` is a valid sweep dictionary.
        `cfg` provides `acc_floor`, bootstrap, and CI settings.
    - Post: Returns dictionary with guard results, monotonicity summary,
        slope inference, and floor sensitivity report.
    """
    guard = apply_floor(record, cfg.acc_floor)
    if guard["n_used"] < 3:
        raise ValueError(
            f"{record['safe_id']}: only {guard['n_used']} beta(s) above acc_floor="
            f"{cfg.acc_floor}; cannot fit a slope. The intervention may be mis-wired, or the "
            "floor is too high for this model."
        )

    betas = guard["betas_used"]
    ms, mr = _matrices(record, betas)
    gaps = (ms.mean(axis=1) - mr.mean(axis=1)).tolist()

    boot = bootstrap_slope(
        ms, mr, betas, n=cfg.bootstrap_n, seed=cfg.bootstrap_seed, alpha=cfg.ci_alpha
    )

    sensitivity = {}
    for f in SENSITIVITY_FLOORS:
        g = apply_floor(record, f)
        if g["n_used"] < 3:
            sensitivity[f"{f:.2f}"] = {"n_used": g["n_used"], "insufficient": True}
            continue
        m2, r2 = _matrices(record, g["betas_used"])
        gp = m2.mean(axis=1) - r2.mean(axis=1)
        x2 = 1.0 - np.asarray(g["betas_used"], float)
        sensitivity[f"{f:.2f}"] = {
            "n_used": g["n_used"],
            "slope": round(float(np.polyfit(x2, gp, 1)[0]), 6),
            "rho": round(spearman_rho(x2, gp), 4),
        }

    return {
        "schema": SCHEMA,
        "model_id": record["model_id"], "safe_id": record["safe_id"],
        "knob": record["knob"], "knob_role": record["knob_role"],
        "guard": guard,
        "betas_used": betas,
        "bg_gap_by_beta": [round(g, 6) for g in gaps],
        "bg_gap_intact": round(gaps[0], 6) if betas and betas[0] == 1.0 else None,
        # Gaps at EVERY beta, including those the floor excluded, so the figure can draw
        # the excluded points as open markers instead of silently truncating the curve.
        "bg_gap_all": {
            f"{b:.2f}": round(
                float(
                    np.mean(record["results"][f"{b:.2f}"]["ms"])
                    - np.mean(record["results"][f"{b:.2f}"]["mr"])
                ),
                6,
            )
            for b in record["betas"]
        },
        "spearman_rho": round(spearman_rho(1.0 - np.asarray(betas, float), gaps), 4),
        "slope": boot,
        "floor_sensitivity": sensitivity,
    }


def analyze_all(cfg, knob: "str | None" = None, q: float = 0.05) -> "dict[str, Any]":
    """
    Runs dose-response analysis for all available models.
    - Pre: `cfg` provides analysis settings.
        `knob` is None or a valid knob string.
        `q` is a float in (0, 1).
    - Post: Returns aggregated report for all analyzed models.
        Writes `dose_response.json` to the metrics directory.
    """
    from .intervene import functional_knob

    per: dict[str, dict[str, Any]] = {}
    for spec in all_models():
        k = knob or functional_knob(spec)
        try:
            rec = load_sweep(spec, k)
        except FileNotFoundError:
            continue
        per[spec.safe_id] = dose_response(rec, cfg)

    keys = sorted(per)
    reject = bh_fdr([per[k]["slope"]["p_one_sided"] for k in keys], q=q)
    for k, r in zip(keys, reject):
        per[k]["fdr_pass"] = bool(r)

    out = {
        "schema": SCHEMA,
        "acc_floor": cfg.acc_floor,
        "fdr_q": q,
        "n_models": len(keys),
        "n_positive_slope_ci_excludes_zero": sum(
            1 for k in keys if per[k]["slope"]["ci_lo"] > 0
        ),
        "n_fdr_pass": int(reject.sum()),
        "models": per,
    }
    d = paths.metrics_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "dose_response.json").write_text(
        json.dumps(out, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    from .config import load_config
    cfg = load_config()
    res = analyze_all(cfg)
    print(f"acc_floor={res['acc_floor']}  models={res['n_models']}")
    for safe_id, r in sorted(res["models"].items()):
        s = r["slope"]
        print(
            f"  {safe_id:22s} {r['knob_role']:10s} rho={r['spearman_rho']:+.3f} "
            f"slope={s['slope']:+.4f} [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] "
            f"p={s['p_one_sided']:.3g} FDR={'PASS' if r['fdr_pass'] else 'no'} "
            f"(betas {r['guard']['n_used']}/{r['guard']['n_total']})"
        )
    print(f"\n{res['n_positive_slope_ci_excludes_zero']}/{res['n_models']} slopes with CI > 0; "
          f"{res['n_fdr_pass']} pass BH-FDR at q={res['fdr_q']}")