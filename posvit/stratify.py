"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import paths
from .data import CANONICAL_CLASSES
from .doseresponse import bh_fdr
from .metrics import join_variants, load_variant, mcnemar_test, paired_bootstrap
from .registry import ModelSpec

SCHEMA = "posvit.stratify/1"
ATTRS = ("fg_area_frac", "center_offset")


# Mask attributes

def load_mask_attributes() -> "tuple[dict[str, dict[str, float]], dict[str, Any]]":
    """
    Loads mask attributes and coverage metadata.
    - Pre: Mask files are expected under `results/raw/masks`.
    - Post: Returns `(attrs, coverage)`.
        `attrs` maps `fg_id` to numeric attribute fields.
        `coverage` reports found, unreadable, and expected counts.
        Raises FileNotFoundError when mask root is missing.
    """
    root = paths.results_raw() / "masks"
    if not root.is_dir():
        raise FileNotFoundError(
            f"no masks at {root}. Run `posvit masks` first.\n"
            "There is deliberately no fallback to another mask set, because silently "
            "substituting one re-bins every size and position result."
        )

    attrs: dict[str, dict[str, float]] = {}
    unreadable: list[str] = []
    for path in sorted(root.rglob("*.npz")):
        try:
            with np.load(path) as d:
                attrs[path.stem] = {k: float(d[k]) for k in ATTRS}
        except Exception as exc:  # noqa: BLE001
            unreadable.append(f"{path}: {type(exc).__name__}")

    manifest = root / "_manifest.json"
    expected = None
    if manifest.is_file():
        expected = json.loads(manifest.read_text(encoding="utf-8")).get("written")

    return attrs, {
        "n_found": len(attrs),
        "n_unreadable": len(unreadable),
        "unreadable": unreadable[:20],
        "n_expected": expected,
        "matches_manifest": expected is None or len(attrs) == expected,
    }


    # Binning

@dataclass(frozen=True, eq=False)
class Binning:
    bins: np.ndarray
    edges: np.ndarray
    n_requested: int
    n_actual: int
    within_class: bool

    def meta(self) -> "dict[str, Any]":
        """
        Builds metadata for binning configuration and outcomes.
        - Pre: Binning fields are initialized.
        - Post: Returns dictionary with requested and actual bins,
            within-class flag, and bin edges.
        """
        return {
            "n_bins_requested": self.n_requested,
            "n_bins_actual": self.n_actual,
            "within_class": self.within_class,
            "bin_edges": [round(float(e), 6) for e in self.edges],
        }


def quantile_bins(values: np.ndarray, n_bins: int, *, within_class: bool = False) -> Binning:
    """
    Computes quantile-based bin assignment.
    - Pre: `values` is a numeric numpy array.
        `n_bins` is a positive integer.
        `within_class` is True or False.
    - Post: Returns Binning object with bin indices and edges.
    """
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)
    return Binning(bins, edges, n_bins, len(edges) - 1, within_class)


def within_class_percentile(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Computes within-class percentile ranks.
    - Pre: `values` and `labels` are aligned numpy arrays.
    - Post: Returns percentile ranks in [0, 1] for each value within its class.
    """
    out = np.zeros_like(values, dtype=float)
    for c in np.unique(labels):
        m = labels == c
        ranks = values[m].argsort().argsort().astype(float)
        out[m] = ranks / max(len(ranks) - 1, 1)
    return out


# Strata analysis

def _gap_ci(ms, mr, cfg, *, alpha: "float | None" = None) -> "dict[str, Any]":
    """
    Computes BG-Gap and bootstrap confidence interval.
    - Pre: `ms` and `mr` are aligned correctness arrays.
        `cfg` contains bootstrap settings.
        `alpha` is None or a float in (0, 1).
    - Post: Returns dictionary with BG-Gap, CI bounds, and sample count.
        Returns null gap and zero count when inputs are empty.
    """
    if len(ms) == 0:
        return {"bg_gap": None, "ci_lo": None, "ci_hi": None, "n": 0}
    d = ms.astype(np.int8) - mr.astype(np.int8)
    b = paired_bootstrap(
        d,
        n=cfg.bootstrap_n,
        seed=cfg.bootstrap_seed,
        alpha=cfg.ci_alpha if alpha is None else alpha,
    )
    return {
        "bg_gap": round(b["point"], 6),
        "ci_lo": round(b["lo"], 6),
        "ci_hi": round(b["hi"], 6),
        "n": int(len(ms)),
    }


def per_class_gap(join, cfg) -> "dict[str, Any]":
    """
    Computes per-class BG-Gap with multiple-test correction.
    - Pre: `join` contains aligned paired results and class labels.
        `cfg` contains alpha and correction settings.
    - Post: Returns dictionary with per-class BG-Gap and significance flags.
    """
    m_tests = len(CANONICAL_CLASSES)
    bonf = cfg.ci_alpha / m_tests
    per: dict[str, Any] = {}
    pvals, order = [], []

    for c, name in enumerate(CANONICAL_CLASSES):
        m = join.labels == c
        alpha = bonf if cfg.per_class_correction == "bonferroni" else cfg.ci_alpha
        res = _gap_ci(join.ms[m], join.mr[m], cfg, alpha=alpha)
        res["class_name"] = name
        if m.sum():
            sub = type(join)(
                keys=tuple(np.asarray(join.keys)[m]),
                ms=join.ms[m],
                mr=join.mr[m],
                labels=join.labels[m],
                losses={},
            )
            mc = mcnemar_test(sub)
            res["mcnemar_p"] = mc["p"]
            pvals.append(mc["p"])
            order.append(str(c))
        per[str(c)] = res

    if cfg.per_class_correction == "bonferroni":
        for k in order:
            per[k]["significant"] = bool(per[k]["mcnemar_p"] < bonf)
    else:
        for k, rej in zip(order, bh_fdr(pvals, q=cfg.ci_alpha)):
            per[k]["significant"] = bool(rej)

    return {
        "_meta": {
            "correction": cfg.per_class_correction,
            "n_tests": m_tests,
            "alpha": cfg.ci_alpha,
            "effective_alpha": bonf if cfg.per_class_correction == "bonferroni" else cfg.ci_alpha,
        },
        "classes": per,
    }


def top_minus_bottom(join, binning: Binning, cfg) -> "dict[str, Any] | None":
    """
    Computes top-minus-bottom bin BG-Gap difference.
    - Pre: `join` contains aligned paired results.
        `binning` has valid bin indices.
        `cfg` contains bootstrap settings.
    - Post: Returns dictionary with difference, CI, and bin sample counts.
        Returns None when top or bottom bin is empty.
    """
    top, bottom = binning.n_actual - 1, 0
    mt, mb = binning.bins == top, binning.bins == bottom
    if not (mt.sum() and mb.sum()):
        return None
    dt = (join.ms[mt].astype(np.int8) - join.mr[mt].astype(np.int8)).astype(float)
    db = (join.ms[mb].astype(np.int8) - join.mr[mb].astype(np.int8)).astype(float)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    diffs = np.empty(cfg.bootstrap_n)
    for i in range(cfg.bootstrap_n):
        diffs[i] = (
            dt[rng.integers(0, len(dt), len(dt))].mean()
            - db[rng.integers(0, len(db), len(db))].mean()
        )
    a = cfg.ci_alpha
    return {
        "top_minus_bottom": round(float(dt.mean() - db.mean()), 6),
        "ci_lo": round(float(np.percentile(diffs, 100 * a / 2)), 6),
        "ci_hi": round(float(np.percentile(diffs, 100 * (1 - a / 2))), 6),
        "top_bin": top,
        "bottom_bin": bottom,
        "n_top": int(mt.sum()),
        "n_bottom": int(mb.sum()),
    }


def gap_by_attribute(join, attrs, attr: str, cfg, *, n_bins: "int | None" = None,
                     within_class: "bool | None" = None) -> "dict[str, Any]":
    """
    Computes BG-Gap stratified by one mask attribute.
    - Pre: `join` contains aligned paired results.
        `attrs` maps fg_id to attribute values.
        `attr` is a supported attribute key.
        `cfg` contains stratification settings.
        `n_bins` and `within_class` are None or override values.
    - Post: Returns dictionary with bin-level gaps, metadata,
        and top-minus-bottom trend summary.
    """
    n_bins = cfg.strat_n_bins if n_bins is None else n_bins
    within_class = cfg.strat_within_class if within_class is None else within_class

    raw = np.array([attrs.get(k, {}).get(attr, np.nan) for k in join.keys])
    valid = ~np.isnan(raw)
    sub = type(join)(
        keys=tuple(np.asarray(join.keys)[valid]),
        ms=join.ms[valid],
        mr=join.mr[valid],
        labels=join.labels[valid],
        losses={},
    )
    vals = raw[valid]

    binvals = within_class_percentile(vals, sub.labels) if within_class else vals
    binning = quantile_bins(binvals, n_bins, within_class=within_class)

    bins_out = {}
    for b in range(binning.n_actual):
        m = binning.bins == b
        res = _gap_ci(sub.ms[m], sub.mr[m], cfg)
        res["mean_attr"] = round(float(vals[m].mean()), 6) if m.sum() else None
        res["underpowered"] = bool(res["n"] < cfg.strat_min_bin_n)
        bins_out[str(b)] = res

    return {
        "_meta": {
            "attr": attr,
            "n_without_mask": int((~valid).sum()),
            "min_bin_n": cfg.strat_min_bin_n,
            **binning.meta(),
        },
        "bins": bins_out,
        "trend_top_minus_bottom": top_minus_bottom(sub, binning, cfg),
    }


def stratify_report(spec: ModelSpec, cfg) -> "dict[str, Any]":
    """
    Builds full stratified BG-Gap report for one model.
    - Pre: `spec` is a valid ModelSpec object.
        `cfg` contains stratification and bootstrap settings.
    - Post: Returns report dictionary with mask coverage, join losses,
        per-class results, and attribute-based stratification outputs.
    """
    join = join_variants(load_variant(spec, "mixed_same"), load_variant(spec, "mixed_rand"))
    attrs, coverage = load_mask_attributes()

    out: dict[str, Any] = {
        "schema": SCHEMA, "model_id": spec.model_id, "safe_id": spec.safe_id,
        "mask_coverage": coverage,
        "join_losses": join.losses,
        "per_class": per_class_gap(join, cfg),
        "by_attribute": {},
    }
    for attr in ATTRS:
        out["by_attribute"][attr] = {
            "global": gap_by_attribute(join, attrs, attr, cfg, within_class=False),
            "within_class": gap_by_attribute(join, attrs, attr, cfg, within_class=True),
            "bins_sensitivity": {
                str(k): (
                    gap_by_attribute(join, attrs, attr, cfg, n_bins=k) or {}
                ).get("trend_top_minus_bottom")
                for k in cfg.strat_bins_sensitivity
            },
        }
    return out


def write_stratify(spec: ModelSpec, report: "dict[str, Any]"):
    """
    Writes stratification report to disk.
    - Pre: `spec` is a valid ModelSpec object.
        `report` is a JSON-serializable dictionary.
    - Post: Writes report JSON in metrics directory and returns output path.
    """
    d = paths.metrics_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"stratify__{spec.safe_id}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    from .config import load_config
    from .registry import all_models

    cfg = load_config()
    for spec in all_models():
        try:
            rep = stratify_report(spec, cfg)
        except FileNotFoundError:
            continue
        sz = rep["by_attribute"]["fg_area_frac"]["global"]
        t = sz["trend_top_minus_bottom"]
        print(f"{spec.safe_id:22s} size trend (largest-smallest) "
              f"{t['top_minus_bottom']*100:+.2f} pp [{t['ci_lo']*100:+.2f},{t['ci_hi']*100:+.2f}]  "
              f"bins={sz['_meta']['n_bins_actual']}  masks={rep['mask_coverage']['n_found']}")
        print(f"  -> {write_stratify(spec, rep)}")