"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any
import numpy as np
from . import paths
from .evaluate import is_complete, prediction_paths, read_records
from .registry import ModelSpec, all_models

SCHEMA = "posvit.bggap/1"


def load_variant(spec: ModelSpec, variant: str, *, require_complete: bool = True):
    """
    Reads records for one evaluated variant.
    - Pre: `spec` is a ModelSpec object.
        `variant` is a valid variant name.
        `require_complete` is True or False.
    - Post: Returns a list of record dictionaries from the variant file.
        Raises FileNotFoundError when completion is required and missing.
    """
    rec_path, _ = prediction_paths(spec, variant)
    if require_complete and not is_complete(spec, variant):
        raise FileNotFoundError(
            f"{spec.safe_id}/{variant}: no complete run at {rec_path}.\n"
            f"Run `posvit evaluate --id {spec.model_id}` first. A missing or incomplete "
            "sidecar means the job did not finish, so its records must not be reduced."
        )
    return read_records(rec_path)


def _correct_by_fg(records):
    """
    Builds correctness mapping by foreground id.

    A duplicate `fg_id` would silently overwrite its predecessor in the mapping, so the
    count is returned alongside it rather than discarded.
    - Pre: `records` is an iterable of prediction records.
    - Post: Returns `(mapping, n_duplicates, sample_duplicates)`.
        `mapping` maps `fg_id` to boolean correctness.
    """
    counts = Counter(r["fg_id"] for r in records)
    dups = [k for k, v in counts.items() if v > 1]
    mapping = {r["fg_id"]: bool(r["pred9"] == r["label9"]) for r in records}
    return mapping, len(dups), sorted(dups)[:10]


@dataclass(frozen=True, eq=False)
class PairedJoin:
    """
    Holds mixed-same and mixed-rand outcomes aligned on foreground identity.

    `labels` carries the IN-9 class per foreground so stratified analyses can subset the
    single join rather than re-deriving one. Equality is disabled because the numpy
    fields make a generated `__eq__` ambiguous.
    """
    keys: "tuple[str, ...]"
    ms: np.ndarray
    mr: np.ndarray
    losses: "dict[str, Any]"
    labels: "np.ndarray | None" = None

    @property
    def n(self) -> int:
        """
        Returns the number of paired foreground keys.
        - Pre: `self.keys` is initialized.
        - Post: Returns the integer length of `self.keys`.
        """
        return len(self.keys)

    @property
    def diff(self) -> np.ndarray:
        """
        Computes per-foreground accuracy differences.
        - Pre: `self.ms` and `self.mr` are boolean arrays of equal length.
        - Post: Returns an int8 array with values in {-1, 0, +1}.
        """
        return self.ms.astype(np.int8) - self.mr.astype(np.int8)


def join_variants(ms_records, mr_records, *, strict: bool = True) -> PairedJoin:
    """
    Aligns mixed-same and mixed-rand records by foreground id.
    - Pre: `ms_records` and `mr_records` are iterables of record dictionaries.
        `strict` is True or False.
    - Post: Returns a PairedJoin object with aligned arrays and loss report.
        Raises ValueError when `strict` is True and pairing is lossy.
    """
    ms_map, n_dup_ms, dup_ms = _correct_by_fg(ms_records)
    mr_map, n_dup_mr, dup_mr = _correct_by_fg(mr_records)
    label_map = {r["fg_id"]: int(r["label9"]) for r in ms_records}

    only_ms = sorted(set(ms_map) - set(mr_map))
    only_mr = sorted(set(mr_map) - set(ms_map))
    keys = tuple(sorted(set(ms_map) & set(mr_map)))

    losses = {
        "n_ms_records": len(ms_records), "n_mr_records": len(mr_records),
        "n_ms_keys": len(ms_map), "n_mr_keys": len(mr_map),
        "n_dup_ms": n_dup_ms, "n_dup_mr": n_dup_mr,
        "sample_dup_ms": dup_ms, "sample_dup_mr": dup_mr,
        "n_only_ms": len(only_ms), "n_only_mr": len(only_mr),
        "sample_only_ms": only_ms[:10], "sample_only_mr": only_mr[:10],
        "n_paired": len(keys),
        "lossless": not (only_ms or only_mr or n_dup_ms or n_dup_mr),
    }

    if strict and not losses["lossless"]:
        raise ValueError(
            "BG-Gap join is lossy, so check C4's pairing precondition no longer holds for "
            f"this data:\n{json.dumps(losses, indent=2)}\n"
            "Re-run acquisition and evaluation, or pass strict=False deliberately."
        )

    return PairedJoin(
        keys=keys,
        ms=np.array([ms_map[k] for k in keys], dtype=bool),
        mr=np.array([mr_map[k] for k in keys], dtype=bool),
        losses=losses,
        labels=np.array([label_map[k] for k in keys], dtype=int),
    )


def bg_gap(join: PairedJoin) -> float:
    """
    Computes BG-Gap from paired results.
    - Pre: `join` is a valid PairedJoin object.
    - Post: Returns the mean of `join.diff` as a float.
    """
    return float(join.diff.mean())


def paired_bootstrap(diff: np.ndarray, *, n: int = 10000, seed: int = 0, alpha: float = 0.05):
    """
    Computes bootstrap confidence interval for BG-Gap.
    - Pre: `diff` is a numeric array.
        `n` is a positive integer.
        `seed` is an integer.
        `alpha` is a float in (0, 1).
    - Post: Returns a dictionary with point estimate, interval, and standard error.
    """
    rng = np.random.default_rng(seed)
    d = diff.astype(np.float64)
    N = len(d)
    gaps = np.empty(n)
    for b in range(n):
        gaps[b] = d[rng.integers(0, N, N)].mean()
    return {
        "point": float(d.mean()),
        "lo": float(np.percentile(gaps, 100 * alpha / 2)),
        "hi": float(np.percentile(gaps, 100 * (1 - alpha / 2))),
        "se": float(gaps.std(ddof=1)),
        "n_boot": n, "seed": seed, "alpha": alpha,
    }


def mcnemar_test(join: PairedJoin) -> "dict[str, Any]":
    """
    Runs McNemar test on paired correctness arrays.
    - Pre: `join` is a valid PairedJoin object.
    - Post: Returns a dictionary with discordant counts, statistic, p-value, and method.
    """
    from statsmodels.stats.contingency_tables import mcnemar as mctest

    ms, mr = join.ms, join.mr
    b = int(np.sum(ms & ~mr))
    c = int(np.sum(~ms & mr))
    exact = (b + c) < 25
    table = [[int(np.sum(ms & mr)), b], [c, int(np.sum(~ms & ~mr))]]
    res = mctest(table, exact=exact)
    return {
        "b_only_ms": b, "c_only_mr": c, "n_discordant": b + c,
        "stat": float(res.statistic), "p": float(res.pvalue),
        "method": "exact_binomial" if exact else "chi2_continuity_corrected",
    }


def bggap_report(
    spec: ModelSpec,
    cfg,
    *,
    ms_variant: str = "mixed_same",
    mr_variant: str = "mixed_rand",
    strict: bool = True,
) -> "dict[str, Any]":
    """
    Builds BG-Gap report for one model spec.
    - Pre: `spec` is a ModelSpec object.
        `cfg` contains bootstrap and CI settings.
        Variant names and `strict` flag are valid.
    - Post: Returns a report dictionary with BG-Gap, CI, McNemar, and pairing diagnostics.
        Raises AssertionError if internal consistency check fails.
    """
    join = join_variants(
        load_variant(spec, ms_variant), load_variant(spec, mr_variant), strict=strict
    )
    gap = bg_gap(join)
    boot = paired_bootstrap(
        join.diff, n=cfg.bootstrap_n, seed=cfg.bootstrap_seed, alpha=cfg.ci_alpha
    )
    mc = mcnemar_test(join)

    expected = (mc["b_only_ms"] - mc["c_only_mr"]) / join.n
    if abs(gap - expected) > 1e-12:
        raise AssertionError(
            f"{spec.model_id}: BG-Gap {gap!r} disagrees with (b-c)/N {expected!r}. "
            "The join and the McNemar cells cannot both be right."
        )

    return {
        "schema": SCHEMA,
        "model_id": spec.model_id,
        "safe_id": spec.safe_id,
        "bg_gap": round(gap, 6),
        "ci_lo": round(boot["lo"], 6),
        "ci_hi": round(boot["hi"], 6),
        "se": round(boot["se"], 6),
        "n_paired": join.n,
        "acc_ms": round(float(join.ms.mean()), 6),
        "acc_mr": round(float(join.mr.mean()), 6),
        "mcnemar": mc,
        "bootstrap_n": cfg.bootstrap_n,
        "bootstrap_seed": cfg.bootstrap_seed,
        "ci_alpha": cfg.ci_alpha,
        "join_losses": join.losses,
        "input_ms": prediction_paths(spec, ms_variant)[0].name,
        "input_mr": prediction_paths(spec, mr_variant)[0].name,
    }


def write_bggap(reports: "dict[str, dict[str, Any]]") -> "Any":
    """
    Writes BG-Gap reports to disk.
    - Pre: `reports` is a dictionary keyed by model safe id.
    - Post: Writes `bggap.json` in the metrics directory and returns its path.
    """
    out_dir = paths.metrics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "bggap.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    from .config import load_config
    cfg = load_config()
    reports: dict[str, dict[str, Any]] = {}
    for spec in all_models():
        try:
            rep = bggap_report(spec, cfg)
        except FileNotFoundError as exc:
            print(f"SKIP {spec.model_id}: {str(exc).splitlines()[0]}")
            continue
        reports[spec.safe_id] = rep
        print(
            f"{spec.safe_id:22s} BG-Gap={rep['bg_gap']:.4f} "
            f"[{rep['ci_lo']:.4f}, {rep['ci_hi']:.4f}]  N={rep['n_paired']}  "
            f"McNemar p={rep['mcnemar']['p']:.3g}"
        )
    print(f"\nWrote {write_bggap(reports)}  ({len(reports)} models)")