"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

N_CLASSES = 9
CHANCE = 1.0 / N_CLASSES


# AUC metrics

def auc(score, label) -> float:
    """
    Computes binary AUC from scores and labels.
    - Pre: `score` and `label` are aligned sequences.
        `label` is binary or boolean-like.
    - Post: Returns AUC as float.
        Returns NaN if one class is missing.
    """
    from scipy.stats import rankdata

    s, y = np.asarray(score, float), np.asarray(label).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_boot(score, label, *, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """
    Computes bootstrap confidence interval for AUC.
    - Pre: `score` and `label` are aligned sequences.
        `n` is a positive integer.
        `seed` is an integer.
        `alpha` is a float in (0, 1).
    - Post: Returns dictionary with AUC, confidence interval,
        class counts, and bootstrap settings.
    """
    s, y = np.asarray(score, float), np.asarray(label).astype(bool)
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(s), len(s))
        vals[i] = auc(s[idx], y[idx])
    ok = np.isfinite(vals)
    return {
        "auc": round(auc(s, y), 4),
        "ci_lo": round(float(np.percentile(vals[ok], 100 * alpha / 2)), 4),
        "ci_hi": round(float(np.percentile(vals[ok], 100 * (1 - alpha / 2))), 4),
        "n_pos": int(y.sum()), "n_neg": int((~y).sum()), "n_boot": n, "seed": seed,
    }


# Attention statistics

def attention_stats(attn_bhgg: torch.Tensor, patch_mask: torch.Tensor):
    """
    Computes per-image attention spread statistics.
    - Pre: `attn_bhgg` has shape `(B, H, grid, grid)`.
        `patch_mask` has shape `(B, grid, grid)` and boolean-like values.
    - Post: Returns `(off_fg_mass, entropy)` numpy arrays per image.
    """
    a = attn_bhgg.mean(1)
    a = a.reshape(a.shape[0], -1)
    a = a / a.sum(1, keepdim=True).clamp_min(1e-12)
    pm = patch_mask.reshape(patch_mask.shape[0], -1).to(a.dtype)
    fg_mass = (a * pm).sum(1)
    entropy = -(a * torch.log(a.clamp_min(1e-12))).sum(1)
    return (1.0 - fg_mass).cpu().numpy(), entropy.cpu().numpy()


def uniform_entropy(n_patches: int = 196) -> float:
    """
    Computes entropy of a uniform patch distribution.
    - Pre: `n_patches` is a positive integer.
    - Post: Returns entropy ceiling value as float.
    """
    return float(np.log(n_patches))


# Reliance labels

def reliance_label(join, *, negative_class: str = "all"):
    """
    Builds reliance labels from paired outcome differences.
    - Pre: `join` provides `diff` vector with values in {-1, 0, +1}.
        `negative_class` is `all` or `discordant`.
    - Post: Returns `(keep_mask, label)` arrays for probe evaluation.
        Raises ValueError for unknown `negative_class`.
    """
    d = join.diff
    pos = d == 1
    if negative_class == "all":
        return np.ones(len(d), dtype=bool), pos
    if negative_class == "discordant":
        keep = d != 0
        return keep, pos[keep]
    raise ValueError(f"unknown negative_class {negative_class!r}")


# Decodability

def fg_disjoint_split(fg_ids, *, test_frac: float = 0.5, seed: int = 0):
    """
    Splits data by unique foreground ids.
    - Pre: `fg_ids` is an iterable of foreground id values.
        `test_frac` is a float in (0, 1).
        `seed` is an integer.
    - Post: Returns `(train_mask, test_mask)` boolean arrays.
        No foreground id appears in both masks.
    """
    uniq = sorted(set(fg_ids))
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_ids = {uniq[i] for i in rng.permutation(len(uniq))[:n_test]}
    test = np.array([f in test_ids for f in fg_ids])
    return ~test, test


def _fit_head(X, y, cfg, seed: int):
    """
    Fits one linear probe head.
    - Pre: `X` and `y` are torch tensors with aligned rows.
        `cfg` contains probe optimizer settings.
        `seed` is an integer.
    - Post: Returns trained linear head in eval mode.
    """
    torch.manual_seed(seed)
    lin = torch.nn.Linear(X.shape[1], N_CLASSES)
    opt = torch.optim.Adam(
        lin.parameters(),
        lr=cfg.probe_lr,
        weight_decay=cfg.probe_weight_decay,
    )
    for _ in range(cfg.probe_steps):
        opt.zero_grad()
        F.cross_entropy(lin(X), y).backward()
        opt.step()
    return lin.eval()


def decodability(features: np.ndarray, labels, fg_ids, cfg) -> "dict[str, Any]":
    """
    Estimates 9-way linear decodability from frozen features.
    - Pre: `features`, `labels`, and `fg_ids` are aligned arrays.
        `cfg` contains probe split and optimization settings.
    - Post: Returns dictionary with probe accuracy statistics,
        chance baseline, and split sizes.
    """
    labels = np.asarray(labels)
    tr, te = fg_disjoint_split(fg_ids, test_frac=cfg.probe_test_frac, seed=cfg.seed)

    X = torch.tensor(features, dtype=torch.float32)
    mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True).clamp_min(1e-6)
    Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ytr = torch.tensor(labels[tr], dtype=torch.long)

    accs = []
    for s in range(cfg.probe_seeds):
        # The head is trained, so fitting must stay outside no_grad; only scoring the
        # held-out slice is done without gradients.
        head = _fit_head(Ztr, ytr, cfg, seed=cfg.seed + s)
        with torch.no_grad():
            pred = head(Zte).argmax(1).numpy()
        accs.append(float((pred == labels[te]).mean()))
    return {
        "accuracy": round(float(np.mean(accs)), 6),
        "accuracy_sd": round(float(np.std(accs)), 6),
        # Chance sits next to the number: an accuracy of 0.30 means nothing until the
        # reader knows the nine-class baseline is 0.111.
        "chance": round(CHANCE, 6),
        "above_chance": round(float(np.mean(accs)) - CHANCE, 6),
        "n_train": int(tr.sum()), "n_test": int(te.sum()), "seeds": cfg.probe_seeds,
    }


# Reports

def attention_probe_report(entropy, off_fg_mass, join, cfg) -> "dict[str, Any]":
    """
    Builds attention-based reliance probe report.
    - Pre: `entropy` and `off_fg_mass` are score arrays aligned with `join` samples.
        `join` contains paired outcome differences.
        `cfg` contains AUC and reliance settings.
    - Post: Returns report dictionary with AUC statistics,
        predicted/observed signs, and hypothesis support flags.
    """
    keep, label = reliance_label(join, negative_class=cfg.reliance_negative_class)
    out = {
        "negative_class": cfg.reliance_negative_class,
        "uniform_entropy": round(uniform_entropy(), 4),
        "hypothesis": "higher attention spread predicts background reliance",
        "predicted_sign": +1,
    }
    for name, score in (("entropy", entropy), ("off_fg_mass", off_fg_mass)):
        out[name] = auc_boot(
            np.asarray(score)[keep],
            label,
            n=cfg.auc_boot_n,
            seed=cfg.bootstrap_seed,
            alpha=cfg.ci_alpha,
        )
        out[name]["observed_sign"] = +1 if out[name]["auc"] > 0.5 else -1
        out[name]["supports_hypothesis"] = bool(out[name]["ci_lo"] > 0.5)
    out["claim_type"] = "diagnostic (correlational), not mechanistic"
    return out


def readout_probe_report(by_beta: "dict[float, dict]", cfg) -> "dict[str, Any]":
    """
    Builds readout probe report across beta values.
    - Pre: `by_beta` maps beta values to probe result dictionaries.
        `cfg` is a valid config object.
    - Post: Returns report dictionary with foreground/background decodability trends,
        chance baseline, and hypothesis direction checks.
    """
    betas = sorted(by_beta)
    bg = [by_beta[b]["bg"]["accuracy"] for b in betas]
    fg = [by_beta[b]["fg"]["accuracy"] for b in betas]
    observed = +1 if bg[-1] > bg[0] else -1
    return {
        "betas": betas, "chance": round(CHANCE, 6),
        "bg_decodability": bg, "fg_decodability": fg,
        "hypothesis": "attenuating position makes the BACKGROUND more decodable "
                      "(more shortcut material available)",
        "predicted_sign": +1,
        "observed_sign": observed,
        "supports_hypothesis": bool(observed == +1),
    }