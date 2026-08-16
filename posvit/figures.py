"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import paths  # noqa: E402
from .registry import ModelSpec, all_models, by_id  # noqa: E402

OKABE_ITO = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "purple": "#CC79A7",
    "vermillion": "#D55E00", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#999999",
}
MODEL_COLOR = {
    "ape": OKABE_ITO["blue"], "axial": OKABE_ITO["orange"], "axial_ape": OKABE_ITO["green"],
    "mixed": OKABE_ITO["purple"], "mixed_ape": OKABE_ITO["vermillion"],
}
PRETTY = {"ape": "APE", "axial": "RoPE-Axial", "axial_ape": "RoPE-Axial+APE",
          "mixed": "RoPE-Mixed", "mixed_ape": "RoPE-Mixed+APE"}
SCALE_LS = {"vit_s": "-", "vit_b": "--"}

RC = {
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
}


def apply_style() -> None:
    """
    Applies standard plotting style.
    - Pre: Matplotlib is available.
    - Post: Updates matplotlib rcParams with project style values.
    """
    plt.rcParams.update(RC)


def label_for(spec: ModelSpec) -> str:
    """
    Builds display label for one model.
    - Pre: `spec` is a valid ModelSpec object.
    - Post: Returns a formatted model label string.
    """
    name = PRETTY.get(spec.key, spec.key)
    return f"{name} (ViT-{'S' if spec.scale == 'vit_s' else 'B'}/16)"


def save_figure(fig, name: str, *, formats=("png", "pdf")):
    """
    Saves a figure to the project figures directory.
    - Pre: `fig` is a valid matplotlib figure.
        `name` is a non-empty output basename.
        `formats` is an iterable of file extensions.
    - Post: Writes figure files in requested formats and closes the figure.
        Returns list of written file paths.
    """
    out_dir = paths.figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        meta = {"CreationDate": None} if ext == "pdf" else {"Software": None}
        fig.savefig(p, metadata=meta)
        written.append(p)
    plt.close(fig)
    return written


def figure_forest_equivalence(bggap: "dict[str, Any]", equiv: "dict[str, Any]", cfg):
    """
    Builds forest figure for intact BG-Gap and equivalence margin.
    - Pre: `bggap` and `equiv` contain required metric fields.
        `cfg` is a valid config object.
    - Post: Returns a matplotlib figure object.
    """
    apply_style()
    specs = [s for s in all_models() if s.safe_id in bggap]
    specs.sort(key=lambda s: (s.scale, s.key))

    fig, ax = plt.subplots(figsize=(7.0, 0.45 * len(specs) + 2.0))
    for i, spec in enumerate(specs):
        r = bggap[spec.safe_id]
        gap, lo, hi = r["bg_gap"] * 100, r["ci_lo"] * 100, r["ci_hi"] * 100
        ax.errorbar(
            gap, i, xerr=[[gap - lo], [hi - gap]], fmt="o", capsize=3,
            color=MODEL_COLOR.get(spec.key, OKABE_ITO["grey"]),
            markerfacecolor="white" if spec.scale == "vit_b" else None,
            markersize=7, linewidth=1.6,
        )
        ax.text(hi + 0.25, i, f"{gap:.2f}", va="center", fontsize=9, color="#444")

    m = equiv["declared_margin_pp"]
    centre = float(sum(bggap[s.safe_id]["bg_gap"] for s in specs) / len(specs)) * 100
    ax.axvspan(centre - m, centre + m, color=OKABE_ITO["grey"], alpha=0.13, zorder=0,
               label=f"±{m:.1f} pp equivalence margin")

    ax.set_yticks(range(len(specs)))
    ax.set_yticklabels([label_for(s) for s in specs])
    ax.set_xlabel("Intact BG-Gap (pp), 95% CI")
    ax.set_title(
        f"The encoding scheme is irrelevant: {equiv['n_equivalent']}/{equiv['n_pairs']} pairs "
        f"TOST-equivalent at ≤{m:.1f} pp (MDE {equiv['mde_pp']:.2f} pp)"
    )
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def figure_dose_response(dose: "dict[str, Any]", cfg):
    """
    Builds dose-response figure for positional removal analysis.
    - Pre: `dose` contains required dose-response fields.
        `cfg` is a valid config object.
    - Post: Returns a matplotlib figure object.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    floor = dose["acc_floor"]

    for ax, role, title in (
        (axes[0], "functional", "Functional signal (the one the model uses)"),
        (axes[1], "vestigial", "Vestigial signal (the unused copy)"),
    ):
        plotted = 0
        for safe_id, r in sorted(dose["models"].items()):
            if r["knob_role"] != role:
                continue
            spec = by_id(r["model_id"])
            colour = MODEL_COLOR.get(spec.key, OKABE_ITO["grey"])
            used = set(r["betas_used"])
            for beta, gap in zip(r["betas_used"], r["bg_gap_by_beta"]):
                ax.plot(1 - beta, gap * 100, "o", color=colour, markersize=6, zorder=3)
            ax.plot(
                [1 - b for b in r["betas_used"]],
                [g * 100 for g in r["bg_gap_by_beta"]],
                linestyle=SCALE_LS[spec.scale],
                color=colour,
                linewidth=1.6,
                label=label_for(spec),
                zorder=2,
            )
            for beta in r["guard"]["betas_excluded"]:
                gap = r.get("bg_gap_all", {}).get(f"{beta:.2f}")
                if gap is not None and beta not in used:
                    ax.plot(
                        1 - beta,
                        gap * 100,
                        "o",
                        markerfacecolor="none",
                        markeredgecolor=colour,
                        markersize=6,
                        zorder=3,
                    )
            plotted += 1
        ax.set_title(title)
        ax.set_xlabel(r"positional removal  $(1-\beta)$")
        if plotted:
            ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("BG-Gap (pp)")
    fig.suptitle(
        f"Attenuating the functional positional signal raises background reliance "
        f"({dose['n_positive_slope_ci_excludes_zero']}/{dose['n_models']} slopes, CI > 0); "
        f"open markers = below the {floor:.2f} accuracy floor, excluded",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def figure_object_size(
    stratify_by_model: "dict[str, dict]",
    cfg,
    *,
    attr: str = "fg_area_frac",
    binning: str = "global",
):
    """
    Builds object-size stratification figure.
    - Pre: `stratify_by_model` maps model ids to stratification reports.
        `cfg` is a valid config object.
        `attr` is a valid stratification attribute key.
        `binning` is a valid binning mode key.
    - Post: Returns a matplotlib figure object.
        Underpowered bins are shown with open markers.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for safe_id, rep in sorted(stratify_by_model.items()):
        spec = by_id(rep["model_id"])
        if spec.scale != "vit_s":
            continue
        block = rep["by_attribute"][attr][binning]
        bins = [block["bins"][str(b)] for b in range(block["_meta"]["n_bins_actual"])]
        x = [b["mean_attr"] for b in bins]
        y = [b["bg_gap"] * 100 for b in bins]
        colour = MODEL_COLOR.get(spec.key, OKABE_ITO["grey"])
        ax.plot(x, y, "-", color=colour, linewidth=1.6, label=label_for(spec), zorder=2)
        for b, xi, yi in zip(bins, x, y):
            ax.plot(
                xi,
                yi,
                "o",
                markersize=6,
                color=colour,
                zorder=3,
                markerfacecolor="none" if b["underpowered"] else colour,
            )
        ax.fill_between(
            x,
            [b["ci_lo"] * 100 for b in bins],
            [b["ci_hi"] * 100 for b in bins],
            color=colour,
            alpha=0.10,
            zorder=1,
        )
    ax.set_xlabel("foreground area fraction (bin mean)")
    ax.set_ylabel("BG-Gap (pp)")
    ax.set_title("Object size governs background reliance - identically across encodings")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main(cfg=None) -> "list[Path]":
    """
    Regenerates every figure whose metrics artifact is present.

    Figures are built from the committed metrics rather than from raw predictions, so a
    plotted number is always also a citable one and the two cannot disagree.
    - Pre: `cfg` is None or a Config object.
    - Post: Returns the list of written figure paths, which is empty when no metrics
        artifact exists yet.
    """
    from .config import load_config

    cfg = cfg if cfg is not None else load_config()
    md = paths.metrics_dir()
    written: list[Path] = []

    bggap_p = md / "bggap.json"
    equiv_p = md / "equivalence.json"
    dose_p = md / "dose_response.json"
    mech_p = md / "mechanisms.json"

    if bggap_p.is_file() and equiv_p.is_file():
        bggap = json.loads(bggap_p.read_text(encoding="utf-8"))
        equiv = json.loads(equiv_p.read_text(encoding="utf-8"))
        written += save_figure(
            figure_forest_equivalence(bggap, equiv, cfg),
            "02_forest_powered_null",
        )
    if dose_p.is_file():
        dose = json.loads(dose_p.read_text(encoding="utf-8"))
        written += save_figure(figure_dose_response(dose, cfg), "03_dose_response")

    strat = {}
    for path in sorted(md.glob("stratify__*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        strat[rec["safe_id"]] = rec
    if strat:
        written += save_figure(figure_object_size(strat, cfg), "06_object_size")

    if mech_p.is_file():
        table = json.loads(mech_p.read_text(encoding="utf-8"))
        written += save_figure(figure_mechanisms(table, cfg), "13_mechanisms")

    return written


if __name__ == "__main__":
    for _path in main():
        print(f"wrote {_path}")
