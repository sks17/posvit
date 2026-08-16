"""
Read-only structural verification of the IN-9 dataset
- Styled after the Backgrounds-Challenge paper's "C1-C7" checks (Xia et al., 2021)
"""

from __future__ import annotations
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from PIL import Image
from .. import paths
from ..config import load_config
from . import CANONICAL_CLASSES, load_in9_map
_FG_RE = re.compile(r"^fg_n\d+_\d+_bg_") # filename pattern
_COMPOSITE_VARIANTS = ("mixed_same", "mixed_rand", "mixed_next")
_IMG_EXT = (".jpeg", ".jpg", ".png")


def _val_dir(variant: str) -> Path:
    """
    Returns the validation directory for a variant.
    - Pre: `variant` is a non-empty string.
    - Post: Returns a Path object for `<in9_bg_root>/<variant>/val`.
    """
    return paths.in9_bg_root() / variant / "val"


def _iter_images(variant: str):
    """
    Iterates over image files in one variant.
    - Pre: `variant` is a non-empty string.
    - Post: Yields tuples `(class_name, filename, image_path)` for valid image files.
    """
    root = _val_dir(variant)
    for cls in CANONICAL_CLASSES:
        d = root / cls
        if not d.is_dir():
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(_IMG_EXT):
                yield cls, name, d / name


def _basenames(variant: str) -> set[str]:
    """
    Collects image basenames for one variant.
    - Pre: `variant` is a non-empty string.
    - Post: Returns a set of image filenames.
    """
    return {name for _, name, _ in _iter_images(variant)}


def check_c1(variants: list[str]) -> dict[str, Any]:
    """
    Checks that variant directories exist.
    - Pre: `variants` is a list of variant names.
    - Post: Returns a report dictionary with pass flag and missing variants.
    """
    missing = [v for v in variants if not _val_dir(v).is_dir()]
    return {"ok": not missing, "missing_variants": missing}


def check_c2(variants: list[str]) -> dict[str, Any]:
    """
    Checks class-directory names for each variant.
    - Pre: `variants` is a list of variant names.
    - Post: Returns a report dictionary with per-variant class listings and pass flag.
    """
    per, ok = {}, True
    want = sorted(CANONICAL_CLASSES)
    for v in variants:
        vd = _val_dir(v)
        if not vd.is_dir():
            continue
        found = sorted(p.name for p in vd.iterdir() if p.is_dir())
        per[v] = found
        if found != want:
            ok = False
    return {"ok": ok, "per_variant_classes": per}


def check_c3(variants: list[str]) -> dict[str, Any]:
    """
    Checks that images are readable and have expected shape/mode.
    - Pre: `variants` is a list of variant names.
    - Post: Returns a report dictionary with counts, samples, and pass flag.
    """
    corrupt, nonconforming, modes, n = [], [], set(), 0
    for v in variants:
        for _, _, path in _iter_images(v):
            n += 1
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    w, h, mode = im.width, im.height, im.mode
            except Exception as exc:
                corrupt.append({"path": str(path), "error": str(exc)})
                continue
            modes.add((w, h, mode))
            if (w, h, mode) != (224, 224, "RGB"):
                nonconforming.append({"path": str(path), "dims": [w, h, mode]})
    return {
        "ok": not corrupt and not nonconforming,
        "n_images": n,
        "n_corrupt": len(corrupt), "corrupt": corrupt[:20],
        "n_nonconforming": len(nonconforming), "nonconforming": nonconforming[:20],
        "observed_modes": sorted(str(m) for m in modes),
    }


def check_c4() -> dict[str, Any]:
    """
    Checks paired basenames between mixed_same and mixed_rand.
    - Pre: Dataset variant directories may or may not exist.
    - Post: Returns a report dictionary with symmetric-difference details and pass flag.
    """
    if not _val_dir("mixed_same").is_dir() or not _val_dir("mixed_rand").is_dir():
        return {"ok": False, "error": "mixed_same and/or mixed_rand missing"}
    ms, mr = _basenames("mixed_same"), _basenames("mixed_rand")
    diff = ms ^ mr
    return {"ok": not diff, "n_mixed_same": len(ms), "n_mixed_rand": len(mr),
            "n_symmetric_diff": len(diff), "sample_diff": sorted(diff)[:10]}


def check_c5() -> dict[str, Any]:
    """
    Checks composite filename pattern coverage.
    - Pre: Composite variant directories may or may not exist.
    - Post: Returns a report dictionary with match rate and pass flag.
    """
    total, matched, samples = 0, 0, []
    for v in _COMPOSITE_VARIANTS:
        if not _val_dir(v).is_dir():
            continue
        for _, name, _ in _iter_images(v):
            total += 1
            if _FG_RE.match(name):
                matched += 1
            elif len(samples) < 20:
                samples.append(name)
    rate = matched / total if total else 0.0
    return {"ok": rate >= 0.99, "total": total, "matched": matched,
            "match_rate": round(rate, 4), "unmatched_samples": samples}


def check_c6(variants: list[str], min_per_class: int = 300) -> dict[str, Any]:
    """
    Checks minimum image count per class.
    - Pre: `variants` is a list of variant names.
        `min_per_class` is a non-negative integer.
    - Post: Returns a report dictionary with low-class entries and pass flag.
    """
    counts, low = {}, []
    for v in variants:
        cc: dict[str, int] = defaultdict(int)
        for cls, _, _ in _iter_images(v):
            cc[cls] += 1
        counts[v] = dict(cc)
        for cls, k in cc.items():
            if k < min_per_class:
                low.append({"variant": v, "class": cls, "n": k})
    return {"ok": not low, "low_classes": low, "min_per_class": min_per_class, "counts": counts}


def check_c7() -> dict[str, Any]:
    """
    Checks ImageNet-to-IN9 mapping coverage.
    - Pre: The IN9 map file is available and readable.
    - Post: Returns a report dictionary with key counts and pass flag.
    """
    m = load_in9_map()
    per = defaultdict(int)
    for val in m.values():
        per[val] += 1
    missing = [c for c in range(9) if per.get(c, 0) < 1]
    return {"ok": len(m) == 1000 and not missing, "n_keys": len(m),
            "classes_missing": missing,
            "n_mapped_to_in9": sum(per.get(c, 0) for c in range(9)),
            "n_mapped_to_minus1": per.get(-1, 0)}


def verify(variants: "list[str] | None" = None) -> dict[str, Any]:
    """
    Runs all dataset structure checks.
    - Pre: `variants` is None or a list of variant names.
    - Post: Returns a JSON-serializable report with all check results and global pass flag.
    """
    cfg = load_config()
    variants = variants or list(cfg.plan_a) + [v for v in cfg.plan_b_extra if v not in cfg.plan_a]
    checks = {
        "C1_variant_dirs": check_c1(variants),
        "C2_canonical_classes": check_c2(variants),
        "C3_dims_and_readable": check_c3(variants),
        "C4_ms_mr_pairing": check_c4(),
        "C5_composite_naming": check_c5(),
        "C6_class_counts": check_c6(variants),
        "C7_in9_map": check_c7(),
    }
    return {"passed": all(c["ok"] for c in checks.values()), "variants": variants, "checks": checks}


def quarantine_corrupt(report: dict[str, Any], dest: "Path | None" = None) -> list[str]:
    """
    Moves corrupt image files to a quarantine directory.
    - Pre: `report` contains a C3 corruption list from `verify()`.
        `dest` is None or a valid destination path.
    - Post: Returns a list of source file paths that were moved.
    """
    import shutil
    dest = dest or paths.data_dir() / "_corrupt"
    moved = []
    for item in report["checks"]["C3_dims_and_readable"]["corrupt"]:
        src = Path(item["path"])
        if src.is_file():
            (dest).mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest / src.name))
            moved.append(str(src))
    return moved


if __name__ == "__main__":
    import json
    import sys
    rep = verify()
    out = paths.verify_report_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print(f"\nWrote {out}")
    sys.exit(0 if rep["passed"] else 1)