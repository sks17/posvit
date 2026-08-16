"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
References:
    - "Rotary Position Embedding for Vision Transformer" (Heo et al., 2024)
    - "The Role of Image Backgrounds in Object Recognition" (Xia et. al 2021)
"""

from __future__ import annotations
import sys
import os
from pathlib import Path


# Every path is derived from this file's own location rather than the working directory,
# so a command resolves the same way from any cwd and on the cluster.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """
    Returns the repository root path.
    - Pre: No input is required.
    - Post: Returns a Path object for the repository root.
    """
    return _REPO_ROOT


def configs_dir() -> Path:
    """
    Returns the configs directory path.
    - Pre: No input is required.
    - Post: Returns a Path object for the configs directory.
    """
    return _REPO_ROOT / "configs"


def results_root() -> Path:
    """
    Returns the results root path.
    - Pre: No input is required.
    - Post: Returns a Path object for the parent of `raw` and `derived`.
    """
    return _REPO_ROOT / "results"


def results_raw() -> Path:
    """
    Returns the raw results directory path.
    - Pre: No input is required.
    - Post: Returns a Path object for `results/raw`.
    """
    return results_root() / "raw"


def results_derived() -> Path:
    """
    Returns the derived results directory path.
    - Pre: No input is required.
    - Post: Returns a Path object for `results/derived`.
    """
    return results_root() / "derived"


def ensure_on_syspath(path: "Path | str") -> None:
    """
    Ensures a path is present in `sys.path`.
    - Pre: `path` is a Path or string.
    - Post: Inserts the path at the start of `sys.path` if it is missing.
    """
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def vendored(*parts: str) -> Path:
    """
    Resolves a vendored dependency path.
    - Pre: `parts` defines a relative path inside vendor locations.
    - Post: Returns the first existing candidate path.
        Raises FileNotFoundError if no candidate exists.
    """
    candidates: list[Path] = []
    override = os.environ.get("POSVIT_VENDOR")
    if override:
        candidates.append(Path(override).joinpath(*parts))
    candidates.append(_REPO_ROOT.joinpath("vendor", *parts))
    candidates.append(_REPO_ROOT.joinpath(*parts))
    for c in candidates:
        if c.exists():
            return c
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Vendored path {'/'.join(parts)!r} not found. Looked in:\n  {searched}\n"
        f"Clone the third-party repo into <repo>/vendor/, or set POSVIT_VENDOR."
    )


def rope_vit_root() -> Path:
    """
    Returns the vendored RoPE-ViT repository path.
    - Pre: The vendored repository is available in a supported location.
    - Post: Returns a Path object for the RoPE-ViT repository.
    """
    return vendored("rope-vit-main", "rope-vit-main")


def bg_challenge_root() -> Path:
    """
    Returns the vendored backgrounds_challenge repository path.
    - Pre: The vendored repository is available in a supported location.
    - Post: Returns a Path object for the backgrounds_challenge repository.
    """
    return vendored("backgrounds_challenge-master", "backgrounds_challenge-master")


def data_dir() -> Path:
    """
    Returns the project data directory path.
    - Pre: No input is required.
    - Post: Returns `POSVIT_DATA` when set, else the default data path.
    """
    override = os.environ.get("POSVIT_DATA")
    return Path(override) if override else _REPO_ROOT / "data"


def in9_bg_root() -> Path:
    """
    Returns the IN-9 background dataset root path.

    This must match where `data.acquire.acquire` extracts the release archive; the
    tarball unpacks a `bg_challenge` directory inside `data/in9`.
    - Pre: The IN-9 data tree exists under the data directory.
    - Post: Returns a Path object for the IN-9 background root.
    """
    return data_dir() / "in9" / "bg_challenge"


def in9_map_path() -> Path:
    """
    Returns the ImageNet-to-IN9 map file path.

    The map ships inside the vendored backgrounds_challenge repository rather than in
    the downloaded archive, so this resolves through `bg_challenge_root` and fails
    loudly when the vendored repository is absent.
    - Pre: The vendored backgrounds_challenge repository is available.
    - Post: Returns a Path object for `in_to_in9.json`.
    """
    return bg_challenge_root() / "in_to_in9.json"


def waterbirds_root() -> Path:
    """
    Returns the Waterbirds dataset root path.

    The release extracts into a same-named subdirectory, so one level down is searched
    as well. This replaces the resolve-or-return-None pattern: a missing dataset raises
    with the command that fixes it.
    - Pre: The Waterbirds release has been downloaded, or `POSVIT_WATERBIRDS` is set.
    - Post: Returns a Path object for the directory holding `metadata.csv`.
        Raises FileNotFoundError when no candidate contains the metadata file.
    """
    override = os.environ.get("POSVIT_WATERBIRDS")
    candidates = [Path(override)] if override else []
    candidates.append(data_dir() / "waterbirds")
    for c in candidates:
        if (c / "metadata.csv").is_file():
            return c
        if c.is_dir():
            for sub in sorted(p for p in c.iterdir() if p.is_dir()):
                if (sub / "metadata.csv").is_file():
                    return sub
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Waterbirds dataset not found. Looked in:\n  {searched}\n"
        f"Download the release into <repo>/data/waterbirds/, or set POSVIT_WATERBIRDS."
    )


def predictions_dir() -> Path:
    """
    Returns the predictions output directory path.
    - Pre: No input is required.
    - Post: Returns a Path object for raw prediction artifacts.
    """
    return results_raw() / "predictions"


def degrade_dir() -> Path:
    """
    Returns the positional sweep output directory path.
    - Pre: No input is required.
    - Post: Returns a Path object for raw beta-sweep records.
    """
    return results_raw() / "degrade"


def verify_report_path() -> Path:
    """
    Returns the dataset verification report path.
    - Pre: No input is required.
    - Post: Returns a Path object for the persisted C1-C7 report.
    """
    return results_raw() / "verify_report.json"


def metrics_dir() -> Path:
    """
    Returns the committed metrics directory path.
    - Pre: No input is required.
    - Post: Returns a Path object for `results/derived/metrics`.
    """
    return results_derived() / "metrics"


def figures_dir() -> Path:
    """
    Returns the committed figures directory path.
    - Pre: No input is required.
    - Post: Returns a Path object for `results/derived/figures`.
    """
    return results_derived() / "figures"


def manifest_path() -> Path:
    """
    Returns the run manifest path.
    - Pre: No input is required.
    - Post: Returns a Path object for `results/MANIFEST.json`.
    """
    return results_root() / "MANIFEST.json"
