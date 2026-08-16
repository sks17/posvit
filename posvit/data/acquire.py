"""
Download and extract the IN-9 Backgrounds-Challenge dataset, if not already present.
"""

from __future__ import annotations
import tarfile
import urllib.request
from typing import Any
from .. import paths
from ..hashing import sha256_file

DATASET_URL = (
    "https://github.com/MadryLab/backgrounds_challenge/releases/download/data/"
    "backgrounds_challenge_data.tar.gz"
)
EXPECTED_SHA256 = "2d66f571b0492986b347e37d2ee498ed34b58012af425d772dacda4422896fe7"
EXPECTED_SIZE_BYTES = 279_581_133

def acquire(force: bool = False) -> dict[str, Any]:
    """
    Downloads and extracts the IN-9 dataset archive.
    - Pre: `force` is True or False.
    - Post: Returns a provenance dictionary.
        The tar file is verified before extraction.
        Extraction runs when `force` is True or marker is missing.
    """
    data_dir = paths.data_dir()
    tar_path = data_dir / "bg_data.tar.gz"
    in9_root = data_dir / "in9"
    marker = in9_root / ".extracted"
    data_dir.mkdir(parents=True, exist_ok=True)

    downloaded = False
    if not tar_path.is_file():
        print(f"Downloading {DATASET_URL}\n  -> {tar_path}")
        urllib.request.urlretrieve(DATASET_URL, tar_path)
        downloaded = True

    digest = sha256_file(tar_path)
    if digest != EXPECTED_SHA256:
        raise ValueError(
            "IN-9 tar SHA-256 mismatch:\n"
            f"  got      {digest}\n  expected {EXPECTED_SHA256}\n"
            f"The download is corrupt or has changed. Delete {tar_path} and re-run."
        )

    extracted = False
    if force or not marker.is_file():
        in9_root.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {tar_path.name} -> {in9_root}")
        with tarfile.open(tar_path, "r:gz") as tf:
            # The archive is only unpacked after its hash matches, and `filter="data"`
            # additionally refuses absolute or parent-relative member paths.
            tf.extractall(path=in9_root, filter="data")
        marker.write_text("ok\n", encoding="utf-8")
        extracted = True

    return {
        "url": DATASET_URL,
        "sha256": digest,
        "size_bytes": tar_path.stat().st_size,
        "tar_path": str(tar_path),
        "bg_challenge_root": str(paths.in9_bg_root()),
        "downloaded": downloaded,
        "extracted": extracted,
    }