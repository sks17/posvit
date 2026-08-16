# Positional Information Causally Constrains Background Shortcut Reliance in Vision Transformers

Scaling the positional signal of eight frozen, identically trained Vision Transformers
shows that attenuating positional information monotonically raises background-shortcut
reliance, that the encoding scheme does not matter, and that the effect is specific to
spatial information loss rather than generic degradation.

**Paper:** [ ]  ·  **License:** [MIT](LICENSE)  ·  **Provenance and limitations:** [docs/PROVENANCE.md](docs/PROVENANCE.md)

---

## Abstract

Vision Transformers (ViTs) match or exceed the strongest architectures for object
recognition, but, like Convolutional Neural Networks (CNNs), they can exploit image
backgrounds rather than focused objects. Whether a ViT's positional encoding causally
governs this reliance has not been thoroughly tested in literature, and it is commonly
assumed that newer rotary encodings (RoPE) resist such shortcuts better than absolute
encodings (APE). In this work, we probe this assumption by scaling the positional signal
of eight identically trained ViT checkpoints and measuring their resulting changes in
background reliance on ImageNet-9. We find that attenuating the positional signal
monotonically raises background reliance in every model (e.g. 5.1 → 12.8 pp), that the
encoding scheme is irrelevant, and that the effect is specific to spatial information
loss, not generic degradation. These findings extend to the Waterbirds dataset [8]. We
investigate several mechanisms and find little evidence any single mechanism carries the
effect.

---

## Installation

Requires Python 3.11 or newer.

```bash
git clone <repository-url> && cd PCCBVTR
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r env/requirements.lock
pip install -e .
```

On a CUDA machine, install torch from the matching index first
(`pip install torch --index-url https://download.pytorch.org/whl/cu121`), then the lockfile.

The third-party code this project builds on is vendored under `vendor/` and is required:
`rope-vit-main` supplies the model constructors, and `backgrounds_challenge-master` supplies
the ImageNet-to-IN9 class map.

## Reproducing the results


| # | Command | Expect |
|---|---|---|
| 1 | `posvit acquire` | `sha256 2d66f571…896fe7`, 279 581 133 bytes |
| 2 | `posvit verify` | `PASS`; check C4 reports |
| 3 | `posvit masks` | `coverage 1.0`, 4050 of 4050 foregrounds |
| 4 | `posvit checkpoints` | `OK` for all eight models |
| 5 | `posvit evaluate --index 0` … `--index 7` | clean ImageNet-9 accuracy ≈ 0.97 |
| 6 | `posvit metrics` | `vit_s__ape` BG-Gap ≈ 0.0514 |
| 7 | `posvit intervene --index 0` … `--index 7` | BG-Gap rises as β falls |
| 8 | `posvit dose` | 8 of 8 slopes with a confidence interval above zero |
| 9 | `posvit equivalence` | all pairs equivalent at the declared 2 pp margin |
| 10 | `posvit controls --index 0` … `--index 7` | positive excess over the non-spatial controls |
| 11 | `posvit stratify` · `posvit waterbirds --index 0..4` · `posvit probes --index 0..7` | see `results/derived/metrics/` |
| 12 | `posvit figures` | figures regenerate with no diff |
| 13 | `posvit manifest` | exits 0 |


`posvit models` prints the registry with the array indices used above, and
`posvit array-spec <command>` prints the SLURM array range for a command.

### On a cluster

```bash
cp hpc/config.example.sh hpc/config.sh
bash hpc/submit.sh
```

## Testing

```bash
pytest
pytest -m smoke
pytest -m gpu
```

## Limitations

Known open questions about method disclosure and artifact provenance are listed in
[docs/PROVENANCE.md](docs/PROVENANCE.md).

## Citation

```

```

## Acknowledgements

This work builds directly on the RoPE-ViT release of Heo et al. (2024), the DeiT-III
recipe of Touvron et al. (2022), the ImageNet-9 / Backgrounds Challenge benchmark of
Xiao et al. (2021), and the Waterbirds benchmark of Sagawa et al. (2020). Vendored
third-party code retains its original licensing; see `vendor/` and
[docs/PROVENANCE.md](docs/PROVENANCE.md).
