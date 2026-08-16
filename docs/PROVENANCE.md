# Provenance, tooling, and known limitations

This file records where the code and data came from, how the repository was produced, and
what remains unresolved.

## Vendored third-party code

`vendor/` holds third-party source

| Directory | Source | Used for |
|---|---|---|
| `vendor/rope-vit-main/` | RoPE-ViT, Heo et al., ECCV 2024 | model constructors (`deit.models_v2`, `models.vit_rope`) and the released checkpoints |
| `vendor/backgrounds_challenge-master/` | Backgrounds Challenge, Xiao et al., 2021 | the 1000-to-9 class map `in_to_in9.json` |
| `vendor/ViT_OOD_generalization-main/` | Zhang et al., CVPR 2022 | reference only |
| `vendor/papers/` | the cited papers | reference only |
| `vendor/HyakDocs/` | UW Hyak cluster documentation | reference only |

## Datasets

- **ImageNet-9 / Backgrounds Challenge** — fetched and hash-checked by `posvit acquire`
  against `2d66f571b0492986b347e37d2ee498ed34b58012af425d772dacda4422896fe7`.
- **Waterbirds** — fetched manually from the Stanford mirror; see
  `posvit/waterbirds.py` and `paths.waterbirds_root`.

## Tooling disclosure

Parts of this repository were written with AI assistance. Specifically, the package
scaffolding, module and function documentation, the command-line interface, the cluster
harness, and the test suite were drafted with the help of a large language model and then
reviewed, corrected, and run by the author.

The scientific content was solely made by the author.

## Reproducibility caveat

Ensure all library versions match manifest.py
