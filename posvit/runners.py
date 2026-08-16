"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
Adapters between the command-line interface and the analysis modules.

Each runner takes the same `(spec, cfg, opts)` triple and owns the orchestration one
pipeline step needs: loading a checkpoint, preloading variants, looping models. The
analysis modules stay free of argument parsing and the CLI stays free of orchestration.

Every per-model runner is idempotent. Jobs run on a preemptible partition with
`--requeue`, so a re-queued task must skip completed work rather than redo it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from . import paths
from .registry import ModelSpec, all_models


@dataclass(frozen=True)
class Options:
    """
    Holds the command-line options shared by every runner.
    """
    force: bool = False
    device: "str | None" = None
    limit: "int | None" = None


def _write_metric(name: str, payload: "dict[str, Any]"):
    """
    Writes one committed metrics artifact.
    - Pre: `name` is a file stem and `payload` is JSON-serializable.
    - Post: Returns the Path written, with keys sorted for a stable diff.
    """
    out_dir = paths.metrics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _variants(cfg) -> "list[str]":
    """
    Lists the evaluation variants in plan order.
    - Pre: `cfg` has `plan_a` and `plan_b_extra`.
    - Post: Returns plan A followed by any plan B extras not already present.
    """
    return list(cfg.plan_a) + [v for v in cfg.plan_b_extra if v not in cfg.plan_a]


def _load_model(spec: ModelSpec, cfg, opts: Options):
    """
    Loads one verified checkpoint onto the resolved device.
    - Pre: `spec` is a ModelSpec and the checkpoint hash is pinned in the registry.
    - Post: Returns `(model, receipt, device)`.
        Raises IntegrityError when the checkpoint is unverified.
    """
    from .checkpoints import load_checkpoint
    from .evaluate import resolve_device

    device = resolve_device(opts.device)
    model, receipt = load_checkpoint(spec, img_size=cfg.img_size, device=device)
    return model, receipt, device


def run_acquire(spec, cfg, opts: Options):
    """
    Downloads and extracts the IN-9 release.
    - Pre: Network access is available on first run.
    - Post: Returns the acquisition provenance dictionary.
    """
    from .data.acquire import acquire

    report = acquire(force=opts.force)
    print(json.dumps(report, indent=2))
    return report


def run_verify(spec, cfg, opts: Options):
    """
    Runs the read-only C1-C7 dataset checks and persists the report.
    - Pre: The dataset has been acquired.
    - Post: Returns the verification report.
        Returns False when any check failed, so the CLI exits non-zero.
    """
    from .data.verify import verify

    report = verify()
    out = paths.verify_report_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"dataset verification: {'PASS' if report['passed'] else 'FAIL'} -> {out}")
    return report if report["passed"] else False


def run_masks(spec, cfg, opts: Options):
    """
    Derives foreground masks from the ONLY-FG composites.
    - Pre: The dataset has been acquired and verified.
    - Post: Returns the mask manifest.
        Returns False when coverage falls below the gate.
    """
    from .data.masks import derive_masks

    manifest = derive_masks(cfg)
    print(json.dumps(manifest, indent=2))
    return manifest if manifest.get("passed", False) else False


def run_checkpoints(spec, cfg, opts: Options):
    """
    Verifies every checkpoint hash, or records observed digests with `--force`.
    - Pre: The vendored RoPE-ViT repository is present.
    - Post: Returns True when all hashes are pinned and match.
        Returns False when any checkpoint is unverified or mismatched.
    """
    from .checkpoints import IntegrityError, download_checkpoint, record_hashes, verify_checkpoint

    if opts.force:
        record_hashes()
        return True
    failed = []
    for s in all_models():
        try:
            verify_checkpoint(s, download_checkpoint(s))
            print(f"OK       {s.model_id}")
        except IntegrityError as exc:
            failed.append(s.model_id)
            print(f"FAIL     {s.model_id}: {str(exc).splitlines()[0]}")
    return not failed


def run_evaluate(spec: ModelSpec, cfg, opts: Options):
    """
    Evaluates one model over every configured variant.
    - Pre: `spec` is a ModelSpec and the dataset is present.
    - Post: Returns the per-variant sidecar dictionaries.
        Returns False when the clean-accuracy sanity gate fails.
    """
    from .evaluate import evaluate_variant, sanity_check
    from .seed import seed_everything

    receipt_seed = seed_everything(cfg.seed)
    model, receipt, device = _load_model(spec, cfg, opts)
    receipt["seed"] = receipt_seed

    out = {}
    for variant in _variants(cfg):
        meta = evaluate_variant(
            model, spec, variant, cfg, device=device, receipt=receipt,
            limit=opts.limit, force=opts.force,
        )
        out[variant] = meta
        print(f"{spec.safe_id:22s} {variant:12s} n={meta['n_records']:5d} "
              f"acc={meta['accuracy']:.4f} out_of_in9={meta['frac_pred_out_of_in9']:.4f}")

    gate = sanity_check(model, spec, cfg, device=device)
    print(f"sanity {spec.model_id}: {gate['original_acc']:.4f} "
          f"{'PASS' if gate['passed'] else 'FAIL'} (>= {gate['threshold']})")
    return out if gate["passed"] else False


def run_metrics(spec, cfg, opts: Options):
    """
    Reduces per-image predictions to BG-Gap for every evaluated model.
    - Pre: At least one model has a complete set of prediction runs.
    - Post: Returns the report keyed by `safe_id` and writes `bggap.json`.
    """
    from .metrics import bggap_report

    reports = {}
    for s in all_models():
        try:
            reports[s.safe_id] = bggap_report(s, cfg)
        except FileNotFoundError:
            continue
        rep = reports[s.safe_id]
        print(f"{s.safe_id:22s} BG-Gap={rep['bg_gap']:.4f} "
              f"[{rep['ci_lo']:.4f}, {rep['ci_hi']:.4f}]  N={rep['n_paired']}")
    print(f"wrote {_write_metric('bggap', reports)}  ({len(reports)} models)")
    return reports


def run_intervene(spec: ModelSpec, cfg, opts: Options):
    """
    Sweeps the functional positional signal for one model.
    - Pre: `spec` is a ModelSpec and the dataset is present.
    - Post: Returns the sweep record and writes it under `results/raw/degrade`.
    """
    from .intervene import functional_knob, sweep
    from .seed import seed_everything

    seed_everything(cfg.seed)
    knob = functional_knob(spec)
    out_path = paths.degrade_dir() / f"{spec.safe_id}__{knob}.json"
    if out_path.is_file() and not opts.force:
        print(f"{spec.safe_id}: sweep already complete -> {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    model, _, device = _load_model(spec, cfg, opts)
    return sweep(model, spec, cfg, knob=knob, device=device)


def run_dose(spec, cfg, opts: Options):
    """
    Turns the beta sweeps into the guarded dose-response with BH-FDR.
    - Pre: At least one model has a sweep record.
    - Post: Returns the dose-response summary and writes `dose_response.json`.
    """
    from .doseresponse import analyze_all

    result = analyze_all(cfg)
    for safe_id, r in sorted(result["models"].items()):
        s = r["slope"]
        print(f"{safe_id:22s} {r['knob_role']:10s} rho={r['spearman_rho']:+.3f} "
              f"slope={s['slope']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] "
              f"FDR={'PASS' if r['fdr_pass'] else 'no'}")
    print(f"{result['n_positive_slope_ci_excludes_zero']}/{result['n_models']} slopes with CI > 0")
    return result


def run_controls(spec: ModelSpec, cfg, opts: Options):
    """
    Sweeps the accuracy-matched specificity controls for one model.
    - Pre: `spec` has a completed positional sweep.
    - Post: Returns the specificity report and writes it to the metrics directory.
    """
    from .controls import CONTROLS, specificity_report, sweep_control
    from .data import load_in9_map
    from .data.loaders import preload_uint8
    from .intervene import functional_knob, load_sweep_rows
    from .seed import seed_everything

    seed_everything(cfg.seed)
    model, _, device = _load_model(spec, cfg, opts)
    in9_map = load_in9_map()
    data = {v: preload_uint8(cfg, v) for v in ("original", "mixed_same", "mixed_rand")}

    rows = {}
    for name, control in CONTROLS.items():
        rows[name] = sweep_control(model, spec, cfg, control, data=data, in9_map=in9_map,
                                   device=device, batch=256)
        print(f"  {name:13s} " + "  ".join(
            f"{r['strength']:.2f}:acc={r['original_acc']:.3f}" for r in rows[name]))

    pos_rows = load_sweep_rows(spec, functional_knob(spec))
    report = specificity_report(spec, cfg, pos_rows, rows,
                                n_images=len(data["mixed_same"][1]))
    print(f"wrote {_write_metric(f'specificity__{spec.safe_id}', {'report': report, 'curves': rows})}")
    return report


def run_equivalence(spec, cfg, opts: Options):
    """
    Runs TOST equivalence and the minimum detectable effect across encodings.
    - Pre: The ViT-S models have complete prediction runs.
    - Post: Returns the equivalence report and writes `equivalence.json`.
    """
    from .equivalence import equivalence_report
    from .metrics import join_variants, load_variant

    corr = {}
    for s in all_models():
        if s.scale != "vit_s":
            continue
        try:
            join = join_variants(load_variant(s, "mixed_same"), load_variant(s, "mixed_rand"))
        except FileNotFoundError:
            continue
        corr[s.safe_id] = (join.ms, join.mr)

    report = equivalence_report(corr, cfg)
    print(f"{report['n_equivalent']}/{report['n_pairs']} pairs equivalent at "
          f"{report['declared_margin_pp']:.1f} pp  (MDE {report['mde_pp']:.2f} pp)")
    print(f"wrote {_write_metric('equivalence', report)}")
    return report


def run_stratify(spec, cfg, opts: Options):
    """
    Stratifies BG-Gap by class, object size, and object position.
    - Pre: Masks are derived and models have complete prediction runs.
    - Post: Returns the per-model reports and writes one artifact per model.
    """
    from .stratify import stratify_report, write_stratify

    reports = {}
    for s in all_models():
        try:
            rep = stratify_report(s, cfg)
        except FileNotFoundError:
            continue
        reports[s.safe_id] = rep
        trend = rep["by_attribute"]["fg_area_frac"]["global"]["trend_top_minus_bottom"]
        print(f"{s.safe_id:22s} size trend {trend['top_minus_bottom'] * 100:+.2f} pp "
              f"[{trend['ci_lo'] * 100:+.2f}, {trend['ci_hi'] * 100:+.2f}]")
        write_stratify(s, rep)
    return reports


def run_waterbirds(spec: ModelSpec, cfg, opts: Options):
    """
    Replicates the positional effect on the Waterbirds benchmark.
    - Pre: `spec` is a ViT-S model and the Waterbirds release is present.
    - Post: Returns the sweep record.
        Raises RuntimeError when the baseline spurious gap fails the positive control.
    """
    from .seed import seed_everything
    from .waterbirds import waterbirds_sweep

    seed_everything(cfg.seed)
    model, _, device = _load_model(spec, cfg, opts)
    record = waterbirds_sweep(model, spec, cfg, device=device, limit=opts.limit)
    for beta in record["betas"]:
        r = record["results"][f"{beta:.2f}"]
        print(f"  beta={beta:.2f}  acc={r['acc']:.4f}  gap={r['spurious_gap'] * 100:+.2f} pp "
              f"(dfr {r['dfr_spurious_gap'] * 100:+.2f} pp)")
    return record


def run_probes(spec: ModelSpec, cfg, opts: Options):
    """
    Scores attention spread as a per-image predictor of background reliance.
    - Pre: `spec` has complete prediction runs and masks are derived.
    - Post: Returns the probe report and writes it to the metrics directory.
    """
    import numpy as np
    import torch

    from .data import load_in9_map
    from .data.loaders import preload_uint8
    from .hooks import capture_attention
    from .metrics import join_variants, load_variant
    from .probes import attention_probe_report, attention_stats
    from .seed import seed_everything
    from .stratify import load_mask_attributes

    seed_everything(cfg.seed)
    model, _, device = _load_model(spec, cfg, opts)
    join = join_variants(load_variant(spec, "mixed_same"), load_variant(spec, "mixed_rand"))
    u8, _, fg_ids = preload_uint8(cfg, "mixed_rand")
    load_in9_map()

    masks_dir = paths.results_raw() / "masks"
    patch_masks = []
    for fg in fg_ids:
        hits = list(masks_dir.rglob(f"{fg}.npz"))
        with np.load(hits[0]) as d:
            patch_masks.append(d["patch_mask"])
    patch_mask = torch.from_numpy(np.stack(patch_masks))

    mean = torch.tensor(cfg.norm_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.norm_std, device=device).view(1, 3, 1, 1)
    off_fg, entropy = [], []
    with capture_attention(model, layers=(-1,)) as store:
        for i in range(0, len(u8), 64):
            x = u8[i:i + 64].to(device).float().div_(255)
            with torch.no_grad():
                model((x - mean) / std)
            o, e = attention_stats(store[-1], patch_mask[i:i + 64])
            off_fg.append(o)
            entropy.append(e)

    report = attention_probe_report(np.concatenate(entropy), np.concatenate(off_fg), join, cfg)
    print(f"{spec.safe_id:22s} entropy AUC={report['entropy']['auc']:.3f} "
          f"[{report['entropy']['ci_lo']:.3f}, {report['entropy']['ci_hi']:.3f}]")
    print(f"wrote {_write_metric(f'probes__{spec.safe_id}', report)}")
    return report


def run_mechanisms(spec: ModelSpec, cfg, opts: Options):
    """
    Sweeps the rotary frequency bands and matches them against the controls.
    - Pre: `spec` is a RoPE-Mixed model with completed control curves.
    - Post: Returns the band report and writes it to the metrics directory.
    """
    from .controls import match_curves
    from .data import load_in9_map
    from .data.loaders import preload_uint8
    from .intervene import evaluate_batch, scaled, snapshot
    from .mechanisms import attenuate_rope_band
    from .seed import seed_everything

    seed_everything(cfg.seed)
    model, _, device = _load_model(spec, cfg, opts)
    in9_map = load_in9_map()
    data = {v: preload_uint8(cfg, v) for v in ("original", "mixed_same", "mixed_rand")}
    snap = snapshot(model, spec)

    bands = {}
    for band in ("high", "low"):
        rows = []
        # Graded attenuation, not hard removal: a band that is deleted outright also
        # destroys general accuracy, which confounds the band with capacity loss.
        for beta in (1.0, 0.75, 0.5, 0.25, 0.0):
            with scaled(model, snap, "rope", 1.0):
                attenuate_rope_band(model, snap, spec, band=band, beta=beta,
                                    frac=cfg.rope_band_frac)
                oc, oconf = evaluate_batch(model, *data["original"][:2], in9_map, cfg, device)
                ms, _ = evaluate_batch(model, *data["mixed_same"][:2], in9_map, cfg, device)
                mr, _ = evaluate_batch(model, *data["mixed_rand"][:2], in9_map, cfg, device)
            rows.append({
                "strength": float(beta),
                "original_acc": round(float(oc.mean()), 6),
                "original_conf": round(float(oconf.mean()), 6),
                "bg_gap": round(float(ms.mean() - mr.mean()), 6),
            })
        bands[band] = rows
        print(f"  {band:5s} band  " + "  ".join(
            f"{r['strength']:.2f}:gap={r['bg_gap'] * 100:.2f}pp" for r in rows))

    report = {
        "model_id": spec.model_id,
        "safe_id": spec.safe_id,
        "band_frac": cfg.rope_band_frac,
        "curves": bands,
        "high_vs_low": match_curves(bands["high"], bands["low"],
                                    n_images=len(data["mixed_same"][1])),
    }
    print(f"wrote {_write_metric(f'mechanisms__{spec.safe_id}', report)}")
    return report


def run_table3(spec, cfg, opts: Options):
    """
    Assembles the candidate-mechanism table from the committed probe artifacts.
    - Pre: The probe and control artifacts exist for at least one model.
    - Post: Returns the table and writes `mechanisms.json`.
    """
    from .mechanisms import PROBES, table3, verdict

    md = paths.metrics_dir()
    rows: dict[str, Any] = {}

    entropy = sorted(md.glob("probes__*.json"))
    if entropy:
        payload = json.loads(entropy[0].read_text(encoding="utf-8"))
        rows["attention_entropy"] = {
            "how_tested": "per-image AUC",
            "summary": f"AUC {payload['entropy']['auc']:.2f}",
            **verdict(PROBES["attention_entropy"], payload["entropy"], cfg),
        }

    specificity = sorted(md.glob("specificity__*.json"))
    if specificity:
        payload = json.loads(specificity[0].read_text(encoding="utf-8"))["report"]
        block = payload["controls"].get("attn_temp", {}).get("matched_acc", {})
        value = None
        if block.get("comparable"):
            value = {"ci_lo_pp": block["excess_min_pp"], "ci_hi_pp": block["excess_max_pp"]}
        rows["attention_routing"] = {
            "how_tested": "attention-temperature sweep at matched accuracy",
            "summary": "null / reversed" if value else "not comparable",
            **verdict(PROBES["attention_routing"], value, cfg),
        }

    band = sorted(md.glob("mechanisms__*.json"))
    if band:
        payload = json.loads(band[0].read_text(encoding="utf-8"))["high_vs_low"]
        value = None
        if payload.get("comparable"):
            value = {"ci_lo_pp": payload["excess_min_pp"], "ci_hi_pp": payload["excess_max_pp"]}
        rows["rope_high_band"] = {
            "how_tested": "excess BG-Gap at matched accuracy",
            "summary": f"{payload.get('excess_mean_pp', float('nan')):+.2f} pp",
            **verdict(PROBES["rope_high_band"], value, cfg),
        }

    table = table3(rows, cfg)
    for name, r in table["probes"].items():
        print(f"{name:22s} {r['verdict']}")
    print(f"wrote {_write_metric('mechanisms', table)}")
    return table


def run_figures(spec, cfg, opts: Options):
    """
    Regenerates every figure from the committed metrics.
    - Pre: The metrics directory holds the artifacts each figure draws.
    - Post: Returns the list of written figure paths.
    """
    from .figures import main as build_figures

    written = build_figures(cfg)
    for p in written:
        print(f"wrote {p}")
    return written


def run_manifest(spec, cfg, opts: Options):
    """
    Collects every receipt into the run manifest and certifies the run.
    - Pre: The pipeline has produced at least some artifacts.
    - Post: Returns the manifest.
        Returns False when certification is blocked, so the CLI exits non-zero.
    """
    from .manifest import certify, collect_manifest, write_manifest

    manifest = collect_manifest(cfg)
    manifest["certification"] = certify(manifest)
    result = manifest["certification"]

    print(f"dataset: {manifest['dataset']['status']}   masks: {manifest['masks']['status']}")
    for safe_id, m in manifest["models"].items():
        pinned = (m.get("checkpoint") or {}).get("sha256_pinned")
        print(f"  {safe_id:22s} sanity={m['sanity']['status']:8s} pinned={pinned}")
    for w in result["warnings"]:
        print(f"WARN     {w}")
    for b in result["blockers"]:
        print(f"BLOCKER  {b}")
    print(f"wrote {write_manifest(manifest)}")
    print("CERTIFIED" if result["certified"] else "NOT CERTIFIED")
    return manifest if result["certified"] else False
