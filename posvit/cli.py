"""
Package for "Positional Information Causally Constrains Background Reliance in Vision Transformers"
License: MIT
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from .registry import ModelSpec, all_models, by_id, by_index


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    run: Callable[..., Any]
    needs_gpu: bool = False
    scope: str = "global"
    applies_to: Callable[[ModelSpec], bool] = lambda spec: True
    requires: "tuple[str, ...]" = field(default_factory=tuple)
    time: str = "01:00:00"
    mem: str = "16G"


def _runner(name: str) -> Callable:
    """
    Builds a lazy dispatcher for one runner.

    The import is deferred so `posvit --help` and the harness metadata queries stay
    importable on a login node without torch installed.
    - Pre: `name` is a function defined in `posvit.runners`.
    - Post: Returns a callable that imports the runner on first invocation.
    """

    def call(*a, **kw):
        """
        Imports and invokes the named runner.
        - Pre: The runner module is importable.
        - Post: Returns whatever the runner returns.
        """
        import importlib

        return getattr(importlib.import_module("posvit.runners"), name)(*a, **kw)

    return call


def _is_rope_mixed(spec: ModelSpec) -> bool:
    """
    Checks if model uses mixed RoPE configuration.
    - Pre: `spec` is a valid ModelSpec object.
    - Post: Returns True when model has RoPE and key contains `mixed`.
    """
    return spec.has_rope and "mixed" in spec.key


COMMANDS = {
    c.name: c for c in (
        Command("acquire", "download and hash-check the IN-9 release",
                _runner("run_acquire")),
        Command("verify", "C1-C7 structural checks, read-only",
                _runner("run_verify"), requires=("acquire",)),
        Command("masks", "derive foreground masks from the ONLY-FG variant",
                _runner("run_masks"), requires=("verify",)),
        Command("checkpoints", "verify checkpoint hashes, or record them with --force",
                _runner("run_checkpoints")),
        Command("evaluate", "per-image predictions for one model",
                _runner("run_evaluate"), needs_gpu=True, scope="per-model",
                requires=("verify", "checkpoints"), time="01:30:00"),
        Command("metrics", "BG-Gap, confidence intervals, and McNemar",
                _runner("run_metrics"), requires=("evaluate",)),
        Command("intervene", "positional attenuation sweep on frozen weights",
                _runner("run_intervene"), needs_gpu=True, scope="per-model",
                requires=("evaluate",), time="03:00:00"),
        Command("dose", "guarded dose-response with BH-FDR",
                _runner("run_dose"), requires=("intervene",)),
        Command("controls", "accuracy-matched specificity controls",
                _runner("run_controls"), needs_gpu=True, scope="per-model",
                requires=("intervene",), time="04:00:00", mem="32G"),
        Command("equivalence", "TOST equivalence and minimum detectable effect",
                _runner("run_equivalence"), requires=("metrics",)),
        Command("stratify", "BG-Gap by class, object size, and object position",
                _runner("run_stratify"), requires=("metrics", "masks")),
        Command("waterbirds", "cross-dataset generality replication",
                _runner("run_waterbirds"), needs_gpu=True, scope="per-model",
                applies_to=lambda s: s.scale == "vit_s",
                requires=("checkpoints",), time="02:00:00"),
        Command("probes", "attention spread as a per-image predictor",
                _runner("run_probes"), needs_gpu=True, scope="per-model",
                requires=("evaluate", "masks")),
        Command("mechanisms", "rotary frequency-band sweep",
                _runner("run_mechanisms"), needs_gpu=True, scope="per-model",
                applies_to=_is_rope_mixed,
                requires=("intervene", "controls"), time="03:00:00"),
        Command("table3", "assemble the candidate-mechanism table",
                _runner("run_table3"), requires=("probes",)),
        Command("figures", "regenerate every figure from committed metrics",
                _runner("run_figures"), requires=("metrics",)),
        Command("manifest", "collect receipts and certify the run",
                _runner("run_manifest")),
    )
}


def applicable_indices(command: Command) -> "list[int]":
    """
    Lists registry indices applicable to a command.
    - Pre: `command` is a valid Command object.
    - Post: Returns list of model indices satisfying `command.applies_to`.
    """
    return [i for i, s in enumerate(all_models()) if command.applies_to(s)]


def array_spec(command: Command) -> str:
    """
    Builds SLURM array specification string for a command.
    - Pre: `command` is a valid Command object.
    - Post: Returns comma-separated range string of applicable registry indices.
        Raises ValueError when no model satisfies command applicability.
    """
    idx = applicable_indices(command)
    if not idx:
        raise ValueError(f"{command.name}: no model in the registry satisfies applies_to")
    runs, start = [], idx[0]
    for a, b in zip(idx, idx[1:] + [None]):
        if b != (a + 1):
            runs.append(f"{start}-{a}" if a > start else f"{start}")
            start = b
    return ",".join(runs)


def build_parser() -> argparse.ArgumentParser:
    """
    Builds CLI argument parser.
    - Pre: No input is required.
    - Post: Returns configured top-level argparse parser with subcommands.
    """
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_mutually_exclusive_group()
    g.add_argument("--index", type=int, help="registry index (SLURM_ARRAY_TASK_ID)")
    g.add_argument("--id", type=str, help="model id, e.g. vit_s/ape")
    common.add_argument("--force", action="store_true", help="redo completed work")
    common.add_argument("--device", default=None)
    common.add_argument("--limit", type=int, default=None, help="smoke-test subset")

    p = argparse.ArgumentParser(
        prog="posvit",
        description="Positional encoding and background-shortcut reliance in Vision Transformers.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    for c in COMMANDS.values():
        sub.add_parser(c.name, parents=[common], help=c.help)

    sub.add_parser("models", help="print the registry with array indices")
    sp = sub.add_parser("array-spec", help="print the SLURM --array string for a command")
    sp.add_argument("target")
    mp = sub.add_parser("meta", help="print a command's harness metadata")
    mp.add_argument("target")
    mp.add_argument("--field", default=None)
    return p


def main(argv=None) -> int:
    """
    Runs the posvit CLI entry point.
    - Pre: `argv` is None or list-like CLI argument sequence.
    - Post: Returns process exit code integer.
        Dispatches selected command and prints command outputs.
    """
    args = build_parser().parse_args(argv)

    if args.command == "models":
        for i, s in enumerate(all_models()):
            print(f"{i}  {s.safe_id:22s} {s.pe_type:16s} ape={s.use_ape!s:5s} rope={s.has_rope}")
        return 0
    if args.command == "array-spec":
        print(array_spec(COMMANDS[args.target]))
        return 0
    if args.command == "meta":
        c = COMMANDS[args.target]
        meta = {
            "scope": c.scope,
            "needs_gpu": c.needs_gpu,
            "time": c.time,
            "mem": c.mem,
            "requires": list(c.requires),
            "array": array_spec(c) if c.scope == "per-model" else "",
        }
        print(meta[args.field] if args.field else meta)
        return 0

    cmd = COMMANDS[args.command]
    if cmd.scope == "per-model" and args.index is None and args.id is None:
        print(
            f"error: `{cmd.name}` is per-model - pass --index or --id "
            f"(valid indices: {array_spec(cmd)})",
            file=sys.stderr,
        )
        return 2

    spec = None
    if args.index is not None or args.id is not None:
        spec = by_index(args.index) if args.index is not None else by_id(args.id)
        if not cmd.applies_to(spec):
            print(f"{cmd.name}: {spec.model_id} is not applicable - skipping (exit 0)")
            return 0

    from .config import load_config
    from .runners import Options

    cfg = load_config()
    opts = Options(force=args.force, device=args.device, limit=args.limit)
    result = cmd.run(spec, cfg, opts)
    return 0 if result is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())