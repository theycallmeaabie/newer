"""Command-line entry point (`burnin`).

Commands: `version`, `generate` (synthetic data, incl. the blind holdout via BLIND_CONFIG),
`plot-examples` (healthy vs defect trajectories). Later phases add `train`, `evaluate`, `report`.

This module is allowed to touch labels.csv (CONVENTIONS.md rule 3): it drives generation,
training and evaluation, never screening.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from burnin import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="burnin", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="print the package version")

    gen = sub.add_parser("generate", help="generate a synthetic burn-in dataset")
    source = gen.add_mutually_exclusive_group()
    source.add_argument(
        "--config", type=Path, default=None, help="generator YAML (default configs/generator.yaml)"
    )
    source.add_argument(
        "--blind",
        action="store_true",
        help="read the config path from $BLIND_CONFIG (blind holdout; keep it outside the repo)",
    )
    gen.add_argument(
        "--out", type=Path, default=None, help="output directory (default: config output_dir)"
    )
    gen.add_argument("--seed", type=int, default=None, help="override the config seed")
    gen.add_argument("--lots", type=int, default=None, help="override the lot count")
    gen.add_argument("--verbose", action="store_true", help="show full tracebacks on errors")

    plot = sub.add_parser("plot-examples", help="plot healthy vs defect example trajectories")
    plot.add_argument("--data", type=Path, default=Path("data/synthetic"), help="dataset directory")
    plot.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PNG (default: <data>/example_trajectories.png)",
    )
    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from burnin.synth.config import GeneratorConfig, format_config_error, load_generator_config
    from burnin.synth.generator import generate

    path: Path | str | None = "$BLIND_CONFIG" if args.blind else args.config
    try:
        cfg, path = load_generator_config(args.config, blind=args.blind)
        if args.seed is not None or args.lots is not None:
            # Re-validate so that overrides get the same checks as the file (e.g. --lots 0).
            raw = cfg.model_dump(mode="json")
            if args.seed is not None:
                raw["seed"] = args.seed
            if args.lots is not None:
                raw["lots"]["count"] = args.lots
            cfg = GeneratorConfig.model_validate(raw)
        out_dir = args.out or Path(cfg.output_dir)
        t0 = time.perf_counter()
        ds = generate(cfg, role="blind" if args.blind else "development")
    except (ValidationError, FileNotFoundError, ImportError, ValueError, KeyError) as exc:
        if args.verbose:
            raise
        print(format_config_error(exc, path), file=sys.stderr)
        print("  (nothing was written; re-run with --verbose for the traceback)", file=sys.stderr)
        return 2
    paths = ds.write(out_dir)
    dt = time.perf_counter() - t0

    m = ds.manifest
    print(f"[generate] {m['dataset_role']} dataset from {path} (seed {m['seed']}) in {dt:.1f}s")
    print(
        f"  lots={m['lots']} parts={m['parts']} rows={m['rows']} hours={m['hours']} "
        f"lot_size={m['lot_size_min']}..{m['lot_size_max']} missing={m['missing_fraction']:.2%}"
    )
    print(
        f"  latent defects: {m['latent_defective_parts']} ({m['latent_prevalence']:.2%}, "
        f"config {cfg.defects.prevalence:.2%}) + catastrophic: {m['catastrophic_parts']} "
        f"({m['catastrophic_fraction']:.2%})"
    )
    print(f"  by type: {m['defect_type_counts']}")
    print(
        f"  latent parts that also cross a datasheet limit: {m['latent_static_fail_counts']}; "
        f"lots with whole-lot shift: {m['lots_with_whole_lot_shift']}"
    )
    for name, p in paths.items():
        print(f"  {name:13s} {p}")
    print("  NOTE: synthetic data; assumptions in data/README.md")
    if args.blind:
        print(
            "  BLIND: hand over ONLY measurements.csv and limits.yaml. Keep labels.csv, "
            "lot_events.csv, manifest.json and the config until the final evaluation."
        )
    return 0


def _cmd_plot_examples(args: argparse.Namespace) -> int:
    from burnin.synth.plots import plot_example_trajectories, plot_lot_overview

    out = args.out or (args.data / "example_trajectories.png")
    path, plotted = plot_example_trajectories(args.data, out)
    print(f"[plot-examples] wrote {path} (panels: {', '.join(plotted)})")
    overview = out.with_name("lot_overview.png")
    print(f"[plot-examples] wrote {plot_lot_overview(args.data, overview)}")
    print("  NOTE: synthetic data")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "plot-examples":
        return _cmd_plot_examples(args)
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
