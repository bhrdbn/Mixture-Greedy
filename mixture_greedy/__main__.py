"""Command-line interface for metric-specific mixture evaluation."""

from __future__ import annotations

import argparse

from mixture_greedy.runner import (
    SUPPORTED_METRICS,
    SUPPORTED_MODES,
    create_config,
    run_evaluation,
)


def build_parser(default_metric: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an online model mixture with RKE, KID, FID, or VNE."
    )
    if default_metric is None:
        parser.add_argument("--metric", required=True, choices=SUPPORTED_METRICS)
    else:
        parser.set_defaults(metric=default_metric)
    parser.add_argument("--dataset", required=True, help="Dataset identifier")
    parser.add_argument(
        "--models",
        required=True,
        nargs="+",
        help="Model/arm names separated by spaces",
    )
    mode_choices = (
        SUPPORTED_MODES[default_metric]
        if default_metric is not None
        else sorted({mode for modes in SUPPORTED_MODES.values() for mode in modes})
    )
    parser.add_argument(
        "--mode",
        default="mixture-greedy",
        choices=mode_choices,
        help="Sampling/optimization strategy (availability depends on metric)",
    )
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--feature-extractor")
    parser.add_argument(
        "--real-dataset",
        help="Real-feature NPZ path; required for FID",
    )
    parser.add_argument("--oracle-alphas", nargs="+", type=float)
    parser.add_argument("--output", help="Optional output NPZ path")

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--optimizer", choices=("eg", "scipy"))
    optimization.add_argument("--eg-eta", type=float)
    optimization.add_argument("--eg-steps", type=int)
    optimization.add_argument(
        "--eg-sqrt-decay",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    optimization.add_argument("--alpha-floor", type=float)

    exploration = parser.add_argument_group("exploration")
    exploration.add_argument("--initial-samples", type=int)
    exploration.add_argument("--mini-batch", type=int)
    exploration.add_argument("--epsilon", type=float)
    exploration.add_argument("--epsilon-decay", type=float)

    vne = parser.add_argument_group("VNE")
    vne.add_argument("--kernel-type", choices=("cosine", "rff"))
    vne.add_argument("--rff-features", type=int)
    vne.add_argument("--rff-sigma", type=float)

    fid = parser.add_argument_group("FID")
    fid.add_argument(
        "--track-mixture-fid",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    fid.add_argument("--mixture-fid-samples", type=int)

    adaptive = parser.add_argument_group("adaptive datasets")
    adaptive.add_argument(
        "--adaptive",
        choices=("none", "change-point", "male-female", "cats-birds"),
        default="none",
    )
    adaptive.add_argument("--change-point-path")
    adaptive.add_argument("--adaptive-data-path")
    adaptive.add_argument("--change-round", type=int)
    return parser


def configure_from_args(config, args) -> None:
    """Apply only explicitly supplied CLI overrides to a metric config."""
    direct_options = {
        "feature_extractor": "feature_extractor",
        "optimizer": "OPTIMIZER",
        "eg_eta": "EG_ETA",
        "eg_sqrt_decay": "EG_SQRT_DECAY",
        "alpha_floor": "ALPHA_FLOOR",
        "initial_samples": "INITIAL_SAMPLE_COUNT",
        "mini_batch": "MINI_BATCH",
        "epsilon": "EPSILON",
        "epsilon_decay": "EPSILON_DECAY",
        "kernel_type": "kernel_type",
        "rff_features": "rff_n",
        "rff_sigma": "rff_sigma",
        "track_mixture_fid": "TRACK_MIXTURE_FID",
        "mixture_fid_samples": "MIXTURE_FID_SAMPLES",
    }
    for argument, setting in direct_options.items():
        value = getattr(args, argument)
        if value is not None:
            if not hasattr(config, setting):
                raise ValueError(f"--{argument.replace('_', '-')} is not valid for this metric")
            setattr(config, setting, value)

    if args.eg_steps is not None:
        setting = "EG_STEPS_PER_ROUND" if hasattr(config, "EG_STEPS_PER_ROUND") else "EG_STEPS"
        if not hasattr(config, setting):
            raise ValueError("--eg-steps is not valid for this metric")
        setattr(config, setting, args.eg_steps)

    if args.adaptive != "none":
        _configure_adaptive_dataset(config, args)


def _configure_adaptive_dataset(config, args) -> None:
    setting_prefix = args.adaptive.replace("-", "_").upper()
    enabled_setting = f"{setting_prefix}_ADAPTIVE"
    if args.adaptive == "change-point":
        enabled_setting = "CHANGE_POINT"
    if not hasattr(config, enabled_setting):
        raise ValueError(f"adaptive mode {args.adaptive!r} is not valid for this metric")

    for setting in ("CHANGE_POINT", "MALE_FEMALE_ADAPTIVE", "CATS_BIRDS_ADAPTIVE"):
        if hasattr(config, setting):
            setattr(config, setting, setting == enabled_setting)

    if args.change_point_path is not None:
        config.CHANGE_POINT_PATH = args.change_point_path
    if args.adaptive_data_path is not None:
        setattr(config, f"{setting_prefix}_DATA_PATH", args.adaptive_data_path)
    if args.change_round is not None and args.adaptive != "change-point":
        setattr(config, f"{setting_prefix}_CHANGE_ROUND", args.change_round)


def main(argv=None, *, default_metric: str | None = None) -> int:
    args = build_parser(default_metric=default_metric).parse_args(argv)
    config = create_config(args.metric)
    try:
        configure_from_args(config, args)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    result = run_evaluation(
        args.metric,
        args.models,
        args.dataset,
        rounds=args.rounds,
        mode=args.mode,
        config=config,
        real_dataset_path=args.real_dataset,
        oracle_alphas=args.oracle_alphas,
    )
    if args.output:
        result.save(args.output)

    print(f"metric: {result.metric}")
    print(f"sample sizes: {result.sample_sizes.tolist()}")
    if len(result.alpha_history):
        print(f"final alphas: {result.alpha_history[-1].tolist()}")
    if len(result.scores):
        print(f"final score: {result.scores[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
