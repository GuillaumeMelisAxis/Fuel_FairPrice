from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fuel_fair_price.models.historical_local import (
    fit_historical_local_model,
    save_historical_model,
    validate_factor_ablation,
    validate_historical_model,
    validate_regime_splits,
    ACTIVE_FACTORS_BY_FUEL,
    DIAGNOSTIC_FACTORS_BY_FUEL,
)

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the v0.5.1 historical local-premium model."
    )
    parser.add_argument(
        "--panel",
        default=str(ROOT / "data" / "historical_station_panel.csv.gz"),
    )
    parser.add_argument("--holdout-months", type=int, default=3)
    parser.add_argument(
        "--shock-start",
        default="2026-02-28",
        help="Energy-regime break used for regime-aware validation.",
    )
    parser.add_argument("--transition-months", type=int, default=4)
    parser.add_argument("--pre-shock-holdout-months", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} does not exist. Run scripts/build_historical_panel.py first."
        )

    panel = pd.read_csv(
        panel_path,
        parse_dates=["date", "updated_at"],
        dtype={
            "station_id": "string",
            "id": "string",
            "cp": "string",
            "code_departement": "string",
            "code_region": "string",
        },
    )

    print(
        f"Panel: {len(panel):,} rows | "
        f"{panel['date'].nunique()} months | "
        f"{panel['station_id'].nunique():,} unique stations"
    )

    validation = validate_historical_model(
        panel,
        holdout_months=int(args.holdout_months),
    )
    print("\nOUT-OF-SAMPLE VALIDATION")
    if validation.empty:
        print("Not enough months for holdout validation.")
    else:
        print(validation.to_string(index=False))

    regime_validation = validate_regime_splits(
        panel,
        shock_start=args.shock_start,
        pre_shock_holdout_months=int(args.pre_shock_holdout_months),
        transition_months=int(args.transition_months),
        current_holdout_months=int(args.holdout_months),
    )
    print("\nREGIME-AWARE VALIDATION")
    if regime_validation.empty:
        print("Not enough history for regime-aware validation.")
    else:
        print(regime_validation.to_string(index=False))

    ablation = validate_factor_ablation(
        panel,
        holdout_months=int(args.holdout_months),
    )
    print("\nFACTOR ABLATION")
    if ablation.empty:
        print("Not enough history for factor ablation.")
    else:
        print(ablation.to_string(index=False))

    model, meta, persistent = fit_historical_local_model(panel)

    save_historical_model(
        model,
        meta,
        persistent,
        model_path=ROOT / "config" / "historical_local_model.csv",
        meta_path=ROOT / "config" / "historical_local_model_meta.json",
        persistent_path=ROOT / "config" / "station_persistent_bias.csv",
    )

    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    validation.to_csv(
        output_dir / "historical_model_validation.csv",
        index=False,
    )
    regime_validation.to_csv(
        output_dir / "historical_model_regime_validation.csv",
        index=False,
    )
    ablation.to_csv(
        output_dir / "historical_model_ablation.csv",
        index=False,
    )
    model.to_csv(
        output_dir / "historical_model_effects.csv",
        index=False,
    )
    persistent.to_csv(
        output_dir / "station_persistent_bias.csv",
        index=False,
    )

    print("\nFINAL PRODUCTION FACTOR POLICY")
    for fuel in sorted(ACTIVE_FACTORS_BY_FUEL):
        print(f"{fuel}: active={list(ACTIVE_FACTORS_BY_FUEL[fuel])}")
        print(f"{fuel}: diagnostic_only={list(DIAGNOSTIC_FACTORS_BY_FUEL.get(fuel, ())) }")

    print("\nMODEL SAVED")
    print(ROOT / "config" / "historical_local_model.csv")
    print(ROOT / "config" / "historical_local_model_meta.json")
    print(ROOT / "config" / "station_persistent_bias.csv")

    for fuel, info in meta["fuels"].items():
        print(
            f"{fuel}: {info['train_start']} -> {info['train_end']} | "
            f"{info['n_months']} months | "
            f"{info['n_unique_stations']:,} stations | "
            f"{info['n_observations']:,} observations"
        )

    print("\nLOGISTICS EFFECTS")
    logistics = model[model["factor"] == "accessibility_class"]
    print(
        logistics[
            [
                "fuel",
                "level",
                "effect_cent_l",
                "provisional_effect_cent_l",
                "raw_effect_cent_l",
                "n_stations",
                "n_obs",
                "confidence",
                "status",
                "constraint_applied",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
