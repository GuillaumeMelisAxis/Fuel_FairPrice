from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fuel_fair_price.models.historical_local import (
    fit_historical_local_model,
    save_historical_model,
    validate_historical_model,
)

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the v0.5 historical local-premium model."
    )
    parser.add_argument(
        "--panel",
        default=str(ROOT / "data" / "historical_station_panel.csv.gz"),
    )
    parser.add_argument("--holdout-months", type=int, default=3)
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
    model.to_csv(
        output_dir / "historical_model_effects.csv",
        index=False,
    )
    persistent.to_csv(
        output_dir / "station_persistent_bias.csv",
        index=False,
    )

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
                "raw_effect_cent_l",
                "n_stations",
                "n_obs",
                "confidence",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
