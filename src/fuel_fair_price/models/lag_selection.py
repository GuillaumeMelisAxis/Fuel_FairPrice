from __future__ import annotations

import numpy as np
import pandas as pd


CANDIDATES = (
    "proxy_eur_l",
    "proxy_ma3_eur_l",
    "proxy_ma5_eur_l",
    "proxy_ma7_eur_l",
)


def build_calibration_panel(
    weekly_htt: pd.DataFrame,
    daily_proxy: pd.DataFrame,
) -> pd.DataFrame:
    panels = []

    for fuel, weekly in weekly_htt.groupby("fuel"):
        daily = (
            daily_proxy[daily_proxy["fuel"] == fuel]
            .sort_values("date")
        )

        cols = ["date", *CANDIDATES]

        panel = pd.merge_asof(
            weekly.sort_values("date"),
            daily[cols].sort_values("date"),
            on="date",
            direction="backward",
        )

        panels.append(panel)

    return pd.concat(
        panels,
        ignore_index=True,
    )


def _fit_linear(x: np.ndarray, y: np.ndarray):
    X = np.column_stack(
        [np.ones(len(x)), x]
    )
    coef, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )
    return float(coef[0]), float(coef[1])


def score_filters(
    panel: pd.DataFrame,
    *,
    train_fraction: float = 0.75,
    min_train: int = 52,
) -> pd.DataFrame:
    """
    Select the timing/smoothing filter out-of-sample.

    HTT is regressed on the public product proxy on TRAIN:
        HTT = alpha + beta * proxy

    The ranking is based on TEST RMSE/MAE.

    We use the proxy only to select short-run dynamics; the final daily
    refined-price level is later re-anchored to the official DGEC quote.
    """
    rows = []

    for fuel, group in panel.groupby("fuel"):
        group = group.sort_values("date").reset_index(drop=True)

        n = len(group)
        split = max(
            min_train,
            int(n * train_fraction),
        )
        split = min(split, n - 1)

        if split <= 0 or n - split <= 0:
            continue

        train = group.iloc[:split]
        test = group.iloc[split:]

        for candidate in CANDIDATES:
            tr = train[
                ["htt_eur_l", candidate]
            ].dropna()
            te = test[
                ["htt_eur_l", candidate]
            ].dropna()

            if len(tr) < min_train or te.empty:
                continue

            alpha, beta = _fit_linear(
                tr[candidate].to_numpy(),
                tr["htt_eur_l"].to_numpy(),
            )

            pred = (
                alpha
                + beta * te[candidate].to_numpy()
            )
            actual = te["htt_eur_l"].to_numpy()

            err = actual - pred

            rows.append(
                {
                    "fuel": fuel,
                    "candidate": candidate,
                    "alpha": alpha,
                    "beta": beta,
                    "rmse_eur_l": float(
                        np.sqrt(np.mean(err ** 2))
                    ),
                    "mae_eur_l": float(
                        np.mean(np.abs(err))
                    ),
                    "n_train": len(tr),
                    "n_test": len(te),
                }
            )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values(
        ["fuel", "rmse_eur_l", "mae_eur_l"]
    ).reset_index(drop=True)


def best_filter_by_fuel(
    scores: pd.DataFrame,
) -> dict[str, str]:
    if scores.empty:
        return {}

    best = (
        scores
        .sort_values(["fuel", "rmse_eur_l"])
        .groupby("fuel", as_index=False)
        .first()
    )

    return dict(
        zip(
            best["fuel"],
            best["candidate"],
        )
    )
