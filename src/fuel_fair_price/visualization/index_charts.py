from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _fuel_frame(index_series: pd.DataFrame, fuel: str) -> pd.DataFrame:
    return (
        index_series[index_series["fuel"].astype(str).str.upper() == fuel.upper()]
        .sort_values("date")
        .copy()
    )


def plot_observed_vs_fair(index_series: pd.DataFrame, fuel: str, output: str | Path) -> Path:
    g = _fuel_frame(index_series, fuel)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(g["date"], g["observed_median_price_eur_l"], marker="o", label="Observed median pump price")
    ax.plot(g["date"], g["fundamental_fair_price_eur_l"], marker="o", label="Fundamental fair price")
    ax.set_title(f"{fuel} — observed price vs fundamental fair price")
    ax.set_ylabel("EUR/L")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_observed_indices(index_series: pd.DataFrame, output: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for fuel in ("SP95", "GAZOLE"):
        g = _fuel_frame(index_series, fuel)
        if not g.empty:
            ax.plot(g["date"], g["observed_price_index"], marker="o", label=fuel)
    ax.axhline(100.0, linewidth=1.0, alpha=0.5)
    ax.set_title("Observed fuel-price indices — base 100")
    ax.set_ylabel("Index (base 100 at first observation)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_market_tension(index_series: pd.DataFrame, output: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for fuel in ("SP95", "GAZOLE"):
        g = _fuel_frame(index_series, fuel)
        if not g.empty:
            ax.plot(g["date"], g["market_tension_cent_l"], marker="o", label=fuel)
    ax.axhline(0.0, linewidth=1.0, alpha=0.6)
    ax.set_title("Market tension — observed median minus fundamental fair price")
    ax.set_ylabel("c€/L")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_local_anomaly_rate(index_series: pd.DataFrame, output: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for fuel in ("SP95", "GAZOLE"):
        g = _fuel_frame(index_series, fuel)
        if not g.empty:
            ax.plot(g["date"], g["local_anomaly_rate_pct"], marker="o", label=fuel)
    ax.set_title("Local anomaly rate")
    ax.set_ylabel("Share of assessed stations (%)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output
