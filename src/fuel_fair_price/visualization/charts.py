from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _monthly_frame(backtest: pd.DataFrame, fuel: str) -> pd.DataFrame:
    g = backtest[backtest["fuel"] == fuel].sort_values("date").copy()
    if g.empty:
        return g
    idx = pd.date_range(g["date"].min(), g["date"].max(), freq="MS")
    return g.set_index("date").reindex(idx).rename_axis("date").reset_index()


def plot_price_vs_fair(backtest: pd.DataFrame, fuel: str, output: str | Path) -> Path:
    g = _monthly_frame(backtest, fuel)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(g["date"], g["national_ttc_eur_l"], marker="o", label="Prix national observé TTC")
    ax.plot(g["date"], g["fair_price_eur_l"], marker="o", label="Fair price")
    ax.set_title(f"{fuel} — prix observé vs fair price")
    ax.set_ylabel("EUR/L")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_margin(backtest: pd.DataFrame, fuel: str, output: str | Path) -> Path:
    g = _monthly_frame(backtest, fuel)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(g["date"], 100 * g["implied_distribution_margin_eur_l"], marker="o", label="Marge implicite")
    ax.plot(g["date"], 100 * g["normal_distribution_margin_eur_l"], marker="o", label="Marge normale retardée")
    ax.set_title(f"{fuel} — marge transport-distribution")
    ax.set_ylabel("c€/L (HT)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_market_components(backtest: pd.DataFrame, fuel: str, output: str | Path) -> Path:
    g = _monthly_frame(backtest, fuel)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(g["date"], 100 * g["refined_quote_eur_l"], marker="o", label="Cotation produit raffiné")
    if "brent_eur_l" in g:
        ax.plot(g["date"], 100 * g["brent_eur_l"], marker="o", label="Brent équivalent")
    ax.set_title(f"{fuel} — produit raffiné et Brent")
    ax.set_ylabel("c€/L")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
