# Fuel Fair Price — market timing patch

This patch adds a defensible high-frequency market layer without inventing
daily European refined-product observations.

## Method

1. DGEC weekly NPG bulletins are the official refined-product anchors.
2. Weekly Eurosuper / Gazole quotes are collected in USD/t.
3. Daily Brent and EUR/USD are downloaded from FRED.
4. Product sensitivity to Brent is estimated from historical weekly log changes.
5. Between two DGEC releases:

   Q_hat(t) = Q_anchor * (Brent(t) / Brent_anchor)^beta

6. Q_hat is converted to EUR/L.
7. 3-, 5- and 7-calendar-day moving averages are constructed.
8. The candidate filters are compared out-of-sample against DGEC weekly HTT
   national prices.

## Files

Copy into the existing repository:

- src/fuel_fair_price/market/daily_proxy.py
- src/fuel_fair_price/models/lag_selection.py
- scripts/build_dgec_weekly.py
- scripts/run_market_calibration.py

## Dependency

Add:

    pdfplumber>=0.11

to pyproject.toml / requirements.

## Run

From the activated venv at repo root:

    python scripts/build_dgec_weekly.py

Then:

    python scripts/run_market_calibration.py

Generated:

- config/dgec_weekly_market.csv
- output/daily_refined_nowcast.csv
- output/market_filter_scores.csv

## Important

The PDF parser relies on the current DGEC NPG layout. If the ministry changes
the PDF tables, the collector will print PARSE ERROR rather than silently
creating bad data.
