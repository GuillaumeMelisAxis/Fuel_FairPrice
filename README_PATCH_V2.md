# Market timing patch v2

The first patch attempted to reconstruct DGEC history by guessing historical
NPG PDF URLs. That is not reliable because old DGEC PDFs are no longer served
even though search engines may still index them.

This v2 uses sources that are actually maintained:

1. European Commission Weekly Oil Bulletin:
   historical France weekly prices without taxes (2005 onward).
2. EIA/FRED daily refined-product spot proxies:
   - SP95 proxy: DGASNYH (NY Harbor conventional gasoline)
   - Gazole proxy: DDFUELNYH (NY Harbor ULSD)
3. Existing DGEC official refined-product quote(s) already in
   config/market_components.csv, used to anchor the current level.

The US spot series are NOT claimed to be French/European refined quotations.
They are used only as public, high-frequency product-level proxies for daily
movements between two official DGEC observations.

## Dependencies

Ensure these are available:

    pandas
    requests
    openpyxl
    numpy

No PDF parser is needed for this patch.

## Install files

Copy/replace:

    src/fuel_fair_price/market/eu_history.py
    src/fuel_fair_price/market/daily_proxy.py
    src/fuel_fair_price/models/lag_selection.py
    scripts/build_eu_market_history.py
    scripts/run_market_calibration.py

The old scripts/build_dgec_weekly.py can be removed or left unused.

## Run

First:

    python scripts/build_eu_market_history.py

Expected result: several hundred weekly France observations since 2020,
not 2 rows.

Then:

    python scripts/run_market_calibration.py

This:
- tests raw / MA3 / MA5 / MA7 product proxies out-of-sample;
- chooses one filter per fuel;
- anchors the selected daily proxy to the latest official DGEC refined quote;
- writes:
    output/market_filter_scores.csv
    output/daily_refined_nowcast.csv

## Interpretation

The final nowcast is:

    Q_nowcast(t)
      = Q_DGEC(anchor)
        * proxy_filter(t) / proxy_filter(anchor)

so the DGEC observation fixes the level while the public daily spot proxy
provides only the short-term movement.
