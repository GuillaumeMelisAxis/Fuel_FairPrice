# Fuel Fair Price France — v0.5

Research prototype for a daily French SP95-E5 / Gazole fair-price index and
station-level anomaly diagnostics.

## What changes in v0.5

v0.4.x estimated local premiums from the current cross-section. v0.5 learns them
from an official historical **station-date panel** built from the annual
government fuel-price archives.

The live decomposition is now:

\[
P^{fair,local}_{i,t}
=
P^{fair,national}_{t}
+
L_i^{hist}
\]

where `L_i^hist` is predicted from persistent observable local factors:

- region;
- department;
- route / autoroute;
- local competition;
- isolation;
- accessibility / logistics class.

The historical target is not the raw price. For each month and fuel, v0.5 removes
the national station median:

\[
y_{i,t}
=
100\left(P_{i,t}
-
\operatorname{Median}_j P_{j,t}\right).
\]

This absorbs the common `date × fuel` price level before estimating local structure.
A national market shock therefore cannot be learned as a local premium.

## Important design choice: no station fixed effect in fair price

v0.5 estimates a `persistent_station_bias_cent_l`, but **does not add it to the
fair price**. Otherwise a station that is persistently expensive could gradually
become its own definition of "normal".

The persistent bias is exported only as a diagnostic.

## Shrinkage is based on unique stations

A ferry-island class represented by two stations observed for 18 months still has
`n_stations = 2`, not an artificial sample size of 36. Factor reliability and
shrinkage use unique station counts.

## Setup

```powershell
python -m pip install -e .
```

### 1. Build the historical panel

By default, the script starts in January 2025 and ends at the last complete month.
It automatically downloads the prior year for carry-forward of valid January
prices.

```powershell
python scripts/build_historical_panel.py
```

Outputs:

```text
data/historical_station_panel.csv.gz
output/historical_panel_summary.csv
```

Optional custom window:

```powershell
python scripts/build_historical_panel.py --start 2025-01 --end 2026-08
```

Annual archives are cached in:

```text
data/raw/annual/
```

### 2. Train and validate the historical model

```powershell
python scripts/train_historical_local_model.py
```

The last three months are used as an out-of-sample holdout diagnostic, then the
final model is refit on all available months.

Outputs:

```text
config/historical_local_model.csv
config/historical_local_model_meta.json
config/station_persistent_bias.csv

output/historical_model_validation.csv
output/historical_model_effects.csv
output/station_persistent_bias.csv
```

### 3. Run the live index

```powershell
python main.py
```

If the historical model exists, the console shows:

```text
LOCAL MODEL: HISTORICAL_PANEL v0.5
```

If it does not exist, `main.py` remains usable and explicitly falls back to:

```text
LOCAL MODEL: CROSS_SECTIONAL_FALLBACK
```

## Historical model

For each fuel, robust median backfitting estimates:

\[
L_{i,t}
=
\alpha_{\mathrm{region}}
+\alpha_{\mathrm{department}}
+\alpha_{\mathrm{road}}
+\alpha_{\mathrm{competition}}
+\alpha_{\mathrm{isolation}}
+\alpha_{\mathrm{accessibility}}.
\]

Each group effect is first estimated from **station-level median residuals** and is
then shrunk toward zero according to the number of unique stations represented.

`FERRY_ISLAND`, `ROAD_CONNECTED_ISLAND` and `REMOTE_ISOLATED` accessibility effects
are constrained to be non-negative as logistics effects.

## Live output additions

Important v0.5 fields:

```text
local_model_type
local_model_train_start
local_model_train_end
local_model_months

base_structural_local_premium_cent_l
logistics_premium_cent_l
structural_local_premium_cent_l
structural_premium_confidence
historical_model_coverage

persistent_station_bias_cent_l
persistent_bias_months

local_fair_price_eur_l
local_residual_cent_l
local_excess_cent_l
local_anomaly_score
peer_confidence
local_flag
```

`persistent_station_bias_cent_l` is informational and is never included in
`local_fair_price_eur_l`.

## Confidence semantics

For historical factor effects, confidence is based on unique stations:

- `HIGH`: at least 100 unique stations;
- `MEDIUM`: at least 30;
- `LOW`: at least 10;
- `INSUFFICIENT`: fewer than 10.

A logistics effect with insufficient support is labelled `PROVISIONAL`.

A station with no valid local peer comparison is:

```text
UNASSESSED_INSUFFICIENT_PEERS
```

—not `NORMAL`.

## Tests

```powershell
pytest -q
```

The packaged v0.5 passes the complete test suite, including tests for:

- time-effect removal;
- unique-station shrinkage;
- ferry-island confidence;
- persistent station bias not entering fair price;
- stabilized local anomaly score;
- insufficient-peer semantics;
- the existing market and station pipelines.

## Research status

v0.5 is still a research prototype, not a legal determination of excessive or
unlawful pricing. The model separates:

1. national fundamental spread;
2. historical structural local premium;
3. current local excess versus peers;
4. persistent station pricing bias as a diagnostic.
