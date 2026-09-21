# Changelog v0.5.0

## Historical station panel

- Added official annual-archive ingestion workflow.
- Added monthly station snapshots with 30-day price-age filtering.
- Added month-specific competition and isolation features.
- Added date × fuel national-median demeaning.

## Historical local model

- Added robust additive panel model for:
  - region;
  - department;
  - road type;
  - competition;
  - isolation;
  - accessibility/logistics.
- Group effects use station-level medians.
- Shrinkage and confidence use unique station counts, not repeated monthly rows.
- Logistics classes are constrained to non-negative logistics effects.
- Added 3-month out-of-sample validation.

## Persistent station bias

- Added a shrunk station-level persistent pricing bias diagnostic.
- The station bias is deliberately excluded from the local fair price.

## Live engine

- `main.py` automatically uses `HISTORICAL_PANEL` when the trained model exists.
- Explicit v0.4.2 cross-sectional fallback remains available.
- Added model coverage and factor-confidence diagnostics.
- Added training-window metadata to live station output.

## v0.4.2.1 semantics folded into v0.5

- `local_peer_z` remains renamed to `local_anomaly_score`.
- `INSUFFICIENT` peers are now `UNASSESSED_INSUFFICIENT_PEERS`.
- Added `logistics_status = PROVISIONAL / ESTIMATED`.

## Tests

- Added historical-panel time-demeaning tests.
- Added unique-station shrinkage tests.
- Added explicit test that persistent station bias is not applied to fair price.
- Full packaged suite: 24 passing tests.
