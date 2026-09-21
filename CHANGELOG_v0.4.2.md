# Changelog v0.4.2

## Public score terminology

- Renamed `local_peer_z` to `local_anomaly_score` throughout the live output and model API.
- Kept the robust definition: `local_excess / max(1.4826*MAD, 2 c/L)`.

## Logistics-aware adjustment

- Added `config/logistics_overrides.csv` for auditable island/accessibility overrides.
- Added `accessibility_class`, `logistics_peer_group`, and `accessibility_source`.
- Initial classes: `MAINLAND`, `FERRY_ISLAND`, `ROAD_CONNECTED_ISLAND`, `REMOTE_ISOLATED`.
- Added a shrunk empirical `logistics_premium_cent_l` relative to mainland.
- Constrained ferry/island/remote logistics premium to be non-negative.
- Added `logistics_train_count` and `logistics_confidence`.
- Preserved the original v0.4 structural premium as `base_structural_local_premium_cent_l`.

## Peer selection

- `FERRY_ISLAND` stations are no longer compared directly with nearby mainland stations.
- Same island peers are preferred; otherwise other ferry-dependent islands are used.
- Ferry-class fallback uses `LOW` confidence when the sample is small.
- Mainland and road-connected islands keep the adaptive geographic peer search.

## Tests

- Added tests for score renaming, zero-MAD stability, ferry peer isolation, non-negative logistics premiums, and Île-d'Yeu classification.
- 19 tests pass.
