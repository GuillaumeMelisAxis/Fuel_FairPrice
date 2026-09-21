# v0.4.1 — Stabilised local peer scoring

This maintenance release fixes the unstable local peer Z-scores observed in v0.4.

## Changes

- Minimum local peer sample increased from **5 to 15**.
- Adaptive peer radii are now **10 / 20 / 50 / 100 km** before nearest-N fallback.
- Robust peer scale now has a **2 c/L floor**:

  `scale = max(1.4826 * MAD, 2.0 c/L)`

- Added `local_excess_cent_l`:

  `local residual - median local residual of peers`.

- Added `local_peer_scale_cent_l` for auditability.
- Added `peer_confidence`: `HIGH`, `MEDIUM`, `LOW`, `INSUFFICIENT`.
- Added `peer_selection_method`: `RADIUS`, `NEAREST_N`, `NONE`.
- Low-confidence anomalies are surfaced as `REVIEW_LOW_CONFIDENCE` instead of being automatically promoted to `HIGH` / `VERY_HIGH`.
- The main output now shows local excess, stabilised scale, peer confidence and peer selection method.
- Added regression tests ensuring zero-MAD peer groups cannot create explosive Z-scores.

The structural local premium model itself is unchanged from v0.4.
