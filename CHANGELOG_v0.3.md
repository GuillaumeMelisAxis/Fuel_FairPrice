# v0.3 changelog

- Live refined-product nowcast is extended to the current date with a Brent futures + EUR/USD bridge when EIA/FRED product proxies lag.
- Yahoo delayed market data is used first for the bridge; FRED is a transparent fallback.
- Explicit market freshness: `CURRENT`, `STALE`, `VERY_STALE`.
- `VERY_STALE` market data disables actionable anomaly flags (`MARKET_DATA_STALE`).
- DGEC provisional anchor dates are read from `source_status` (e.g. `provisional_to_2026-09-11`) instead of anchoring September data to September 1.
- Station prices/timestamps are parsed from the raw official `prix` field; timestamps are localized directly to `Europe/Paris`.
- Station observations older than 30 days remain available but are excluded from live scoring.
- 2025 baseline distribution margins are explicitly loaded in the live fair-price calculation.
- Market timing calibration writes `config/market_model.csv`, including the selected filter and the calibrated Brent bridge beta.
- Live CSV outputs added for the index summary, market snapshot and station scores.
