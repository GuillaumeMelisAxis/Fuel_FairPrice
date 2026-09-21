# Fuel Fair Price France — v0.3

Prototype Python d'indice de **fair price SP95-E5 / Gazole** pour la France métropolitaine hors Corse.

## Ce que fait la v0.3

Le prix théorique reste :

```text
fair price = (refined market component + normal distribution margin + excise) * (1 + VAT)
```

Mais la composante raffinée est maintenant **quotidienne et auditée** :

1. Niveau officiel : dernière cotation raffinée DGEC disponible.
2. Dynamique principale : proxy raffiné quotidien public (essence / diesel NY Harbor via EIA/FRED).
3. Si ce proxy n'est plus à jour : **bridge Brent futures + EUR/USD** jusqu'à la date courante.
4. Le bridge ne change jamais le niveau d'ancrage DGEC ; il prolonge uniquement la variation depuis la dernière observation produit.
5. Si la source live Yahoo est indisponible, le code retombe sur FRED et la fraîcheur passe automatiquement en `STALE` / `VERY_STALE` selon l'âge réel.

La v0.3 corrige aussi le timestamp stations : le champ brut `prix.@maj` est interprété directement comme heure `Europe/Paris`, au lieu d'utiliser les champs dérivés `*_maj` qui introduisaient un décalage de deux heures dans le flux observé.

## Fraîcheur marché

Chaque fair price expose :

- `market_mode`: `PRODUCT_PROXY` ou `BRENT_FX_BRIDGE`;
- `bridge_source`: `YAHOO_DELAYED`, `FRED_FALLBACK` ou `FRED_PRODUCT_PROXY`;
- `effective_market_date`;
- `product_proxy_date`;
- `official_anchor_date`;
- `bridge_span_days`;
- `market_freshness`: `CURRENT`, `STALE`, `VERY_STALE`.

Règles par défaut :

- `CURRENT`: observation marché <= 1 jour ouvré, bridge <= 7 jours, ancrage DGEC <= 21 jours;
- `STALE`: observation <= 3 jours ouvrés, bridge <= 14 jours, ancrage <= 35 jours;
- `VERY_STALE`: au-delà. Dans ce cas le fair price reste affiché, mais les flags d'anomalie sont remplacés par `MARKET_DATA_STALE`.

## Données stations

Les prix SP95/Gazole et leurs timestamps sont reconstruits depuis le champ brut officiel `prix` (`@nom`, `@valeur`, `@maj`). Les prix vieux de plus de 30 jours restent dans les données brutes mais ne participent pas au scoring live.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

## Calibration du filtre marché

La Commission européenne fournit l'historique hebdomadaire français HTT. Pour le reconstruire :

```powershell
python scripts/build_eu_market_history.py
```

Puis :

```powershell
python scripts/run_market_calibration.py
```

Cette commande :

- compare proxy brut / MA3 / MA5 / MA7 hors échantillon;
- choisit un filtre par carburant;
- calibre la sensibilité de bridge à Brent;
- met à jour `config/market_model.csv`;
- produit un nowcast quotidien et son statut de fraîcheur.

## Lancer l'indice live

```powershell
python main.py
```

Sorties :

```text
output/live_index_summary.csv
output/live_market_snapshot.csv
output/live_station_scores.csv
output/daily_refined_nowcast.csv
```

## Interprétation

`spread_cent_l` mesure l'écart au fair price fondamental.

`anomaly_z` mesure le caractère atypique de la station relativement à ses pairs (`region x road_type x fuel`).

Ces métriques détectent des **anomalies tarifaires**, pas une preuve juridique d'abus.
