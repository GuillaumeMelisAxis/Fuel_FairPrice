# Fuel Fair Price France — v0.4

Prototype Python d'indice de **fair price SP95-E5 / Gazole** pour la France métropolitaine hors Corse.

## Architecture de l'indice

La v0.4 sépare désormais trois niveaux :

```text
1. Fair price national fondamental
2. Prime locale structurelle
3. Anomalie relative aux stations géographiquement proches
```

Le fair national reste :

```text
fair_national = (refined market component + normal distribution margin + excise) * (1 + VAT)
```

La composante raffinée utilise la logique v0.3 : ancrage DGEC, proxy raffiné quotidien, puis bridge Brent + EUR/USD si le proxy produit n'est plus à jour.

## Local adjustment v0.4

### 1. Variables locales

La v0.4 reconstruit, pour chaque carburant :

- type de station : route / autoroute (`pop` officiel : `R` / `A`);
- région;
- département;
- distance géodésique au concurrent le plus proche;
- nombre de stations vendant le même carburant dans un rayon de 2, 5, 10, 20 et 50 km;
- bucket de concurrence locale;
- bucket d'isolement.

Les distances sont des distances Haversine à vol d'oiseau, pas des distances routières.

### 2. Prime locale structurelle

Le modèle n'apprend **pas** le niveau national du spread. Il travaille sur :

```text
relative_spread_i = spread_i - median_national_spread
```

Ainsi, si tout le marché gazole est +13 c/L au-dessus du fair national, ce +13 c/L ne devient pas une "prime locale normale".

La prime locale est un modèle additif robuste :

```text
local_premium
  = region_effect
  + department_effect
  + road_type_effect
  + competition_effect
  + isolation_effect
```

Les effets sont estimés par backfitting sur des médianes robustes, avec shrinkage vers zéro pour les groupes de petite taille. Les observations extrêmes (|robust z| > 3 par défaut) ne servent pas à calibrer les effets locaux.

La prime finale est recentrée pour avoir une médiane nationale nulle.

```text
local_fair_i = fair_national + structural_local_premium_i
```

### 3. Score de voisinage

Après l'ajustement structurel :

```text
local_residual_i = station_price_i - local_fair_i
```

Chaque station est ensuite comparée à ses voisins géographiques sur ce résidu ajusté.

Le peer radius est adaptatif :

```text
10 km -> si au moins 5 pairs
20 km -> sinon
50 km -> sinon
nearest 5 stations -> fallback final
```

Le score est robuste :

```text
local_anomaly_score =
    (local_residual - median(peer residuals))
    / (1.4826 * MAD(peer residuals))
```

Flags par défaut :

- `NORMAL`;
- `HIGH` si `local_anomaly_score > 2` et résidu local > 5 c/L;
- `VERY_HIGH` si `local_anomaly_score > 3` et résidu local > 10 c/L;
- `MARKET_DATA_STALE` si la donnée marché fondamentale est trop vieille.

Cette double condition évite de signaler comme anomalie une station statistiquement différente mais seulement de quelques dixièmes de centime.

## Pourquoi conserver les deux spreads ?

La v0.4 garde simultanément :

- `spread_cent_l` : écart au fair fondamental national;
- `structural_local_premium_cent_l` : coût/prix local structurel estimé;
- `local_residual_cent_l` : écart restant après ajustement local;
- `local_anomaly_score` : caractère atypique par rapport aux stations proches.

Cela permet de distinguer :

```text
marché national cher
vs
zone locale structurellement chère
vs
station individuellement atypique
```

## Fraîcheur marché

Chaque fair price expose :

- `market_mode`: `PRODUCT_PROXY` ou `BRENT_FX_BRIDGE`;
- `bridge_source`: `YAHOO_DELAYED`, `FRED_FALLBACK` ou `FRED_PRODUCT_PROXY`;
- `effective_market_date`;
- `product_proxy_date`;
- `official_anchor_date`;
- `bridge_span_days`;
- `market_freshness`: `CURRENT`, `STALE`, `VERY_STALE`.

Si `market_freshness == VERY_STALE`, les flags d'anomalie sont désactivés.

## Données stations

Les prix SP95/Gazole et leurs timestamps sont reconstruits depuis le champ brut officiel `prix` (`@nom`, `@valeur`, `@maj`) en timezone `Europe/Paris`.

Les prix vieux de plus de 30 jours restent accessibles dans le flux brut mais ne participent pas au scoring live.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

## Calibration marché

```powershell
python scripts/build_eu_market_history.py
python scripts/run_market_calibration.py
```

## Lancer l'indice live

```powershell
python main.py
```

Sorties principales :

```text
output/live_index_summary.csv
output/live_market_snapshot.csv
output/live_station_scores.csv
output/local_adjustment_effects.csv
output/daily_refined_nowcast.csv
```

`local_adjustment_effects.csv` permet d'auditer les primes apprises par région, département, route/autoroute, concurrence et isolement.

## Limites de la v0.4

Le modèle local est un modèle transversal robuste, pas un modèle causal. En particulier :

- les distances sont géodésiques et ne capturent pas les temps de trajet;
- l'insularité n'est pas codée manuellement : elle est approchée via l'isolement géographique et la densité de concurrents;
- l'enseigne n'est pas disponible dans l'Open Data officiel;
- les effets locaux sont recalibrés sur la coupe instantanée et devront être stabilisés historiquement dans une version ultérieure.

Les scores détectent des **anomalies tarifaires**, pas une preuve juridique d'abus.

## v0.4.2 peer-score stabilisation

The local peer score uses at least 15 peers. Its standardized scale is floored at 2 c/L to avoid numerical explosions when nearby stations all quote identical or nearly identical prices:

```
local_excess = local_residual - median(peer_residuals)
peer_scale   = max(1.4826 * MAD(peer_residuals), 2.0)
local_anomaly_score = local_excess / peer_scale
```

`peer_confidence` describes the geographic strength of the comparison:

- `HIGH`: at least 30 peers within 20 km;
- `MEDIUM`: at least 15 peers within 50 km;
- `LOW`: wider-radius or nearest-N fallback;
- `INSUFFICIENT`: fewer than 15 usable peers.

Low-confidence outliers are labelled `REVIEW_LOW_CONFIDENCE` rather than automatically treated as high-confidence anomalies.


## v0.4.2 — Logistics-aware adjustment

The public local score is now named `local_anomaly_score` rather than a Z-score.
Its definition is unchanged statistically but is more accurately described as a
robust peer anomaly score:

```text
local_excess = local_residual - median(peer residuals)
peer_scale = max(1.4826 * MAD(peer residuals), 2 c/L)
local_anomaly_score = local_excess / peer_scale
```

Stations are also assigned an `accessibility_class`:

- `MAINLAND`
- `FERRY_ISLAND`
- `ROAD_CONNECTED_ISLAND`
- `REMOTE_ISOLATED`

Known island classifications are kept in `config/logistics_overrides.csv` so the
registry is auditable and extendable without changing model code. The feed does
not contain road-topology/altitude data, so the model deliberately uses
`REMOTE_ISOLATED` rather than guessing that an isolated station is mountainous.

### Logistics premium

`MAINLAND` is fixed to a logistics premium of zero. Other accessibility classes
receive an empirical cross-sectional effect relative to mainland, shrunk toward
zero. For ferry/road-island/remote classes, negative logistics effects are floored
at zero: constrained access may add logistics cost but is not interpreted as a
logistics discount. The estimate is exposed through:

- `logistics_premium_cent_l`
- `logistics_train_count`
- `logistics_confidence`

This is still a v0.4.x cross-sectional correction; the v0.5 historical panel is
needed before treating these logistics premiums as stable structural estimates.

### Logistics-aware peers

For `FERRY_ISLAND`, geographic distance to the mainland is not used as the main
peer rule. The model first tries the same `logistics_peer_group` (same island),
then other `FERRY_ISLAND` stations. Small cross-island comparisons are labelled
`LOW` confidence and can only produce `REVIEW_LOW_CONFIDENCE`, not a high-confidence
`VERY_HIGH` flag.
