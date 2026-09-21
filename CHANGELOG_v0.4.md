# Changelog — v0.4.0

## Local adjustment

- Ajout de `models/local_adjustment.py`.
- Conversion robuste des coordonnées PTV_GEODECIMAL / WGS84.
- Calcul Haversine chunké, testé sur ~8 000 stations sans matrice NxN en mémoire.
- Nouvelles features :
  - `nearest_station_km`;
  - `stations_within_2km`;
  - `stations_within_5km`;
  - `stations_within_10km`;
  - `stations_within_20km`;
  - `stations_within_50km`;
  - `competition_bucket`;
  - `isolation_bucket`;
  - `isolated_market`.
- Modèle additif robuste de prime locale : région + département + route/autoroute + concurrence + isolement.
- Calibration sur le spread relatif à la médiane nationale pour ne pas absorber un écart national généralisé.
- Exclusion des outliers extrêmes de la calibration des effets locaux.
- Shrinkage des effets de petits groupes vers zéro.
- Recentrage de la prime locale à médiane nulle.

## Local peer scoring

- Pairing géographique adaptatif 10/20/50 km.
- Fallback vers les 5 voisins les plus proches si nécessaire.
- `local_peer_z` robuste par MAD.
- `local_flag` avec double condition statistique + économique.

## Outputs

Ajout de :

```text
output/local_adjustment_effects.csv
```

`live_station_scores.csv` contient maintenant notamment :

```text
structural_local_premium_cent_l
local_fair_price_eur_l
local_residual_cent_l
nearest_station_km
stations_within_10km
local_peer_radius_km
local_peer_count
local_peer_z
local_flag
```

## Validation

- 13 tests unitaires passent.
- Benchmark géographique synthétique sur 8 150 stations : ~2.3 s dans l'environnement de test.
