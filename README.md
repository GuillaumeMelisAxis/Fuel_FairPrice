# Fuel Fair Price France — V0.2

Prototype Python d'un indice de **fair price** pour le SP95-E5 et le gazole en France métropolitaine hors Corse, avec détection d'anomalies par station.

## 1. Modèle économique

Le prix théorique TTC est défini par :

```text
fair_price = (refined_quote + normal_distribution_margin + excise) * (1 + VAT)
```

La cotation du produit raffiné est utilisée plutôt que `Brent / 159` comme coût d'approvisionnement direct. Elle contient déjà la dynamique du brut et du raffinage. Le Brent reste une variable de diagnostic :

```text
brent_eur_l = brent_usd_bbl / EURUSD / 158.9873
refining_market_spread = refined_quote_eur_l - brent_eur_l
```

La marge transport-distribution implicite est :

```text
implied_margin = observed_ttc / (1 + VAT) - excise - refined_quote
```

La **marge normale** est une médiane glissante retardée : l'observation courante est exclue. Une hausse de marge au mois `t` ne peut donc pas se déclarer elle-même « normale » au mois `t`.

## 2. Sources

- **Prix stations** : Open Data officiel `prix-carburants.gouv.fr` / `donnees.roulez-eco.fr`.
- **Cotations raffinées et prix HTT** : bulletin DGEC *Cours, prix et marges des produits pétroliers en France et dans l'Union européenne*.
- **Fiscalité live** : fichier `config/taxes.csv`, à maintenir à partir des textes officiels.
- **Backtest historique** : la composante fiscale effective est reconstruite à partir des séries officielles HTT/TTC, ce qui absorbe automatiquement les dispositifs temporaires de remise.
- **Brent / EURUSD** : séries FRED/EIA et Federal Reserve, utilisées uniquement comme diagnostics.

## 3. Périmètre

- SP95 = **SP95-E5**.
- Gazole.
- France métropolitaine hors Corse.
- Route / autoroute conservé pour les comparaisons de stations.
- Le score indique une **anomalie statistique de prix**, pas une qualification juridique d'abus.

## 4. Structure

```text
config/
  taxes.csv
  baseline_margins.csv
  market_components.csv

data/
  bootstrap_national_prices_2026.csv
  bootstrap_macro_2026.csv

src/fuel_fair_price/
  data/
    stations.py
    station_archive.py
    taxes.py
  market/
    components.py
    dgec.py
    fred.py
  models/
    fair_price.py
    brent.py
    anomaly.py
  analytics/
    backtest.py
    stations.py
  visualization/
    charts.py

scripts/
  collect_dgec_history.py
  build_station_history.py
  run_bootstrap_backtest.py
  run_station_backtest.py
```

## 5. Installation

Sous Windows :

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Puis :

```bash
pytest -q
```

## 6. Rejouer le premier backtest fourni

Le repo contient un bootstrap réel 2026 construit à partir des bulletins DGEC publiés. Juin et juillet ne sont volontairement **pas interpolés** dans le bootstrap lorsque la cotation DGEC correspondante n'est pas incluse dans les données embarquées.

```bash
python scripts/run_bootstrap_backtest.py
```

Sorties :

```text
output/bootstrap_backtest_2026.csv
output/bootstrap_backtest_summary_2026.csv
output/sp95_price_vs_fair.png
output/sp95_margin.png
output/sp95_market_components.png
output/gazole_price_vs_fair.png
output/gazole_margin.png
output/gazole_market_components.png
```

Le bootstrap utilise la moyenne nationale DGEC comme `P_observed`. Pour le vrai indice station, utiliser l'étape suivante.

## 7. Construire l'historique des stations

Les archives officielles annuelles sont disponibles depuis 2007. Le script télécharge l'année demandée ainsi que l'année précédente afin de pouvoir reporter le dernier prix connu au début de janvier.

```bash
python scripts/build_station_history.py --year 2026
```

Le parser reconstruit un snapshot de fin de mois par station et carburant, puis calcule médiane, moyenne, P10, P90 et nombre de stations.

Pour construire directement les anomalies station par station :

```bash
python scripts/run_station_backtest.py --year 2026
```

## 8. Automatiser les cotations DGEC

```bash
python scripts/collect_dgec_history.py \
  --start 2025-01 \
  --end 2026-08 \
  --output config/market_components.csv
```

Pour chaque mois `M`, le collecteur tente les bulletins du mois `M+1` et récupère la colonne du mois précédent, afin d'éviter de retenir une moyenne mensuelle encore incomplète.

Les PDF sont mis en cache dans `data/cache/npg/`.

**Limite pratique :** certaines anciennes URL de bulletins peuvent être retirées ou déplacées par le site du ministère. Le collecteur ignore proprement les 404 ; une source alternative/manifest peut alors être ajoutée sans modifier le moteur de fair price.

## 9. Pourquoi le backtest reconstruit la fiscalité effective

Pour un mois historique :

```text
effective_excise = national_TTC / 1.20 - national_HTT
```

Cela permet de backtester la marge commerciale sans coder à la main toutes les mesures fiscales temporaires historiques. Pour le calcul live, on continue d'utiliser les taux légaux de `config/taxes.csv`.

## 10. Étape suivante de recherche

La V0.2 sépare déjà :

1. brut (Brent),
2. produit raffiné,
3. fiscalité,
4. transport-distribution,
5. prix station.

La prochaine amélioration économétrique naturelle est d'estimer une **prime locale attendue** par zone/station (autoroute, densité de concurrence, région, saisonnalité) avant de calculer le résidu anormal. C'est ce résidu local, et non le simple écart au prix national, qui doit servir au détecteur final.
