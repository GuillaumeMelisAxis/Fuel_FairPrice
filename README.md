# Fuel Fair Price France — V1

Prototype d'un indice de prix théorique pour le SP95 (E5) et le gazole en France métropolitaine hors Corse.

## Principe

Le benchmark recommandé n'ajoute pas directement le Brent au prix raffiné :

`fair TTC = (cotation produit raffiné + marge transport/distribution normale + accise) * (1 + TVA)`

La cotation du produit raffiné contient déjà l'économie du pétrole brut et du raffinage. Le Brent est conservé comme variable explicative/diagnostique (crack spread), afin d'éviter un double comptage.

La marge transport/distribution "normale" est une médiane glissante historique et non la marge observée de la semaine courante. Ainsi, une hausse ponctuelle de marge ne devient pas automatiquement le nouveau niveau "fair".

## Sources prévues

- Stations : flux officiel DGCCRF / prix-carburants.gouv.fr, actualisé fréquemment.
- Cotations raffinées et marges : bulletin hebdomadaire DGEC "Cours, prix et marges des produits pétroliers".
- Fiscalité : Code des impositions sur les biens et services / Légifrance.
- TVA : taux métropolitain normal.
- Brent / EURUSD : utilisés comme diagnostics et variables de modélisation, pas ajoutés à la cotation raffinée.

## Périmètre V1

- SP95 = SP95-E5 dans les publications DGEC.
- Gazole.
- France métropolitaine hors Corse.
- Séparation route / autoroute pour le score d'anomalie.
- Score robuste par région, type de route et carburant.

## Donnée marché à renseigner

Remplir `config/market_components.csv` avec des observations hebdomadaires :

- `date`
- `fuel` (`SP95` ou `GAZOLE`)
- `refined_quote_eur_l`
- `observed_distribution_margin_eur_l`
- `source_url`

Pour SP95-E5, si le bulletin fournit directement `prix HTT France` et `cotation`, calculer :

`observed_distribution_margin_eur_l = prix_HTT_eur_l - refined_quote_eur_l`

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -e .
python main.py
```

## Étape suivante

Automatiser l'ingestion hebdomadaire du bulletin DGEC, ajouter l'historique station et estimer une prime locale par station/zone plutôt que de s'arrêter à un score de pairs.
