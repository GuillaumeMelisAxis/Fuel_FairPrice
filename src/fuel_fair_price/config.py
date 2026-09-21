from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "config"

DATASET_ID = "prix-des-carburants-en-france-flux-instantane-v2"
# Opendatasoft export endpoint published by data.economie.gouv.fr.
STATION_EXPORT_URL = (
    "https://www.data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/"
    f"{DATASET_ID}/exports/csv"
    "?lang=fr&timezone=Europe%2FParis&use_labels=false&delimiter=%3B"
)

# Metropolitan INSEE region codes, excluding Corsica (94).
MAINLAND_EX_CORSICA_REGION_CODES = {
    "11", "24", "27", "28", "32", "44",
    "52", "53", "75", "76", "84", "93",
}

SUPPORTED_FUELS = {"SP95", "GAZOLE"}
