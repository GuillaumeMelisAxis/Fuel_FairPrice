from __future__ import annotations

import pandas as pd

# Metropolitan France excluding Corsica. Official INSEE region codes.
DEPARTMENT_TO_REGION = {
    # Auvergne-Rhône-Alpes
    **{d: "84" for d in ["01","03","07","15","26","38","42","43","63","69","73","74"]},
    # Bourgogne-Franche-Comté
    **{d: "27" for d in ["21","25","39","58","70","71","89","90"]},
    # Bretagne
    **{d: "53" for d in ["22","29","35","56"]},
    # Centre-Val de Loire
    **{d: "24" for d in ["18","28","36","37","41","45"]},
    # Grand Est
    **{d: "44" for d in ["08","10","51","52","54","55","57","67","68","88"]},
    # Hauts-de-France
    **{d: "32" for d in ["02","59","60","62","80"]},
    # Île-de-France
    **{d: "11" for d in ["75","77","78","91","92","93","94","95"]},
    # Normandie
    **{d: "28" for d in ["14","27","50","61","76"]},
    # Nouvelle-Aquitaine
    **{d: "75" for d in ["16","17","19","23","24","33","40","47","64","79","86","87"]},
    # Occitanie
    **{d: "76" for d in ["09","11","12","30","31","32","34","46","48","65","66","81","82"]},
    # Pays de la Loire
    **{d: "52" for d in ["44","49","53","72","85"]},
    # Provence-Alpes-Côte d'Azur
    **{d: "93" for d in ["04","05","06","13","83","84"]},
}


def department_code_from_cp(cp) -> str:
    if pd.isna(cp):
        return "UNKNOWN"
    text = "".join(ch for ch in str(cp) if ch.isdigit())
    if len(text) < 2:
        return "UNKNOWN"
    # Overseas already excluded by archive parser; Corsica is excluded too.
    return text[:2].zfill(2)


def region_code_from_department(department) -> str:
    dep = str(department).zfill(2)
    return DEPARTMENT_TO_REGION.get(dep, "UNKNOWN")


def add_admin_codes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "code_departement" not in out.columns:
        if "cp" not in out.columns:
            out["code_departement"] = "UNKNOWN"
        else:
            out["code_departement"] = out["cp"].map(department_code_from_cp)
    out["code_departement"] = out["code_departement"].fillna("UNKNOWN").astype(str)

    if "code_region" not in out.columns:
        out["code_region"] = out["code_departement"].map(region_code_from_department)
    out["code_region"] = out["code_region"].fillna("UNKNOWN").astype(str)
    return out
