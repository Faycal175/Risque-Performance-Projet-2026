# Calculs/eq_engine.py
import re
import numpy as np
import pandas as pd

def _norm_key(x):
    if pd.isna(x): return ""
    return re.sub(r"[^A-Z0-9]", "", str(x).upper().strip())

def build_strategic_weights_equilibre(uni_df: pd.DataFrame) -> pd.DataFrame:
    """Génère les poids cibles avec détection robuste de 'Code ISIN'."""
    # On force le renommage pour être sûr
    clean_cols = []
    for c in uni_df.columns:
        c_up = str(c).strip().upper()
        if c_up in ["CODE ISIN", "ISIN", "TICKER"]: clean_cols.append("ISIN")
        elif c_up == "NOM": clean_cols.append("Nom")
        else: clean_cols.append(c)
    uni_df.columns = clean_cols

    if "ISIN" not in uni_df.columns or "Nom" not in uni_df.columns:
        raise KeyError(f"Colonnes ISIN/Nom manquantes. Trouvées : {list(uni_df.columns)}")

    mapping = dict(zip(uni_df["Nom"].astype(str).str.strip(), uni_df["ISIN"].astype(str).str.strip()))

    models = {
        "Obligataire": [
            "EdR Fd Bond Allocation A EUR", "Sycomore Sélection Crédit R",
            "EdR Fd Emerging Sovereign A EUR (H)", "BlackRock European High Yield Bond A2 €",
            "BNPP Euro Bond Classic EUR Acc", "Amundi Funds Euro Aggr Bond A EUR C",
            "BNPP Gb Infla-Linked Bd Clc EUR Acc", "Carmignac Pf Global Bond A EUR Acc",
            "Mirova Gbl Green Bond R/A EUR", "Amundi Impact Social Bonds P"
        ],
        "Actions - Défense": [
            "Pictet Security P EUR Acc", "NIF Lux I Them Safety R/A EUR", "EdR Goldsphere A EUR"
        ],
        "Actions - Core": [
            "BlackRock European Value Fund A2", "BlackRock US Flexible Equity Fund A2",
            "Amundi Funds US Eq Fda Gr A EUR C", "EdR Fd Big Data A EUR",
            "JPMF Emerging Markets Equity A (acc) EUR", "Comgest Monde C",
            "Robeco QI Glb Dev 3D Enh Index Eq D EUR", "Amundi S&P 500 Screened Index AE Acc"
        ]
    }

    rows = []
    for name in models["Obligataire"]:
        if name in mapping: rows.append({"ISIN": mapping[name], "Nom": name, "Poids (%)": 5.0})
    for name in models["Actions - Défense"]:
        if name in mapping: rows.append({"ISIN": mapping[name], "Nom": name, "Poids (%)": 2.5})
    for name in models["Actions - Core"]:
        if name in mapping: rows.append({"ISIN": mapping[name], "Nom": name, "Poids (%)": 42.5 / 8})
    return pd.DataFrame(rows)

def compute_ex_ante_risk(asset_ret: pd.DataFrame, weights: pd.Series):
    """Moteur de calcul avec clés de sortie harmonisées."""
    r = asset_ret.copy()
    r.columns = [_norm_key(c) for c in r.columns]
    r = r.dropna(how="any")
    
    w = weights.copy()
    w.index = [_norm_key(i) for i in w.index]
    
    common = r.columns.intersection(w.index)
    if len(common) < 2:
        raise ValueError(f"Pas assez d'actifs communs ({len(common)}).")
    
    r_aligned = r[common]
    w_aligned = w.reindex(common).fillna(0)
    w_aligned = w_aligned / w_aligned.sum()

    cov_a = r_aligned.cov() * 252
    w_arr = w_aligned.values
    
    sigma2 = w_arr @ cov_a.values @ w_arr
    if hasattr(sigma2, "item"): sigma2 = sigma2.item()
    vol_pf = np.sqrt(max(sigma2, 0.0))

    mctr = (cov_a @ w_aligned) / vol_pf
    ctr = w_aligned * mctr
    rb = (ctr / vol_pf) * 100.0

    # ON UTILISE LES NOMS ATTENDUS PAR L'INTERFACE
    return {
        "vol_ex_ante": vol_pf, 
        "mctr": mctr, 
        "ctr": ctr, 
        "risk_budget_pct": rb,
        "corr": r_aligned.corr(), 
        "weights_final": w_aligned, 
        "returns_final": r_aligned
    }

def weights_from_w_data(w_data, weight_col="Poids (%)", ticker_col="ISIN"):
    if isinstance(w_data, tuple): raise TypeError("L'objet est un Tuple.")
    df = w_data.copy()
    tick = df[ticker_col].astype(str).map(_norm_key)
    poids = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
    s = pd.Series(poids.values, index=tick.values)
    return s[s > 0]