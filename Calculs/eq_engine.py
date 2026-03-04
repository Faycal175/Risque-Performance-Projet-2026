# Calculs/eq_engine.py
import re
import numpy as np
import pandas as pd

def _norm_key(x):
    
    if pd.isna(x):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(x).upper().strip())

def compute_ex_ante_risk(asset_ret: pd.DataFrame, weights: pd.Series):


    # Alignement & nettoyage
    r = asset_ret.copy()
    r.columns = [_norm_key(c) for c in r.columns]
    r = r.replace([np.inf, -np.inf], np.nan)
    r = r.dropna(how="all")  # ✅ moins agressif

    w = weights.copy()
    w.index = [_norm_key(i) for i in w.index]
    w = w.replace([np.inf, -np.inf], np.nan).dropna()
    w = w[w > 0]

    # Intersection données / poids
    common = r.columns.intersection(w.index)
    if len(common) < 2:
        raise ValueError(f"Pas assez d'actifs communs ({len(common)}). Vérifiez les ISIN dans le CSV.")

    r_aligned = r[common].copy()
    r_aligned = r_aligned.fillna(0.0)  

    w_aligned = w.reindex(common).fillna(0.0)
    w_sum = float(w_aligned.sum())
    if w_sum <= 0:
        raise ValueError("Somme des poids alignés = 0 (actifs droppés ?).")
    w_aligned = w_aligned / w_sum  

    # 3) Calcul Markowitz annualisé
    cov_a = r_aligned.cov() * 252.0
    w_arr = w_aligned.values

    sigma2 = w_arr @ cov_a.values @ w_arr
    sigma2 = float(sigma2)  

    vol_pf = float(np.sqrt(max(sigma2, 0.0)))
    if not np.isfinite(vol_pf) or vol_pf <= 0:
        raise ValueError("Volatilité ex-ante nulle/invalide (données insuffisantes).")

    # 4) Risk budgeting
    mctr = (cov_a @ w_aligned) / vol_pf
    ctr = w_aligned * mctr
    rb = (ctr / vol_pf) * 100.0

    return {
        "vol_ex_ante": vol_pf,
        "mctr": mctr,
        "ctr": ctr,
        "risk_budget_pct": rb,
        "corr": r_aligned.corr(),
        "weights_final": w_aligned,
        "returns_final": r_aligned,
    }