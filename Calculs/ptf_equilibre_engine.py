# ptf_equilibre_engine.py
from __future__ import annotations

import re
import numpy as np
import pandas as pd

# on réutilise le core + helpers du fichier client
from Calculs.ptf_client_engine import _build_portfolio_from_prices


# =========================================================
# PARAMÈTRES ÉQUILIBRÉ FIXE
# =========================================================
EQUILIBRE_MACRO = {"Monétaire": 0.00, "Obligataire": 0.50, "Actions": 0.50}
EQUILIBRE_W_DEF = 0.15  # 15% de la poche actions

EQUILIBRE_MANUAL_NAMES = {
    "Monétaire": [],
    "Obligataire": [
        "EdR Fd Bond Allocation A EUR",
        "Sycomore Sélection Crédit R",
        "EdR Fd Emerging Sovereign A EUR (H)",
        "BlackRock European High Yield Bond A2 €",
        "BNPP Euro Bond Classic EUR Acc",
        "Amundi Funds Euro Aggr Bond A EUR C",
        "BNPP Gb Infla-Linked Bd Clc EUR Acc",
        "Carmignac Pf Global Bond A EUR Acc",
        "Mirova Gbl Green Bond R/A EUR",
        "Amundi Impact Social Bonds P",
    ],
    "Actions - Défense": [
        "Pictet Security P EUR Acc",
        "NIF Lux I Them Safety R/A EUR",
        "EdR Goldsphere A EUR",
    ],
    "Actions - Core": [
        "BlackRock European Value Fund A2",
        "BlackRock US Flexible Equity Fund A2",
        "Amundi Funds US Eq Fda Gr A EUR C",
        "EdR Fd Big Data A EUR",
        "JPMF Emerging Markets Equity A (acc) EUR",
        "Comgest Monde C",
        "Robeco QI Glb Dev 3D Enh Index Eq D EUR",
        "Amundi S&P 500 Screened Index AE Acc",
    ],
}


# =========================================================
# OUTILS ÉQUILIBRÉ
# =========================================================
def ucits_51040_check(weights: pd.Series) -> tuple[bool, float, float]:
    """UCITS 5/10/40 (simplifié) : max <= 10% et somme des positions > 5% <= 40%."""
    if weights is None or weights.empty:
        return False, np.nan, np.nan
    w = weights.astype(float)
    max_w = float(w.max())
    over5_sum = float(w[w > 0.05].sum())
    ok = (max_w <= 0.10 + 1e-12) and (over5_sum <= 0.40 + 1e-12)
    return ok, max_w, over5_sum


def _normalize_universe(uni: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    uni attendu avec colonnes: Nom, ISIN (optionnel: Categorie, Societe)
    retourne (uni_clean, uni_isin) avec Nom_norm.
    """
    uni = uni.copy()

    col_map = {}
    for c in uni.columns:
        cl = str(c).strip().lower()
        if cl == "nom":
            col_map[c] = "Nom"
        elif "isin" in cl:
            col_map[c] = "ISIN"
    uni = uni.rename(columns=col_map).copy()

    if not {"Nom", "ISIN"}.issubset(set(uni.columns)):
        raise ValueError("Universe doit contenir au minimum les colonnes 'Nom' et 'ISIN'.")

    uni["Nom"] = uni["Nom"].astype(str).str.strip()
    uni["ISIN"] = uni["ISIN"].astype(str).str.strip()

    uni = uni[uni["Nom"].ne("") & uni["ISIN"].ne("")].drop_duplicates(subset=["ISIN"]).copy()
    uni["Nom_norm"] = uni["Nom"].str.lower().str.strip()
    return uni, uni.set_index("ISIN")


def _resolve_name_to_isin(uni: pd.DataFrame, name: str) -> str | None:
    """Nom -> ISIN via match exact puis contains (si unique)."""
    if not isinstance(name, str) or not name.strip():
        return None
    q = name.strip().lower()

    m = uni.loc[uni["Nom_norm"] == q, "ISIN"]
    if len(m) == 1:
        return str(m.iloc[0])

    m2 = uni.loc[uni["Nom_norm"].str.contains(re.escape(q), na=False), "ISIN"]
    if len(m2) == 1:
        return str(m2.iloc[0])

    return None


def build_equilibre_portfolio(uni: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le portefeuille Équilibré (fixe) à partir des noms,
    résout en ISIN, calcule les poids, normalise si introuvables.
    Output: colonnes [Bucket, Nom, ISIN, Poids]
    """
    uni, uni_isin = _normalize_universe(uni)

    rows = []
    used_isins = set()

    # 1) Résolution noms -> ISIN
    for bucket, names in EQUILIBRE_MANUAL_NAMES.items():
        for nm in names:
            if isinstance(nm, tuple):
                nom_fonds, isin = nm[0], nm[1]
            else:
                nom_fonds, isin = nm, _resolve_name_to_isin(uni, nm)

            if isin is None:
                rows.append({"Bucket": bucket, "Nom": f"⚠️ Introuvable: {nom_fonds}", "ISIN": "", "Poids": 0.0})
                continue

            if isin in used_isins:
                continue
            used_isins.add(isin)

            nom_final = uni_isin.loc[isin, "Nom"] if isin in uni_isin.index else nom_fonds
            rows.append({"Bucket": bucket, "Nom": nom_final, "ISIN": isin, "Poids": 0.0})

    df = pd.DataFrame(rows)
    if df.empty:
        return df[["Bucket", "Nom", "ISIN", "Poids"]]

    # 2) Pondérations
    df["Poids"] = 0.0

    # Obligataire
    m_obl = df["Bucket"].eq("Obligataire")
    n_obl = int(m_obl.sum())
    if n_obl > 0:
        df.loc[m_obl, "Poids"] = EQUILIBRE_MACRO["Obligataire"] / n_obl

    # Actions split Défense / Core
    m_def = df["Bucket"].eq("Actions - Défense")
    m_core = df["Bucket"].eq("Actions - Core")
    n_def = int(m_def.sum())
    n_core = int(m_core.sum())

    w_actions = EQUILIBRE_MACRO["Actions"]
    w_def = min(EQUILIBRE_W_DEF, w_actions) if n_def > 0 else 0.0
    w_core = w_actions - w_def

    if n_def > 0:
        df.loc[m_def, "Poids"] = w_def / n_def
    if n_core > 0:
        df.loc[m_core, "Poids"] = w_core / n_core

    # 3) Normalisation finale
    s = float(df["Poids"].sum())
    if s > 0:
        df["Poids"] = df["Poids"] / s

    return df[["Bucket", "Nom", "ISIN", "Poids"]]


# =========================================================
# PIPELINE ÉQUILIBRÉ FIXE
# =========================================================
def get_clean_data_equilibre_fixed(
    *,
    INVESTMENT=1_000_000.0,
    start_target="2022-01-02",
    end_target="2026-01-30",
    benchmark_weights=(0.50, 0.50),

    DATA_UNIVERSE_PATH=r"C:\Users\Fayca\Documents\VSCode\Master272-RisquePerformance-2026\Projet\Univers-Prix.parquet",
    DATA_CLIENT_PATH=r"C:\Users\Fayca\Documents\VSCode\Master272-RisquePerformance-2026\Projet\Data.xlsb",
    FX_PATH=r"C:\Users\Fayca\Documents\VSCode\Master272-RisquePerformance-2026\Projet\forex.parquet",
    INDICES_PATH=r"C:\Users\Fayca\Documents\VSCode\Master272-RisquePerformance-2026\Projet\Indices\indice_prices.parquet",
    EAGG_PATH=r"C:\Users\Fayca\Documents\VSCode\Master272-RisquePerformance-2026\Projet\Données historiques EAGG(1).csv",

    UNIVERSE_PATH=r"C:\Users\Fayca\Documents\VSCode\Master272-RisquePerformance-2026\Projet\UniversInvestissement.xlsb",

    # holdings EQUILIBRE: liste (isin, poids_fraction) somme~1
    HOLDINGS_EQ=None,

    # optionnel: noms/buckets
    NAME_MAP=None,      # dict isin -> nom
    BUCKET_MAP=None,    # dict isin -> bucket
):
    """
    Portefeuille équilibré avec holdings FIXES (plus de CSV).
    - Si HOLDINGS_EQ est None: reconstruit depuis UNIVERSE_PATH en résolvant EQUILIBRE_MANUAL_NAMES
    - HOLDINGS_EQ : liste de tuples (ISIN, poids) somme ~1 (sinon normalisé)
    - INVESTMENT : montant total investi (sert à construire holdings en €)
    """

    # 0) Construire HOLDINGS_EQ automatiquement si absent
    if HOLDINGS_EQ is None:
        uni = pd.read_excel(UNIVERSE_PATH, sheet_name=0, engine="pyxlsb")

        col_map = {}
        for c in uni.columns:
            cl = str(c).strip().lower()
            if cl == "nom":
                col_map[c] = "Nom"
            elif "isin" in cl:
                col_map[c] = "ISIN"
        uni = uni.rename(columns=col_map).copy()

        if not {"Nom", "ISIN"}.issubset(set(uni.columns)):
            raise ValueError("UniversInvestissement.xlsb: colonnes 'Nom' et 'ISIN' introuvables.")

        uni["Nom"] = uni["Nom"].astype(str).str.strip()
        uni["ISIN"] = uni["ISIN"].astype(str).str.strip()
        uni = uni[uni["Nom"].ne("") & uni["ISIN"].ne("")].drop_duplicates("ISIN")

        df_eq_auto = build_equilibre_portfolio(uni)

        df_eq_auto["ISIN"] = df_eq_auto["ISIN"].astype(str).str.strip()
        df_eq_auto["Poids"] = pd.to_numeric(df_eq_auto["Poids"], errors="coerce")
        df_eq_auto = df_eq_auto.dropna(subset=["ISIN", "Poids"])
        df_eq_auto = df_eq_auto[df_eq_auto["ISIN"].ne("") & (df_eq_auto["Poids"] > 0)].copy()

        if df_eq_auto.empty:
            dbg = build_equilibre_portfolio(uni)
            introuvables = dbg[dbg["ISIN"].astype(str).str.strip().eq("")].copy()
            msg = "Portefeuille équilibré vide: aucun nom n'a été résolu en ISIN."
            if not introuvables.empty:
                msg += "\nExemples introuvables:\n" + "\n".join(introuvables["Nom"].head(10).astype(str).tolist())
            raise ValueError(msg)

        HOLDINGS_EQ = list(zip(df_eq_auto["ISIN"].astype(str), df_eq_auto["Poids"].astype(float)))

        if NAME_MAP is None:
            NAME_MAP = dict(zip(df_eq_auto["ISIN"].astype(str), df_eq_auto["Nom"].astype(str)))
        if BUCKET_MAP is None:
            BUCKET_MAP = dict(zip(df_eq_auto["ISIN"].astype(str), df_eq_auto["Bucket"].astype(str)))

    # 1) Construire holdings € depuis HOLDINGS_EQ
    df_eq = pd.DataFrame(HOLDINGS_EQ, columns=["ISIN", "Poids"]).copy()
    df_eq["ISIN"] = df_eq["ISIN"].astype(str).str.strip()

    df_eq["Poids"] = (
        df_eq["Poids"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df_eq["Poids"] = pd.to_numeric(df_eq["Poids"], errors="coerce")
    df_eq = df_eq.dropna(subset=["ISIN", "Poids"])
    df_eq = df_eq[df_eq["ISIN"].ne("") & (df_eq["Poids"] > 0)].copy()

    if not df_eq.empty and df_eq["Poids"].max() > 1.5:
        df_eq["Poids"] = df_eq["Poids"] / 100.0

    s = float(df_eq["Poids"].sum())
    if s <= 0:
        raise ValueError("HOLDINGS_EQ: somme des poids <= 0 (HOLDINGS_EQ vide ou poids illisibles).")

    df_eq["Poids"] = df_eq["Poids"] / s

    holdings = pd.DataFrame({
        "ticker": df_eq["ISIN"],
        "val": df_eq["Poids"] * float(INVESTMENT)
    }).set_index("ticker")

    # 2) Charger prices_long (univers parquet + client xlsb long)
    df_u = pd.read_parquet(DATA_UNIVERSE_PATH).copy()
    df_u.columns = [str(c).strip().lower() for c in df_u.columns]
    frames = [df_u]

    try:
        df_c = pd.read_excel(DATA_CLIENT_PATH, engine="pyxlsb")
        df_c.columns = [str(c).strip().lower() for c in df_c.columns]
        frames.append(df_c)
    except Exception:
        pass

    prices_long = pd.concat(frames, ignore_index=True)

    fx_long = pd.read_parquet(FX_PATH).copy()

    # 3) Core pipeline
    core = _build_portfolio_from_prices(
        prices_long=prices_long,
        fx_long=fx_long,
        holdings=holdings,
        start_target=start_target,
        end_target=end_target,
        benchmark_weights=benchmark_weights,
        indices_path=INDICES_PATH,
        eagg_path=EAGG_PATH,
        benchmark_code="NQGIN",
    )

    # 4) w_data (affichage)
    NAME_MAP = {} if NAME_MAP is None else NAME_MAP
    BUCKET_MAP = {} if BUCKET_MAP is None else BUCKET_MAP
    tickers_final = core["w_final"].index.astype(str).tolist()

    perf_c = core["perf_c"]
    if not isinstance(perf_c, pd.Series):
        perf_c = pd.Series(perf_c, index=core["w_final"].index)

    risk_c = core["risk_c"]
    if not isinstance(risk_c, pd.Series):
        risk_c = pd.Series(risk_c, index=core["w_final"].index)

    w_data = (
        pd.DataFrame({
            "ISIN": tickers_final,  # ✅ indispensable pour l’ex-ante
            "Nom du Fonds": [NAME_MAP.get(t, t) for t in tickers_final],
            "Bucket": [BUCKET_MAP.get(t, "") for t in tickers_final],
            "Exposition (€)": core["exposure_eur"].reindex(tickers_final).astype(float).values,
            "Poids (%)": (core["w_final"].reindex(tickers_final) * 100).round(2).values,
            "Contr. Perf (%)": perf_c.reindex(tickers_final).astype(float).round(2).values,
            "Contr. Risque (%)": risk_c.reindex(tickers_final).astype(float).round(2).values,
        })
        .sort_values("Poids (%)", ascending=False)
        .reset_index(drop=True)
    )

    if w_data["Bucket"].astype(str).str.strip().ne("").any():
        geo_df = (
            w_data.assign(P=lambda d: d["Poids (%)"] / 100.0)
            .groupby("Bucket", as_index=False)["P"].sum()
            .rename(columns={"Bucket": "Géo"})
        )
    else:
        geo_df = pd.DataFrame({"Géo": ["Portefeuille"], "P": [1.0]})

    sec_df = pd.DataFrame({"Sec": ["N/A"], "P": [1.0]})

    return (
        core["pf_v"], core["bm_v"],
        core["ann_p"], core["ann_b"],
        core["r_p"], core["r_b"],
        w_data, geo_df, sec_df,
        core["mu_p"], core["vol_p"],
        core["missing_isins"],
        core["asset_ret"],
    )