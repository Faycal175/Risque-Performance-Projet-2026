# Calculs/ptf_client_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================
# 0) HELPERS
# =========================================================
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # tolérances fréquentes
    if "date" not in df.columns:
        for alt in ["dates", "datetime", "jour", "day"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "date"})
                break
    if "ticker" not in df.columns:
        for alt in ["isin", "code", "id", "instrument"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "ticker"})
                break
    if "close" not in df.columns:
        for alt in ["price", "last", "dernier", "vl", "nav"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "close"})
                break
    return df


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    """
    Gère dates mixtes:
    - datetime déjà OK
    - texte ("14/06/2019", "2022-01-05", etc.)
    - excel serial (43630)
    """
    s = series.copy()

    # Déjà datetime
    if np.issubdtype(s.dtype, np.datetime64):
        out = pd.to_datetime(s, errors="coerce")
        return out.dt.normalize()

    # Strings: essayer ISO d'abord si on voit des "-"
    s_str = s.astype(str).str.strip()
    iso_mask = s_str.str.contains(r"^\d{4}-\d{2}-\d{2}", regex=True, na=False)

    d_iso = pd.to_datetime(s_str.where(iso_mask), errors="coerce", dayfirst=False)
    d_txt = pd.to_datetime(s_str.where(~iso_mask), errors="coerce", dayfirst=True)

    # Excel serial
    d_num = pd.to_numeric(s, errors="coerce")
    d_xls = pd.to_datetime(d_num, unit="D", origin="1899-12-30", errors="coerce")

    out = d_iso.fillna(d_txt).fillna(d_xls)
    return out.dt.normalize()


def _safe_to_float(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _get_ann_ret(series: pd.Series) -> pd.Series:
    y_vals = series.resample("YE").last()
    ret = y_vals.pct_change(fill_method=None).dropna() * 100
    r_first = pd.Series([((y_vals.iloc[0] / series.iloc[0]) - 1) * 100], index=[y_vals.index[0]])
    r_last = pd.Series([((series.iloc[-1] / y_vals.iloc[-1]) - 1) * 100], index=[series.index[-1]])
    return pd.concat([r_first, ret, r_last])


# =========================================================
# 1) METRICS 1Y/3Y/5Y
# =========================================================
def compute_multi_horizon_metrics(pf_v: pd.Series, bm_v: pd.Series) -> pd.DataFrame:
    df = pd.concat([pf_v.rename("PF"), bm_v.rename("BM")], axis=1).dropna()
    ret = df.pct_change(fill_method=None).dropna()
    rp, rb = ret["PF"], ret["BM"]

    def metrics_over_years(years: int):
        w = int(252 * years)
        if len(df) < (w + 1):
            return [np.nan] * 6

        df_w = df.iloc[-(w + 1):]
        rp_w = rp.iloc[-w:]
        rb_w = rb.iloc[-w:]

        total_ret = df_w["PF"].iloc[-1] / df_w["PF"].iloc[0] - 1
        ann_ret = (1 + total_ret) ** (1 / years) - 1

        vol = rp_w.std() * np.sqrt(252)

        cov = np.cov(rp_w, rb_w, ddof=1)[0, 1]
        var_b = np.var(rb_w, ddof=1)
        beta = cov / var_b if var_b != 0 else np.nan
        corr = np.corrcoef(rp_w, rb_w)[0, 1]

        te = (rp_w - rb_w).std() * np.sqrt(252)

        ann_excess = (rp_w.mean() - rb_w.mean()) * 252
        ir = ann_excess / te if te != 0 else np.nan

        return [ann_ret, vol, beta, corr, te, ir]

    out = {k: metrics_over_years(v) for k, v in {"1 an": 1, "3 ans": 3, "5 ans": 5}.items()}

    dfm = pd.DataFrame(
        out,
        index=[
            "Rendement annualisé",
            "Volatilité annualisée",
            "Beta",
            "Corrélation",
            "Tracking Error",
            "Information Ratio",
        ],
    ).reset_index().rename(columns={"index": "Indicateur"})

    return dfm


# =========================================================
# 2) CORE ENGINE (commun)
# =========================================================
def _build_portfolio_from_prices(
    prices_long: pd.DataFrame,
    fx_long: pd.DataFrame,
    holdings: pd.DataFrame,
    *,
    start_target: str,
    end_target: str,
    benchmark_weights=(0.65, 0.35),
    indices_path: str,
    eagg_path: str,
    benchmark_code="NQGIN",
):
    start_target = pd.to_datetime(start_target).normalize()
    end_target = pd.to_datetime(end_target).normalize()

    # --- clean prices_long ---
    df = _standardize_columns(prices_long)
    if not {"date", "ticker", "close"}.issubset(df.columns):
        raise ValueError(
            "prices_long doit contenir au minimum: date, ticker, close.\n"
            f"Colonnes reçues: {list(df.columns)}"
        )

    df["date"] = _parse_mixed_dates(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip()

    if "currency" in df.columns:
        df["currency"] = df["currency"].astype(str).str.strip().str.upper()
    else:
        df["currency"] = "EUR"

    df["close"] = _safe_to_float(df["close"])
    df = df.dropna(subset=["date", "ticker", "close"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last").sort_values(["date", "ticker"])

    if df.empty:
        raise ValueError("prices_long vide après nettoyage (dates/close/ticker).")

    # DEBUG-friendly min/max
    min_d = df["date"].min()
    max_d = df["date"].max()

    prices_local = (
        df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
        .ffill()
    )

    # --- FX wide ---
    fx = _standardize_columns(fx_long)
    if not {"date", "ticker", "close"}.issubset(fx.columns):
        raise ValueError("fx_long doit contenir: date, ticker, close.")

    fx["date"] = _parse_mixed_dates(fx["date"])
    fx["ticker"] = fx["ticker"].astype(str).str.strip().str.upper()
    fx["close"] = _safe_to_float(fx["close"])
    fx = fx.dropna(subset=["date", "ticker", "close"])

    fx = fx[fx["ticker"].str.startswith("EUR")].copy()
    fx["ccy"] = fx["ticker"].str[3:6]  # EURUSD -> USD

    fx_wide = (
        fx.pivot_table(index="date", columns="ccy", values="close", aggfunc="last")
        .sort_index()
        .ffill()
    )
    fx_wide["EUR"] = 1.0

    # --- holdings dispo ---
    available = [t for t in holdings.index if t in prices_local.columns]
    missing_isins = sorted(set(holdings.index) - set(available))
    if len(available) == 0:
        raise ValueError(
            "Aucun ticker holdings n'est présent dans prices_local.\n"
            f"holdings tickers (ex): {list(holdings.index)[:10]}\n"
            f"prices_local cols (ex): {list(prices_local.columns)[:10]}"
        )
    holdings = holdings.loc[available].copy()

    # --- currency map ---
    ticker_cur = (
        df.dropna(subset=["ticker", "currency"])
        .drop_duplicates("ticker")
        .set_index("ticker")["currency"]
        .to_dict()
    )

    # --- conversion EUR ---
    prices_eur = pd.DataFrame(index=prices_local.index, columns=available, dtype=float)
    for t in available:
        ccy = str(ticker_cur.get(t, "EUR")).upper()
        if ccy not in fx_wide.columns:
            ccy = "EUR"
        fx_series = fx_wide[ccy].reindex(prices_local.index).ffill().bfill()
        prices_eur[t] = prices_local[t] / fx_series

    # --- start robuste + message utile ---
    dates_ok = prices_eur.index[prices_eur.index >= start_target]
    if len(dates_ok) == 0:
        raise ValueError(
            "Aucune date >= start_target dans les prix.\n"
            f"start_target={start_target.date()} | min_date={min_d.date()} | max_date={max_d.date()}"
        )

    prices_start_window = prices_eur.loc[dates_ok]
    valid_start_mask = prices_start_window[available].notna().all(axis=1)

    if valid_start_mask.any():
        actual_start = prices_start_window.index[int(np.argmax(valid_start_mask.values))]
    else:
        actual_start = prices_start_window.index[0]
        ok_cols = prices_start_window.loc[actual_start].dropna().index.tolist()
        dropped = sorted(set(available) - set(ok_cols))
        available = ok_cols
        holdings = holdings.loc[available].copy()
        prices_eur = prices_eur[available].copy()
        missing_isins = sorted(set(missing_isins) | set(dropped))

    # --- units + valeur portefeuille ---
    p0 = prices_eur.loc[actual_start, available]
    units = holdings["val"] / p0

    prices_filtered = prices_eur[available].loc[actual_start:end_target].copy()
    portfolio_value = prices_filtered.mul(units, axis=1).sum(axis=1)
    portfolio_value = portfolio_value[~portfolio_value.index.duplicated(keep="last")]

    # --- benchmark ---
    w_eq, w_bond = benchmark_weights

    df_eagg = pd.read_csv(eagg_path)
    # tolérance colonne Date/Dernier
    if "Date" not in df_eagg.columns and "date" in df_eagg.columns:
        df_eagg = df_eagg.rename(columns={"date": "Date"})
    if "Dernier" not in df_eagg.columns and "dernier" in df_eagg.columns:
        df_eagg = df_eagg.rename(columns={"dernier": "Dernier"})

    df_eagg["date"] = pd.to_datetime(df_eagg["Date"], dayfirst=True, errors="coerce").dt.normalize()
    s_eagg = df_eagg.set_index("date")["Dernier"].astype(str).str.replace(",", ".", regex=False)
    s_eagg = pd.to_numeric(s_eagg, errors="coerce").dropna().sort_index()
    s_eagg = s_eagg[~s_eagg.index.duplicated(keep="last")]

    df_idx = pd.read_parquet(indices_path).copy()
    df_idx.columns = [str(c).strip().lower() for c in df_idx.columns]
    if not {"date", "code", "price"}.issubset(df_idx.columns):
        raise ValueError(f"indices parquet doit contenir date/code/price. Colonnes: {list(df_idx.columns)}")

    df_idx["date"] = pd.to_datetime(df_idx["date"], errors="coerce").dt.normalize()
    nq = df_idx[df_idx["code"] == benchmark_code].set_index("date")["price"].sort_index()
    nq = nq[~nq.index.duplicated(keep="last")]

    if "USD" not in fx_wide.columns:
        raise ValueError("FX: colonne USD absente (besoin EURUSD...) dans forex.parquet")
    nq_eur = nq / fx_wide["USD"].reindex(nq.index).ffill().bfill()

    st_nq = nq_eur.index[nq_eur.index >= actual_start]
    st_ea = s_eagg.index[s_eagg.index >= actual_start]
    if len(st_nq) == 0 or len(st_ea) == 0:
        raise ValueError("Benchmark: pas assez de données après actual_start (NQGIN/EAGG).")

    bench_perf = (w_eq * (nq_eur / nq_eur.loc[st_nq[0]])) + (w_bond * (s_eagg / s_eagg.loc[st_ea[0]]))
    bench_perf = bench_perf[~bench_perf.index.duplicated(keep="last")]

    aligned = pd.concat([portfolio_value.rename("PF"), bench_perf.rename("BM_PERF")], axis=1).dropna()
    portfolio_value = aligned["PF"]
    benchmark_value = aligned["BM_PERF"] * portfolio_value.iloc[0]

    # --- ret / mu / vol ---
    ret_pf = portfolio_value.pct_change(fill_method=None).dropna()
    ret_bm = benchmark_value.pct_change(fill_method=None).dropna()

    vol_p = float(ret_pf.std() * np.sqrt(252))
    dur = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    mu_p = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / dur) - 1 if dur > 0 else np.nan

    # --- contributions ---
    w_final = prices_filtered.iloc[-1].mul(units) / portfolio_value.iloc[-1]
    asset_ret = prices_filtered.pct_change(fill_method=None).dropna()

    perf_c = (prices_filtered.iloc[-1].mul(units) - prices_filtered.iloc[0].mul(units)) / portfolio_value.iloc[0] * 100

    if vol_p != 0 and len(ret_pf) > 2:
        cov_to_pf = asset_ret.apply(lambda x: x.cov(ret_pf)) * 252
        risk_c = (w_final * cov_to_pf) / vol_p * 100
    else:
        risk_c = w_final * 0.0

    exposure_eur = (prices_filtered.iloc[-1] * units).rename("Exposition (€)")

    # outputs communs
    ann_p = _get_ann_ret(portfolio_value)
    ann_b = _get_ann_ret(benchmark_value)
    r_p = (ret_pf.rolling(252).std() * np.sqrt(252) * 100).dropna()
    r_b = (ret_bm.rolling(252).std() * np.sqrt(252) * 100).dropna()

    return {
        "pf_v": portfolio_value,
        "bm_v": benchmark_value,
        "ann_p": ann_p,
        "ann_b": ann_b,
        "r_p": r_p,
        "r_b": r_b,
        "mu_p": mu_p,
        "vol_p": vol_p,
        "w_final": w_final,
        "perf_c": perf_c,
        "risk_c": risk_c,
        "exposure_eur": exposure_eur,
        "asset_ret": asset_ret,
        "missing_isins": missing_isins,
    }


# =========================================================
# 3) PIPELINE CLIENT
# =========================================================
def get_clean_data(
    DATA_PATH=r"C:\Users\Fayca\Documents\VSCode\Master272-RisquePerformance-2026\Projet\Data.xlsb",
    FX_PATH=r"C:\Users\Fayca\Documents\VSCode\Master272-RisquePerformance-2026\Projet\forex.parquet",
    INDICES_PATH=r"C:\Users\Fayca\Documents\VSCode\Master272-RisquePerformance-2026\Projet\Indices\indice_prices.parquet",
    EAGG_PATH=r"C:\Users\Fayca\Documents\VSCode\Master272-RisquePerformance-2026\Projet\Données historiques EAGG(1).csv",
    start_target="2020-01-02",
    end_target="2026-01-30",
):
    df_raw = pd.read_excel(DATA_PATH, sheet_name="Prices", engine="pyxlsb")
    df_raw = _standardize_columns(df_raw)

    if not {"date", "ticker", "close"}.issubset(df_raw.columns):
        raise ValueError(
            "Sheet 'Prices' doit contenir au minimum date/ticker/close.\n"
            f"Colonnes reçues: {list(df_raw.columns)}"
        )

    df_raw["date"] = _parse_mixed_dates(df_raw["date"])
    df_raw["ticker"] = df_raw["ticker"].astype(str).str.strip()
    df_raw["close"] = _safe_to_float(df_raw["close"])

    if "currency" not in df_raw.columns:
        df_raw["currency"] = "EUR"
    else:
        df_raw["currency"] = df_raw["currency"].astype(str).str.strip().str.upper()

    # holdings client FIXES (valeurs €)
    holdings = pd.DataFrame(
        {
            "ticker": [
                "LU0072462186","LU1893597309","LU0154236417","LU1883854199",
                "FR0010983924","LU1244893696","LU1919842267","LU1279613365",
                "FR0010868901","LU1161527038","LU1191877379","FR0011288513",
                "LU1882449801","LU1897613763",
            ],
            "val": [
                224934.50,224915.04,149951.66,150013.50,74926.60,74883.69,44909.55,29973.96,
                45195.35,179968.80,74982.11,90055.06,29991.99,105079.80,
            ],
        }
    ).set_index("ticker")

    df_fx = pd.read_parquet(FX_PATH).copy()

    core = _build_portfolio_from_prices(
        prices_long=df_raw[["date", "ticker", "close", "currency"]].copy(),
        fx_long=df_fx[["date", "ticker", "close"]].copy(),
        holdings=holdings,
        start_target=start_target,
        end_target=end_target,
        benchmark_weights=(0.65, 0.35),
        indices_path=INDICES_PATH,
        eagg_path=EAGG_PATH,
        benchmark_code="NQGIN",
    )

    # mapping / labels pour w_data (client)
    mapping_fonds = {
        "LU0072462186": {"Nom": "🇪🇺 BGF European Value A2", "Géo": "Europe", "Sec": "Finance"},
        "LU1893597309": {"Nom": "🇪🇺 BSF European Unconstrained", "Géo": "Europe", "Sec": "Actions"},
        "LU0154236417": {"Nom": "🇺🇸 BGF US Flexible Equity", "Géo": "USA", "Sec": "Actions"},
        "LU1883854199": {"Nom": "🇺🇸 Amundi US Eq Fundm Gr", "Géo": "USA", "Sec": "Tech"},
        "FR0010983924": {"Nom": "🇯🇵 EdR Japan C", "Géo": "Japon", "Sec": "Actions"},
        "LU1244893696": {"Nom": "🌐 EdRF Big Data A EUR", "Géo": "Monde", "Sec": "Tech"},
        "LU1919842267": {"Nom": "🌐 Oddo Artificial Intellig", "Géo": "Monde", "Sec": "Tech"},
        "LU1279613365": {"Nom": "🌏 BGF Asian Dragon A2", "Géo": "Asie", "Sec": "Actions"},
        "FR0010868901": {"Nom": "🇪🇺 Ellipsis European Conv", "Géo": "Europe", "Sec": "Conv."},
        "LU1161527038": {"Nom": "🇪🇺 EdRF Bond Allocation Acc", "Géo": "Europe", "Sec": "Oblig."},
        "LU1191877379": {"Nom": "🇪🇺 BGF European High Yield", "Géo": "Europe", "Sec": "Oblig."},
        "FR0011288513": {"Nom": "🇪🇺 Sycomore Sélection Crédit", "Géo": "Europe", "Sec": "Oblig."},
        "LU1882449801": {"Nom": "🚩 Amundi Em Mkts Bd", "Géo": "Emergents", "Sec": "Emergents"},
        "LU1897613763": {"Nom": "🚩 EdRF Emerging Sovereign", "Géo": "Emergents", "Sec": "Emergents"},
    }
    type_map = {
        "Actions": "Actions", "Tech": "Actions", "Finance": "Actions",
        "Conv.": "Hybride",
        "Oblig.": "Obligations", "Emergents": "Obligations",
    }

    tickers = holdings.index.astype(str).tolist()

    w_data = pd.DataFrame({
        "ISIN": tickers,  # ✅ IMPORTANT pour les match ex-ante
        "Nom du Fonds": [mapping_fonds.get(i, {}).get("Nom", i) for i in tickers],
        "Type d'actif": [type_map.get(mapping_fonds.get(i, {}).get("Sec", ""), "Autre") for i in tickers],
        "Exposition (€)": [float(core["exposure_eur"].reindex(tickers).loc[i]) for i in tickers],
        "Poids (%)": (core["w_final"].reindex(tickers) * 100).round(2).values,
        "Contr. Perf (%)": pd.Series(core["perf_c"], index=core["w_final"].index).reindex(tickers).values.round(2),
        "Contr. Risque (%)": pd.Series(core["risk_c"], index=core["w_final"].index).reindex(tickers).values.round(2),
    }).sort_values("Poids (%)", ascending=False).reset_index(drop=True)

    geo_df = (
        pd.DataFrame({"Géo": [mapping_fonds[i]["Géo"] for i in tickers],
                      "P": core["w_final"].reindex(tickers).values})
        .groupby("Géo").sum().reset_index()
    )
    sec_df = (
        pd.DataFrame({"Sec": [mapping_fonds[i]["Sec"] for i in tickers],
                      "P": core["w_final"].reindex(tickers).values})
        .groupby("Sec").sum().reset_index()
    )

    return (
        core["pf_v"], core["bm_v"],
        core["ann_p"], core["ann_b"],
        core["r_p"], core["r_b"],
        w_data, geo_df, sec_df,
        core["mu_p"], core["vol_p"],
    )