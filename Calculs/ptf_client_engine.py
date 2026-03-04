from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# Helpers (clean + factorisés)

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower/strip + rename common aliases -> date/ticker/close."""
    df = df.copy()
    df.columns = pd.Index(df.columns).astype(str).str.strip().str.lower()

    renames = {}
    if "date" not in df.columns:
        for alt in ("dates", "datetime", "jour", "day"):
            if alt in df.columns:
                renames[alt] = "date"
                break

    if "ticker" not in df.columns:
        for alt in ("isin", "code", "id", "instrument"):
            if alt in df.columns:
                renames[alt] = "ticker"
                break

    if "close" not in df.columns:
        for alt in ("price", "last", "dernier", "vl", "nav"):
            if alt in df.columns:
                renames[alt] = "close"
                break

    return df.rename(columns=renames)


def _parse_mixed_dates(s: pd.Series) -> pd.Series:
    """ISO / dd-mm / excel-serial -> datetime normalized."""
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="coerce").dt.normalize()

    s_str = s.astype(str).str.strip()
    iso_mask = s_str.str.match(r"^\d{4}-\d{2}-\d{2}")

    d_iso = pd.to_datetime(s_str.where(iso_mask), errors="coerce", dayfirst=False)
    d_txt = pd.to_datetime(s_str.where(~iso_mask), errors="coerce", dayfirst=True)

    d_num = pd.to_numeric(s, errors="coerce")
    d_xls = pd.to_datetime(d_num, unit="D", origin="1899-12-30", errors="coerce")

    return d_iso.fillna(d_txt).fillna(d_xls).dt.normalize()


def _to_float(s: pd.Series) -> pd.Series:
    """Robuste pour '1,23' -> 1.23."""
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _annualize_vol(ret: pd.Series) -> float:
    ret = ret.dropna().astype(float)
    return float(ret.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(ret) else np.nan


def _annual_returns_calendar(v: pd.Series) -> pd.Series:
    """Rendements calendaires (%) sur années civiles, basé sur YE."""
    v = v.dropna().astype(float)
    if v.empty:
        return v

    y_end = v.resample("YE").last()
    y_start = v.resample("YE").first()
    out = (y_end / y_start - 1.0) * 100.0
    out.index = out.index.to_period("Y").to_timestamp()
    return out


def _rolling_vol_pct(ret: pd.Series, win: int = TRADING_DAYS) -> pd.Series:
    ret = ret.dropna().astype(float)
    if ret.empty:
        return ret
    return ret.rolling(win).std(ddof=0) * np.sqrt(TRADING_DAYS) * 100.0


def _pivot_prices_wide(df_long: pd.DataFrame, *, date_col="date", ticker_col="ticker", value_col="close") -> pd.DataFrame:
    """Pivot wide + sort + ffill."""
    wide = df_long.pivot_table(index=date_col, columns=ticker_col, values=value_col, aggfunc="last")
    return wide.sort_index().ffill()


def _first_valid_start(prices_wide: pd.DataFrame, start_target: pd.Timestamp, cols: list[str]) -> pd.Timestamp:
    """Première date >= start_target où toutes les colonnes cols ont une valeur."""
    dates_ok = prices_wide.index[prices_wide.index >= start_target]
    if len(dates_ok) == 0:
        raise ValueError(
            f"Aucune date >= start_target. start_target={start_target.date()} | "
            f"min_date={prices_wide.index.min().date()} | max_date={prices_wide.index.max().date()}"
        )

    window = prices_wide.loc[dates_ok, cols]
    mask = window.notna().all(axis=1)
    return window.index[int(np.argmax(mask.values))] if mask.any() else window.index[0]


def compute_multi_horizon_metrics(pf_v: pd.Series, bm_v: pd.Series) -> pd.DataFrame:
    """Fenêtres glissantes (1y/3y/5y) sur PF/BM alignés."""
    df = pd.concat([pf_v.rename("PF"), bm_v.rename("BM")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame({"Indicateur": [], "1 an": [], "3 ans": [], "5 ans": []})

    ret = df.pct_change().dropna()
    rp, rb = ret["PF"], ret["BM"]

    def _metrics(years: int) -> list[float]:
        w = int(TRADING_DAYS * years)
        if len(df) < (w + 1) or len(ret) < w:
            return [np.nan] * 6

        df_w = df.iloc[-(w + 1) :]
        rp_w = rp.iloc[-w:]
        rb_w = rb.iloc[-w:]

        total_ret = df_w["PF"].iloc[-1] / df_w["PF"].iloc[0] - 1.0
        ann_ret = (1.0 + total_ret) ** (1.0 / years) - 1.0

        vol = float(rp_w.std(ddof=0) * np.sqrt(TRADING_DAYS))

        cov = float(np.cov(rp_w.values, rb_w.values, ddof=0)[0, 1])
        var_b = float(np.var(rb_w.values, ddof=0))
        beta = cov / var_b if var_b > 0 else np.nan

        corr = float(np.corrcoef(rp_w.values, rb_w.values)[0, 1]) if rp_w.std(ddof=0) > 0 and rb_w.std(ddof=0) > 0 else np.nan
        te = float((rp_w - rb_w).std(ddof=0) * np.sqrt(TRADING_DAYS))

        ann_excess = float((rp_w.mean() - rb_w.mean()) * TRADING_DAYS)
        ir = ann_excess / te if te > 0 else np.nan

        return [ann_ret, vol, beta, corr, te, ir]

    out = {lbl: _metrics(y) for lbl, y in {"1 an": 1, "3 ans": 3, "5 ans": 5}.items()}
    return (
        pd.DataFrame(
            out,
            index=[
                "Rendement annualisé",
                "Volatilité annualisée",
                "Beta",
                "Corrélation",
                "Tracking Error",
                "Information Ratio",
            ],
        )
        .reset_index()
        .rename(columns={"index": "Indicateur"})
    )



# Core pricer (build portfolio + benchmark)

def _build_portfolio_from_prices(
    prices_long: pd.DataFrame,
    fx_long: pd.DataFrame,
    holdings: pd.DataFrame,  # index=ticker, col 'val'
    *,
    start_target: str,
    end_target: str,
    benchmark_weights=(0.65, 0.35),
    indices_path: Path,
    eagg_path: Path,
    benchmark_code="NQGIN",
    parquet_engine: str = "fastparquet",
) -> dict:
    start_target = pd.to_datetime(start_target).normalize()
    end_target = pd.to_datetime(end_target).normalize()

    # ---------- Prices long -> wide local
    df = _standardize_columns(prices_long)
    needed = {"date", "ticker", "close"}
    if not needed.issubset(df.columns):
        raise ValueError(f"prices_long doit contenir {needed}. Colonnes: {list(df.columns)}")

    df["date"] = _parse_mixed_dates(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["currency"] = (df["currency"].astype(str).str.strip().str.upper() if "currency" in df.columns else "EUR")
    df["close"] = _to_float(df["close"])
    df = df.dropna(subset=["date", "ticker", "close"])
    if df.empty:
        raise ValueError("prices_long vide après nettoyage.")

    prices_local = _pivot_prices_wide(df, date_col="date", ticker_col="ticker", value_col="close")

    # FX long -> wide (EURxxx)
    fx = _standardize_columns(fx_long)
    if not needed.issubset(fx.columns):
        raise ValueError("fx_long doit contenir date/ticker/close.")

    fx["date"] = _parse_mixed_dates(fx["date"])
    fx["ticker"] = fx["ticker"].astype(str).str.strip().str.upper()
    fx["close"] = _to_float(fx["close"])
    fx = fx.dropna(subset=["date", "ticker", "close"])
    fx = fx[fx["ticker"].str.startswith("EUR")].copy()
    fx["ccy"] = fx["ticker"].str[3:6]

    fx_wide = _pivot_prices_wide(fx, date_col="date", ticker_col="ccy", value_col="close")
    fx_wide["EUR"] = 1.0

    
    holdings = holdings.copy()
    holdings.index = holdings.index.astype(str).str.strip()
    available = [t for t in holdings.index if t in prices_local.columns]
    missing_isins = sorted(set(holdings.index) - set(available))
    if not available:
        raise ValueError("Aucun ticker holdings n'est présent dans prices_local.")
    holdings = holdings.loc[available]

    # ticker -> ccy mapping (from prices_long cleaned)
    ticker_cur = (
        df[["ticker", "currency"]]
        .dropna()
        .drop_duplicates("ticker")
        .set_index("ticker")["currency"]
        .to_dict()
    )

    # Convert to EUR (vectorisé via division colonne par série FX)
    prices_eur = pd.DataFrame(index=prices_local.index, columns=available, dtype=float)
    for t in available:
        ccy = str(ticker_cur.get(t, "EUR")).upper()
        if ccy not in fx_wide.columns:
            ccy = "EUR"
        fx_series = fx_wide[ccy].reindex(prices_local.index).ffill().bfill()
        prices_eur[t] = prices_local[t] / fx_series

    # Start date (first date >= start_target with full data)
    actual_start = _first_valid_start(prices_eur, start_target, available)

    # if some columns still NaN at actual_start, drop them (fallback)
    ok_cols = prices_eur.loc[actual_start, available].dropna().index.tolist()
    dropped = sorted(set(available) - set(ok_cols))
    if dropped:
        available = ok_cols
        holdings = holdings.loc[available]
        prices_eur = prices_eur[available]
        missing_isins = sorted(set(missing_isins) | set(dropped))

    # Units + portfolio value
    p0 = prices_eur.loc[actual_start, available]
    units = holdings["val"].astype(float) / p0

    prices_filtered = prices_eur.loc[actual_start:end_target, available].copy()
    portfolio_value = prices_filtered.mul(units, axis=1).sum(axis=1)
    portfolio_value = portfolio_value[~portfolio_value.index.duplicated(keep="last")]

    # Benchmark components (NQGIN + EAGG)
    w_eq, w_bond = map(float, benchmark_weights)

    # EAGG CSV
    df_eagg = pd.read_csv(eagg_path)
    df_eagg.columns = pd.Index(df_eagg.columns).astype(str)
    if "Date" not in df_eagg.columns and "date" in df_eagg.columns:
        df_eagg = df_eagg.rename(columns={"date": "Date"})
    if "Dernier" not in df_eagg.columns and "dernier" in df_eagg.columns:
        df_eagg = df_eagg.rename(columns={"dernier": "Dernier"})

    df_eagg["date"] = pd.to_datetime(df_eagg["Date"], dayfirst=True, errors="coerce").dt.normalize()
    s_eagg = _to_float(df_eagg["Dernier"]).rename("Dernier")
    s_eagg = pd.Series(s_eagg.values, index=df_eagg["date"]).dropna().sort_index()
    s_eagg = s_eagg[~s_eagg.index.duplicated(keep="last")]

    # Indices parquet
    df_idx = pd.read_parquet(indices_path, engine=parquet_engine).copy()
    df_idx.columns = pd.Index(df_idx.columns).astype(str).str.strip().str.lower()
    if not {"date", "code", "price"}.issubset(df_idx.columns):
        raise ValueError(f"indices parquet doit contenir date/code/price. Colonnes: {list(df_idx.columns)}")
    df_idx["date"] = pd.to_datetime(df_idx["date"], errors="coerce").dt.normalize()

    nq = df_idx.loc[df_idx["code"] == benchmark_code].set_index("date")["price"].sort_index()
    nq = nq[~nq.index.duplicated(keep="last")]

    if "USD" not in fx_wide.columns:
        raise ValueError("FX: colonne USD absente (besoin EURUSD...) dans forex.parquet")

    nq_eur = nq / fx_wide["USD"].reindex(nq.index).ffill().bfill()

    # align start (need data after actual_start)
    st_nq = nq_eur.index[nq_eur.index >= actual_start]
    st_ea = s_eagg.index[s_eagg.index >= actual_start]
    if len(st_nq) == 0 or len(st_ea) == 0:
        raise ValueError("Benchmark: pas assez de données après actual_start (NQGIN/EAGG).")

    bench_perf = (w_eq * (nq_eur / nq_eur.loc[st_nq[0]])) + (w_bond * (s_eagg / s_eagg.loc[st_ea[0]]))
    bench_perf = bench_perf[~bench_perf.index.duplicated(keep="last")]

    # ---------- Align PF/BM
    aligned = pd.concat([portfolio_value.rename("PF"), bench_perf.rename("BM_PERF")], axis=1).dropna()
    portfolio_value = aligned["PF"]
    benchmark_value = aligned["BM_PERF"] * portfolio_value.iloc[0]

    # ---------- Returns + stats
    ret_pf = portfolio_value.pct_change().dropna()
    ret_bm = benchmark_value.pct_change().dropna()

    vol_p = _annualize_vol(ret_pf)

    dur_years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    mu_p = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / dur_years) - 1 if dur_years > 0 else np.nan

    # weights final
    w_final = prices_filtered.loc[portfolio_value.index[-1]].mul(units) / float(portfolio_value.iloc[-1])

    # asset returns
    asset_ret = prices_filtered.pct_change().dropna()

    # perf contrib (% sur valeur initiale)
    perf_c = (prices_filtered.iloc[-1].mul(units) - prices_filtered.iloc[0].mul(units)) / float(portfolio_value.iloc[0]) * 100.0

    # risk contrib (%)
 # risk contrib (%)
    if np.isfinite(vol_p) and vol_p > 0 and len(ret_pf) > 2:
        # cov(asset, portefeuille) annualisée
        cov_to_pf = asset_ret.apply(lambda s: s.cov(ret_pf)) * TRADING_DAYS
        risk_c = (w_final * cov_to_pf) / vol_p * 100
    else:
        risk_c = w_final * 0.0

    exposure_eur = (prices_filtered.iloc[-1] * units).rename("Exposition (€)")
    # calendaires + rolling vol
    ann_p = _annual_returns_calendar(portfolio_value)
    ann_b = _annual_returns_calendar(benchmark_value)
    r_p = _rolling_vol_pct(ret_pf, win=TRADING_DAYS)
    r_b = _rolling_vol_pct(ret_bm, win=TRADING_DAYS)

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



def get_clean_data(
    DATA_PATH: str | Path | None = None,
    FX_PATH: str | Path | None = None,
    INDICES_PATH: str | Path | None = None,
    EAGG_PATH: str | Path | None = None,
    start_target: str = "2020-01-02",
    end_target: str = "2026-01-30",
    parquet_engine: str = "fastparquet",
):
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "Projet"

    data_path = Path(DATA_PATH) if DATA_PATH is not None else (data_dir / "Data.xlsb")
    fx_path = Path(FX_PATH) if FX_PATH is not None else (data_dir / "forex.parquet")
    indices_path = Path(INDICES_PATH) if INDICES_PATH is not None else (data_dir / "Indices" / "indice_prices.parquet")
    eagg_path = Path(EAGG_PATH) if EAGG_PATH is not None else (data_dir / "Données historiques EAGG(1).csv")

    df_raw = pd.read_excel(data_path, sheet_name="Prices", engine="pyxlsb")
    df_raw = _standardize_columns(df_raw)
    if not {"date", "ticker", "close"}.issubset(df_raw.columns):
        raise ValueError(f"Sheet 'Prices' doit contenir date/ticker/close. Colonnes: {list(df_raw.columns)}")

    df_raw["date"] = _parse_mixed_dates(df_raw["date"])
    df_raw["ticker"] = df_raw["ticker"].astype(str).str.strip()
    df_raw["close"] = _to_float(df_raw["close"])
    df_raw["currency"] = (df_raw["currency"].astype(str).str.strip().str.upper() if "currency" in df_raw.columns else "EUR")

    holdings = (
        pd.DataFrame(
            {
                "ticker": [
                    "LU0072462186", "LU1893597309", "LU0154236417", "LU1883854199",
                    "FR0010983924", "LU1244893696", "LU1919842267", "LU1279613365",
                    "FR0010868901", "LU1161527038", "LU1191877379", "FR0011288513",
                    "LU1882449801", "LU1897613763",
                ],
                "val": [
                    224934.50, 224915.04, 149951.66, 150013.50,
                    74926.60, 74883.69, 44909.55, 29973.96,
                    45195.35, 179968.80, 74982.11, 90055.06,
                    29991.99, 105079.80,
                ],
            }
        )
        .set_index("ticker")
    )

    df_fx = pd.read_parquet(fx_path, engine=parquet_engine).copy()

    core = _build_portfolio_from_prices(
        prices_long=df_raw[["date", "ticker", "close", "currency"]].copy(),
        fx_long=df_fx[["date", "ticker", "close"]].copy(),
        holdings=holdings,
        start_target=start_target,
        end_target=end_target,
        benchmark_weights=(0.65, 0.35),
        indices_path=indices_path,
        eagg_path=eagg_path,
        benchmark_code="NQGIN",
        parquet_engine=parquet_engine,
    )

    # mapping -> w_data + agrégations
    mapping_fonds = {
        "LU0072462186": {"Nom": "BGF European Value A2", "Géo": "Europe", "Sec": "Finance"},
        "LU1893597309": {"Nom": "BSF European Unconstrained", "Géo": "Europe", "Sec": "Actions"},
        "LU0154236417": {"Nom": "BGF US Flexible Equity", "Géo": "USA", "Sec": "Actions"},
        "LU1883854199": {"Nom": "Amundi US Eq Fundm Gr", "Géo": "USA", "Sec": "Tech"},
        "FR0010983924": {"Nom": "EdR Japan C", "Géo": "Japon", "Sec": "Actions"},
        "LU1244893696": {"Nom": "EdRF Big Data A EUR", "Géo": "Monde", "Sec": "Tech"},
        "LU1919842267": {"Nom": "Oddo Artificial Intellig", "Géo": "Monde", "Sec": "Tech"},
        "LU1279613365": {"Nom": "BGF Asian Dragon A2", "Géo": "Asie", "Sec": "Actions"},
        "FR0010868901": {"Nom": "Ellipsis European Conv", "Géo": "Europe", "Sec": "Conv."},
        "LU1161527038": {"Nom": "EdRF Bond Allocation Acc", "Géo": "Europe", "Sec": "Oblig."},
        "LU1191877379": {"Nom": "BGF European High Yield", "Géo": "Europe", "Sec": "Oblig."},
        "FR0011288513": {"Nom": "Sycomore Sélection Crédit", "Géo": "Europe", "Sec": "Oblig."},
        "LU1882449801": {"Nom": "Amundi Em Mkts Bd", "Géo": "Emergents", "Sec": "Emergents"},
        "LU1897613763": {"Nom": "EdRF Emerging Sovereign", "Géo": "Emergents", "Sec": "Emergents"},
    }
    type_map = {
        "Actions": "Actions",
        "Tech": "Actions",
        "Finance": "Actions",
        "Conv.": "Hybride",
        "Oblig.": "Obligations",
        "Emergents": "Obligations",
    }

    tickers = holdings.index.astype(str).tolist()

    mf = pd.DataFrame.from_dict(mapping_fonds, orient="index")
    mf.index.name = "ISIN"
    mf = mf.reset_index()

    # base table
    base = pd.DataFrame({"ISIN": tickers}).merge(mf, on="ISIN", how="left")
    base["Nom"] = base["Nom"].fillna(base["ISIN"])
    base["Type d'actif"] = base["Sec"].map(type_map).fillna("Autre")

    w_final = core["w_final"].reindex(tickers)
    exposure = core["exposure_eur"].reindex(tickers)
    perf_c = pd.Series(core["perf_c"]).reindex(tickers)
    risk_c = pd.Series(core["risk_c"]).reindex(tickers)

    w_data = (
        base.assign(
            **{
                "Nom du Fonds": base["Nom"],
                "Exposition (€)": exposure.astype(float).values,
                "Poids (%)": (w_final.astype(float) * 100.0).values,
                "Contr. Perf (%)": perf_c.astype(float).values,
                "Contr. Risque (%)": risk_c.astype(float).values,
            }
        )
        .drop(columns=["Nom"])
        .sort_values("Poids (%)", ascending=False)
        .reset_index(drop=True)
    )

    geo_df = (
        base.assign(P=w_final.values)
        .dropna(subset=["Géo"])
        .groupby("Géo", as_index=False)["P"]
        .sum()
    )

    sec_df = (
        base.assign(P=w_final.values)
        .dropna(subset=["Sec"])
        .groupby("Sec", as_index=False)["P"]
        .sum()
    )

    return (
        core["pf_v"],
        core["bm_v"],
        core["ann_p"],
        core["ann_b"],
        core["r_p"],
        core["r_b"],
        w_data,
        geo_df,
        sec_df,
        core["mu_p"],
        core["vol_p"],
    )