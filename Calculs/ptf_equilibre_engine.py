# Calculs/ptf_equilibre_engine.py
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from Calculs.ptf_client_engine import _build_portfolio_from_prices


EQUILIBRE_MACRO = {"Monétaire": 0.00, "Obligataire": 0.50, "Actions": 0.50}

BONDS_FIXED_W = 0.50
ACTIONS_FIXED_W = 0.50

DEF_MIN_W = 0.15
DEF_MAX_W = 0.20
DEF_BASE = 0.10
DEF_K = 0.06

CORE_MIN_PER_FUND = 0.02
DEFAULT_MAX_LINE = 0.10

DEFAULT_CORE_TILT = 0.30
DEFAULT_DEF_TILT = 0.60

TRADING_DAYS = 252


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
        "Amundi S&P 500 Screened Index AE Acc",
        "EdR Fd Big Data A EUR",
        ("BSF European Unconstrained", "LU1893597309"),
        ("EdR Japan C", "FR0010983924"),
        ("BGF Asian Dragon A2", "LU1279613365"),
    ],
}


def _normalize_universe(uni: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    uni = uni.copy()
    uni.columns = [str(c).strip() for c in uni.columns]

    col_map = {}
    for c in uni.columns:
        cl = str(c).strip().lower()
        if cl == "nom":
            col_map[c] = "Nom"
        elif "isin" in cl:
            col_map[c] = "ISIN"

    uni = uni.rename(columns=col_map).copy()
    if not {"Nom", "ISIN"}.issubset(set(uni.columns)):
        raise ValueError("Universe doit contenir les colonnes 'Nom' et 'ISIN'.")

    uni["Nom"] = uni["Nom"].astype(str).str.strip()
    uni["ISIN"] = uni["ISIN"].astype(str).str.strip()
    uni = uni[uni["Nom"].ne("") & uni["ISIN"].ne("")].drop_duplicates(subset=["ISIN"]).copy()
    uni["Nom_norm"] = uni["Nom"].str.lower().str.strip()
    return uni, uni.set_index("ISIN")


def _resolve_name_to_isin(uni: pd.DataFrame, name: str) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None

    q = name.strip().lower()
    exact = uni.loc[uni["Nom_norm"].eq(q), "ISIN"]
    if len(exact) == 1:
        return str(exact.iloc[0])

    contains = uni.loc[uni["Nom_norm"].str.contains(re.escape(q), na=False), "ISIN"]
    if len(contains) == 1:
        return str(contains.iloc[0])

    return None


def build_equilibre_portfolio(uni: pd.DataFrame) -> pd.DataFrame:
    """
    Liste les fonds (ISIN + bucket + nom). Poids=1.0 (placeholder).
    Les vrais poids init doivent venir du CSV.
    """
    uni, uni_isin = _normalize_universe(uni)

    rows: list[dict] = []
    used: set[str] = set()

    for bucket, names in EQUILIBRE_MANUAL_NAMES.items():
        for nm in names:
            if isinstance(nm, tuple):
                nom_fonds, isin = nm
            else:
                nom_fonds, isin = nm, _resolve_name_to_isin(uni, nm)

            if not isin:
                rows.append({"Bucket": bucket, "Nom": f"Introuvable: {nom_fonds}", "ISIN": "", "Poids": 0.0})
                continue

            if isin in used:
                continue
            used.add(isin)

            nom_final = uni_isin.loc[isin, "Nom"] if isin in uni_isin.index else nom_fonds
            rows.append({"Bucket": bucket, "Nom": nom_final, "ISIN": str(isin), "Poids": 1.0})

    df = pd.DataFrame(rows)
    cols = ["Bucket", "Nom", "ISIN", "Poids"]
    return df[cols] if not df.empty else pd.DataFrame(columns=cols)


def _load_initial_weights_csv(
    csv_path: str | Path,
    tickers: list[str],
    *,
    weight_col_candidates=("Poids", "Poids (%)", "Weight", "weight"),
    isin_col_candidates=("ISIN", "Isin", "isin"),
) -> pd.Series:
    """Lit le CSV et renvoie weights (index=ISIN) normalisés (somme=1) filtrés sur tickers."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    isin_col = next((c for c in isin_col_candidates if c in cols), None)
    w_col = next((c for c in weight_col_candidates if c in cols), None)
    if isin_col is None:
        raise ValueError(f"{csv_path.name} doit contenir une colonne ISIN (ex: 'ISIN').")
    if w_col is None:
        raise ValueError(f"{csv_path.name} doit contenir une colonne de poids (ex: 'Poids' ou 'Poids (%)').")

    df = df[[isin_col, w_col]].copy()
    df[isin_col] = df[isin_col].astype(str).str.strip()
    df[w_col] = pd.to_numeric(df[w_col], errors="coerce")
    df = df.dropna(subset=[isin_col, w_col]).copy()
    df = df[df[isin_col].ne("") & (df[w_col] > 0)].copy()

    if len(df):
        if float(df[w_col].max()) > 1.0 + 1e-12:
            df[w_col] = df[w_col] / 100.0

    w = df.groupby(isin_col, as_index=True)[w_col].sum().astype(float)

    tickers_set = set(map(str, tickers))
    w = w[w.index.astype(str).isin(tickers_set)].copy()
    if w.empty:
        raise ValueError(
            "Aucun ISIN du CSV ne matche les tickers backtestés. "
            f"Vérifie la colonne ISIN et l'encodage du CSV: {csv_path}"
        )

    return w / (float(w.sum()) + 1e-12)


def _make_rebalance_dates(index: pd.DatetimeIndex, *, freq: str) -> list[pd.Timestamp]:
    """Dates de rebal annuelles: début + fins d’année (YE) + fin."""
    idx = index.dropna().sort_values()
    if len(idx) == 0:
        return []

    freq = str(freq).upper().strip()
    if freq != "Y":
        raise ValueError("REBAL_FREQ doit être 'Y' (annuel uniquement).")

    anchors = pd.date_range(idx[0], idx[-1], freq="YE")
    dates = [pd.Timestamp(idx[0])]

    for d in anchors:
        d2 = idx[idx <= d]
        if len(d2):
            dates.append(pd.Timestamp(d2[-1]))

    dates.append(pd.Timestamp(idx[-1]))

    out: list[pd.Timestamp] = []
    for d in dates:
        if not out or d > out[-1]:
            out.append(d)

    return out if len(out) >= 2 else [pd.Timestamp(idx[0]), pd.Timestamp(idx[-1])]


def _calendar_returns(v: pd.Series) -> pd.Series:
    """Calendaire (%): (fin d’année / début d’année - 1)*100."""
    v = v.dropna().astype(float)
    if v.empty:
        return v

    y_end = v.resample("YE").last()
    y_start = v.resample("YE").first()
    out = (y_end / y_start - 1.0) * 100.0
    out.index = out.index.to_period("Y").to_timestamp()
    return out


def _rolling_vol(ret: pd.Series, win: int = TRADING_DAYS) -> pd.Series:
    """Vol glissante annualisée en %."""
    ret = ret.dropna().astype(float)
    if ret.empty:
        return ret
    return ret.rolling(win).std(ddof=0) * np.sqrt(TRADING_DAYS) * 100.0


def _scores_12m_1m_from_returns(ret: pd.DataFrame) -> pd.DataFrame:
    """Momentum 12m-1m (sur rendements simples) via log-returns."""
    lr = np.log1p(ret.replace([np.inf, -np.inf], np.nan))
    mom12 = np.exp(lr.rolling(TRADING_DAYS, min_periods=TRADING_DAYS).sum()) - 1.0
    mom1 = np.exp(lr.rolling(21, min_periods=21).sum()) - 1.0
    return (mom12 - mom1).replace([np.inf, -np.inf], np.nan)


def _tilt_weights_from_scores(
    tickers: list[str],
    score_row: pd.Series,
    *,
    w_total: float,
    tilt_strength: float,
    z_clip: float = 2.0,
    min_mult: float = 0.40,
    max_mult: float = 2.00,
) -> pd.Series:
    """Base égalitaire -> multiplicateur basé z-score -> normalise -> * w_total."""
    tickers = [str(x) for x in tickers]
    if not tickers:
        return pd.Series(dtype=float)

    base = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
    s = score_row.reindex(tickers).astype(float)

    if s.notna().sum() >= 3 and float(s.std(ddof=0)) > 0:
        z = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
        z = z.clip(-z_clip, z_clip).fillna(0.0)
    else:
        z = pd.Series(0.0, index=tickers, dtype=float)

    mult = (1.0 + float(tilt_strength) * z).clip(min_mult, max_mult)
    w = base * mult
    w = w / (float(w.sum()) + 1e-12)
    return w * float(w_total)


def _enforce_min_max_sum(
    w: pd.Series,
    *,
    total: float,
    min_w: float | None,
    max_w: float | None,
    iters: int = 80,
) -> pd.Series:
    """Projette des poids >=0 sur somme=total + contraintes min/max (itératif)."""
    w = w.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if len(w) == 0:
        return w

    total = float(total)
    if total <= 0:
        return w * 0.0

    s = float(w.sum())
    w = (w / s) if s > 0 else (w * 0.0 + 1.0 / len(w))
    w = w * total

    if min_w is None and max_w is None:
        return w

    min_w = None if min_w is None else float(min_w)
    max_w = None if max_w is None else float(max_w)

    for _ in range(iters):
        changed = False

        if min_w is not None:
            low = w < min_w
            if low.any():
                deficit = float((min_w - w[low]).sum())
                w[low] = min_w
                changed = True

                free = ~low
                free_sum = float(w[free].sum())
                if deficit > 0 and free.any():
                    w[free] = w[free] * (1.0 - deficit / free_sum) if free_sum > 1e-12 else 0.0

        if max_w is not None:
            high = w > max_w
            if high.any():
                excess = float((w[high] - max_w).sum())
                w[high] = max_w
                changed = True

                free = ~high
                free_sum = float(w[free].sum())
                if excess > 0 and free.any():
                    w[free] = w[free] + (w[free] / free_sum) * excess if free_sum > 1e-12 else 0.0

        s2 = float(w.sum())
        if s2 > 0:
            w = w / s2 * total

        if not changed:
            break

    return w


def _weights_at_rebalance(
    *,
    df_static: pd.DataFrame,
    scores_row_all: pd.Series | None,
    w_bonds_fixed: float,
    w_actions_fixed: float,
    def_min: float,
    def_max: float,
    def_base: float,
    def_k: float,
    core_min_per_fund: float,
    max_line_weight: float,
    core_tilt_strength: float,
    def_tilt_strength: float,
) -> pd.Series:
    """Construit les poids (bonds fixed + actions fixed) avec poche défense pilotée."""
    df = df_static.copy()
    df["ISIN"] = df["ISIN"].astype(str).str.strip()

    tick_bonds = df.loc[df["Bucket"].eq("Obligataire"), "ISIN"].tolist()
    tick_def = df.loc[df["Bucket"].eq("Actions - Défense"), "ISIN"].tolist()
    tick_core = df.loc[df["Bucket"].eq("Actions - Core"), "ISIN"].tolist()

    w_bonds = pd.Series(dtype=float)
    if tick_bonds:
        w_bonds = _enforce_min_max_sum(
            pd.Series(1.0, index=tick_bonds),
            total=float(w_bonds_fixed),
            min_w=None,
            max_w=float(max_line_weight),
        )

    def_signal = 0.0
    if scores_row_all is not None and tick_def:
        v = float(scores_row_all.reindex(tick_def).mean())
        def_signal = 0.0 if not np.isfinite(v) else v

    w_def_total = float(def_base + def_k * def_signal)
    w_def_total = float(np.clip(w_def_total, float(def_min), min(float(def_max), float(w_actions_fixed))))
    w_core_total = float(max(0.0, float(w_actions_fixed) - w_def_total))

    w_def = pd.Series(dtype=float)
    if tick_def:
        if scores_row_all is None:
            raw = pd.Series(1.0, index=tick_def)
        else:
            raw = _tilt_weights_from_scores(
                tickers=tick_def,
                score_row=scores_row_all,
                w_total=w_def_total,
                tilt_strength=float(def_tilt_strength),
            )
        w_def = _enforce_min_max_sum(raw, total=w_def_total, min_w=None, max_w=float(max_line_weight))

    w_core = pd.Series(dtype=float)
    if tick_core:
        if scores_row_all is None:
            raw = pd.Series(1.0, index=tick_core)
        else:
            raw = _tilt_weights_from_scores(
                tickers=tick_core,
                score_row=scores_row_all,
                w_total=w_core_total,
                tilt_strength=float(core_tilt_strength),
            )
        core_min = min(float(core_min_per_fund), w_core_total / max(1, len(tick_core)))
        w_core = _enforce_min_max_sum(raw, total=w_core_total, min_w=core_min, max_w=float(max_line_weight))

    w_all = pd.concat([w_bonds, w_def, w_core])
    if w_all.empty:
        return w_all

    w_all = w_all.groupby(level=0).sum()
    return w_all / (float(w_all.sum()) + 1e-12)


def get_clean_data_equilibre_fixed(
    *,
    INVESTMENT: float = 1_000_000.0,
    start_target: str = "2022-01-02",
    end_target: str = "2026-01-30",
    benchmark_weights=(0.50, 0.50),
    DATA_UNIVERSE_PATH: str | Path | None = None,
    DATA_CLIENT_PATH: str | Path | None = None,
    FX_PATH: str | Path | None = None,
    INDICES_PATH: str | Path | None = None,
    EAGG_PATH: str | Path | None = None,
    UNIVERSE_PATH: str | Path | None = None,
    HOLDINGS_EQ=None,
    NAME_MAP=None,
    BUCKET_MAP=None,
    APPLY_MOMENTUM_CORE: bool = True,
    REBAL_FREQ: str = "Y",
    MOM_TILT_STRENGTH: float = DEFAULT_CORE_TILT,
    DEF_TILT_STRENGTH: float = DEFAULT_DEF_TILT,
    MAX_LINE_WEIGHT: float = DEFAULT_MAX_LINE,
    BONDS_W_FIXED: float = BONDS_FIXED_W,
    ACTIONS_W_FIXED: float = ACTIONS_FIXED_W,
    DEF_W_MIN: float = DEF_MIN_W,
    CORE_MIN_W: float = CORE_MIN_PER_FUND,
    DEF_W_MAX: float = DEF_MAX_W,
    DEF_BASE_W: float = DEF_BASE,
    DEF_K_SIGNAL: float = DEF_K,
    INIT_WEIGHTS_CSV: str | Path | None = None,
):
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "Projet"

    data_universe_path = Path(DATA_UNIVERSE_PATH) if DATA_UNIVERSE_PATH is not None else (data_dir / "Univers-Prix.parquet")
    data_client_path = Path(DATA_CLIENT_PATH) if DATA_CLIENT_PATH is not None else (data_dir / "Data.xlsb")
    fx_path = Path(FX_PATH) if FX_PATH is not None else (data_dir / "forex.parquet")
    indices_path = Path(INDICES_PATH) if INDICES_PATH is not None else (data_dir / "Indices" / "indice_prices.parquet")
    eagg_path = Path(EAGG_PATH) if EAGG_PATH is not None else (data_dir / "Données historiques EAGG(1).csv")
    universe_path = Path(UNIVERSE_PATH) if UNIVERSE_PATH is not None else (data_dir / "UniversInvestissement.xlsb")

    init_csv_path = Path(INIT_WEIGHTS_CSV) if INIT_WEIGHTS_CSV is not None else (data_dir / "portefeuille_equilibre.csv")

    df_u = pd.read_parquet(data_universe_path).copy()
    df_u.columns = [str(c).strip().lower() for c in df_u.columns]
    frames = [df_u]

    try:
        df_c = pd.read_excel(data_client_path, engine="pyxlsb")
        df_c.columns = [str(c).strip().lower() for c in df_c.columns]
        frames.append(df_c)
    except Exception:
        try:
            df_c = pd.read_excel(data_client_path)
            df_c.columns = [str(c).strip().lower() for c in df_c.columns]
            frames.append(df_c)
        except Exception:
            pass

    prices_long = pd.concat(frames, ignore_index=True)
    fx_long = pd.read_parquet(fx_path).copy()

    if HOLDINGS_EQ is None:
        uni = pd.read_excel(universe_path, sheet_name=0, engine="pyxlsb")
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

        df_eq = build_equilibre_portfolio(uni)
        df_eq["ISIN"] = df_eq["ISIN"].astype(str).str.strip()
        df_eq = df_eq[df_eq["ISIN"].ne("")].copy()

        if df_eq.empty:
            dbg = build_equilibre_portfolio(uni)
            introuvables = dbg[dbg["ISIN"].astype(str).str.strip().eq("")].copy()
            msg = "Portefeuille équilibré vide: aucun nom n'a été résolu en ISIN."
            if not introuvables.empty:
                msg += "\nExemples introuvables:\n" + "\n".join(introuvables["Nom"].head(10).astype(str).tolist())
            raise ValueError(msg)

        HOLDINGS_EQ = list(zip(df_eq["ISIN"].astype(str), np.ones(len(df_eq), dtype=float)))
        if NAME_MAP is None:
            NAME_MAP = dict(zip(df_eq["ISIN"].astype(str), df_eq["Nom"].astype(str)))
        if BUCKET_MAP is None:
            BUCKET_MAP = dict(zip(df_eq["ISIN"].astype(str), df_eq["Bucket"].astype(str)))

    NAME_MAP = {} if NAME_MAP is None else NAME_MAP
    BUCKET_MAP = {} if BUCKET_MAP is None else BUCKET_MAP

    df_hold = pd.DataFrame(HOLDINGS_EQ, columns=["ISIN", "Poids"]).copy()
    df_hold["ISIN"] = df_hold["ISIN"].astype(str).str.strip()
    tickers = df_hold.loc[df_hold["ISIN"].ne(""), "ISIN"].astype(str).tolist()
    tickers = list(dict.fromkeys(tickers))

    df_static = pd.DataFrame({"ISIN": tickers}).merge(
        pd.DataFrame({"ISIN": list(BUCKET_MAP.keys()), "Bucket": list(BUCKET_MAP.values())}),
        on="ISIN",
        how="left",
    )
    df_static["Bucket"] = df_static["Bucket"].fillna("")

    start_dt = pd.to_datetime(start_target)
    end_dt = pd.to_datetime(end_target)

    holdings_probe = pd.DataFrame({"ticker": tickers, "val": np.ones(len(tickers), dtype=float)}).set_index("ticker")

    core_all = _build_portfolio_from_prices(
        prices_long=prices_long,
        fx_long=fx_long,
        holdings=holdings_probe,
        start_target=str(start_dt.date()),
        end_target=str(end_dt.date()),
        benchmark_weights=benchmark_weights,
        indices_path=indices_path,
        eagg_path=eagg_path,
        benchmark_code="NQGIN",
    )

    asset_ret = core_all.get("asset_ret")
    if not isinstance(asset_ret, pd.DataFrame) or asset_ret.empty:
        raise ValueError("asset_ret indisponible: impossible de backtester la stratégie.")

    asset_ret = asset_ret.copy()
    asset_ret.index = pd.to_datetime(asset_ret.index)
    asset_ret = asset_ret.sort_index().reindex(columns=tickers)
    asset_ret = asset_ret.loc[(asset_ret.index >= start_dt) & (asset_ret.index <= end_dt)]

    idx = asset_ret.index.dropna()
    if len(idx) < 60:
        raise ValueError("Historique trop court (il faut au moins ~3 mois de points).")

    bm_v = core_all.get("bm_v")
    if not isinstance(bm_v, pd.Series) or bm_v.empty:
        raise ValueError("bm_v indisponible (benchmark).")

    bm_v = bm_v.copy()
    bm_v.index = pd.to_datetime(bm_v.index)
    bm_v = bm_v.sort_index().loc[(bm_v.index >= idx[0]) & (bm_v.index <= idx[-1])]

    scores_ts = _scores_12m_1m_from_returns(asset_ret) if APPLY_MOMENTUM_CORE else None
    reb_dates = _make_rebalance_dates(idx, freq=REBAL_FREQ)

    pf_nav = pd.Series(index=idx, dtype=float)
    nav = float(INVESTMENT)

    w_init_csv = _load_initial_weights_csv(init_csv_path, tickers)
    w_prev = pd.Series(0.0, index=tickers, dtype=float)
    w_prev.loc[w_init_csv.index.astype(str)] = w_init_csv.values
    w_prev = w_prev / (float(w_prev.sum()) + 1e-12)

    weights_hist: dict[int, pd.Series] = {int(pd.Timestamp(reb_dates[0]).year): w_prev.copy()}

    for k in range(len(reb_dates) - 1):
        d0 = pd.Timestamp(reb_dates[k])
        d1 = pd.Timestamp(reb_dates[k + 1])

        is_last = (k == len(reb_dates) - 2)
        seg_mask = ((idx >= d0) & (idx <= d1)) if is_last else ((idx >= d0) & (idx < d1))
        seg_idx = idx[seg_mask]
        if len(seg_idx) == 0:
            continue

        if k >= 1:
            sc_row = None
            if APPLY_MOMENTUM_CORE and scores_ts is not None and len(scores_ts.loc[:d0]) > 0:
                cand = scores_ts.loc[:d0].iloc[-1]
                sc_row = cand if cand.notna().sum() >= 3 else None

            w_prev = _weights_at_rebalance(
                df_static=df_static,
                scores_row_all=sc_row,
                w_bonds_fixed=float(BONDS_W_FIXED),
                w_actions_fixed=float(ACTIONS_W_FIXED),
                def_min=float(DEF_W_MIN),
                def_max=float(DEF_W_MAX),
                def_base=float(DEF_BASE_W),
                def_k=float(DEF_K_SIGNAL),
                core_min_per_fund=float(CORE_MIN_W),
                max_line_weight=float(MAX_LINE_WEIGHT),
                core_tilt_strength=float(MOM_TILT_STRENGTH),
                def_tilt_strength=float(DEF_TILT_STRENGTH),
            )
            weights_hist[int(pd.Timestamp(d0).year)] = w_prev.copy()

        seg_ret = asset_ret.reindex(seg_idx).fillna(0.0)
        pf_ret = seg_ret.mul(w_prev, axis=1).sum(axis=1)

        seg_nav = (1.0 + pf_ret).cumprod() * nav
        pf_nav.loc[seg_idx] = seg_nav.values
        nav = float(seg_nav.iloc[-1])

    weights_hist_df = pd.DataFrame(weights_hist).T.sort_index()

    pf_v = pf_nav.dropna()
    if pf_v.empty:
        raise ValueError("Backtest vide: pf_v vide.")

    bm_v = bm_v.reindex(pf_v.index).dropna()
    df_align = pd.concat([pf_v.rename("PF"), bm_v.rename("BM")], axis=1).dropna()
    pf_v = df_align["PF"]
    bm_v = df_align["BM"]

    ret_pf = pf_v.pct_change(fill_method=None).dropna()
    ret_bm = bm_v.pct_change(fill_method=None).dropna()

    ann_p = _calendar_returns(pf_v)
    ann_b = _calendar_returns(bm_v)
    r_p = _rolling_vol(ret_pf, win=TRADING_DAYS)
    r_b = _rolling_vol(ret_bm, win=TRADING_DAYS)

    dur_years = (pf_v.index[-1] - pf_v.index[0]).days / 365.25
    mu_p = float((pf_v.iloc[-1] / pf_v.iloc[0]) ** (1 / dur_years) - 1) if dur_years > 0 else np.nan
    vol_p = float(ret_pf.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(ret_pf) else np.nan

    w_last = w_prev.copy()
    tickers_final = w_last.index.astype(str).tolist()
    exposure_eur = w_last.astype(float) * float(pf_v.iloc[-1])

    w_data = (
        pd.DataFrame(
            {
                "ISIN": tickers_final,
                "Nom du Fonds": [NAME_MAP.get(t, t) for t in tickers_final],
                "Bucket": [BUCKET_MAP.get(t, "") for t in tickers_final],
                "Exposition (€)": exposure_eur.reindex(tickers_final).astype(float).values,
                "Poids (%)": (w_last.reindex(tickers_final).astype(float) * 100.0).round(2).values,
                "Contr. Perf (%)": np.nan,
                "Contr. Risque (%)": np.nan,
            }
        )
        .sort_values("Poids (%)", ascending=False)
        .reset_index(drop=True)
    )

    geo_df = (
        w_data.assign(P=lambda d: d["Poids (%)"] / 100.0)
        .groupby("Bucket", as_index=False)["P"]
        .sum()
        .rename(columns={"Bucket": "Géo"})
        if w_data["Bucket"].astype(str).str.strip().ne("").any()
        else pd.DataFrame({"Géo": ["Portefeuille"], "P": [1.0]})
    )

    sec_df = pd.DataFrame({"Sec": ["N/A"], "P": [1.0]})

    missing = core_all.get("missing_isins", [])
    missing_list = sorted(map(str, missing)) if isinstance(missing, (list, tuple, set)) else []

    return (
        pf_v,
        bm_v,
        ann_p,
        ann_b,
        r_p,
        r_b,
        w_data,
        geo_df,
        sec_df,
        mu_p,
        vol_p,
        missing_list,
        asset_ret,
        weights_hist_df,
    )