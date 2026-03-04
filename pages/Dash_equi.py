import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Portefeuille Équilibré (Rétrospectif)", layout="wide")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Calculs.ptf_equilibre_engine import build_equilibre_portfolio, get_clean_data_equilibre_fixed  # noqa: E402
from Calculs.ptf_client_engine import compute_multi_horizon_metrics  # noqa: E402


CSS = """
<style>
.metrics-wrap{ width:100%; margin-top:6px; }
.metrics-table{ width:100%; border-collapse:collapse; font-size:13.5px; }
.metrics-table thead th{ text-align:right; padding:10px 12px; border-bottom:1px solid #2a2a2a; color:#2a76ff; font-weight:700; }
.metrics-table thead th:first-child{ text-align:left; color:#ffffff; }
.metrics-table tbody td{ padding:10px 12px; border-bottom:1px solid #2a2a2a; vertical-align:middle; }
.metrics-table tbody td:first-child{ text-align:left; color:#ffffff; font-weight:600; white-space:nowrap; }
.metrics-table tbody td:not(:first-child){ text-align:right; color:#ffffff; font-variant-numeric: tabular-nums; }
.metrics-muted{ color:#9aa0a6; font-size:12px; margin-top:6px; }

.alloc-wrap{ width:100%; margin-top:6px; }
.alloc-table{ width:100%; border-collapse:collapse; font-size:13.5px; }
.alloc-table thead th{ text-align:right; padding:10px 12px; border-bottom:1px solid #2a2a2a; color:#2a76ff; font-weight:700; }
.alloc-table thead th:first-child{ text-align:left; color:#ffffff; }
.alloc-table tbody td{ padding:10px 12px; border-bottom:1px solid #2a2a2a; vertical-align:middle; }
.alloc-table tbody td:first-child{ text-align:left; color:#ffffff; font-weight:600; white-space:nowrap; }
.alloc-table tbody td:not(:first-child){ text-align:right; color:#ffffff; font-variant-numeric: tabular-nums; }

.perf-table-wrap { width: 100%; border-top: 1px solid #2a2a2a; }
.perf-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.perf-table thead th{ text-align: right; font-weight: 700; color: #2a76ff; padding: 10px 12px; border-bottom: 1px solid #2a2a2a; }
.perf-table thead th:first-child, .perf-table thead th:nth-child(2){ text-align: left; color: #ffffff; font-weight: 600; }
.perf-table tbody td{ padding: 14px 12px; border-bottom: 1px solid #2a2a2a; vertical-align: top; }
.perf-table tbody td:first-child{ font-weight: 600; color: #ffffff; width: 22%; }
.perf-table tbody td:nth-child(2){ color: #b0b3b8; width: 46%; }
.perf-table tbody td:nth-child(3), .perf-table tbody td:nth-child(4){ text-align: right; color: #ffffff; font-variant-numeric: tabular-nums; width: 16%; }
.perf-muted { color: #9aa0a6; font-size: 12px; font-weight: 500; margin-bottom: 2px; }
.perf-note { display: flex; justify-content: flex-end; color: #9aa0a6; font-size: 12px; margin: 4px 0 8px 0; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

nav_left, _ = st.columns([3, 1.8])
with nav_left:
    if st.button("←", help="Page précédente"):
        st.switch_page("pages/DetailFond.py")
st.title("Portefeuille Équilibré Proposé (Rétrospectif 2022 - 2026)")

universe_path = os.path.join(PROJECT_ROOT, "Projet", "UniversInvestissement.xlsb")
csv_ptf_path = os.path.join(PROJECT_ROOT, "Projet", "portefeuille_equilibre.csv")

uni = pd.read_excel(universe_path, sheet_name=0, engine="pyxlsb").copy()
col_map = {}
for c in uni.columns:
    cl = str(c).strip().lower()
    if cl == "nom":
        col_map[c] = "Nom"
    elif "isin" in cl:
        col_map[c] = "ISIN"
    elif "cat" in cl:
        col_map[c] = "Categorie"
    elif "soci" in cl and "gest" in cl:
        col_map[c] = "Societe"
uni = uni.rename(columns=col_map).copy()
if not {"Nom", "ISIN"}.issubset(set(uni.columns)):
    raise ValueError("UniversInvestissement.xlsb doit contenir des colonnes 'Nom' et 'ISIN'.")
uni["Nom"] = uni["Nom"].astype(str).str.strip()
uni["ISIN"] = uni["ISIN"].astype(str).str.strip()
uni = uni[uni["Nom"].ne("") & uni["ISIN"].ne("")].drop_duplicates("ISIN")

df_eq_auto = build_equilibre_portfolio(uni).copy()
df_eq_auto["ISIN"] = df_eq_auto["ISIN"].astype(str).str.strip()
name_map_auto = dict(zip(df_eq_auto["ISIN"].astype(str), df_eq_auto["Nom"].astype(str)))
bucket_map_auto = dict(zip(df_eq_auto["ISIN"].astype(str), df_eq_auto["Bucket"].astype(str)))

df_csv = pd.read_csv(csv_ptf_path).copy()
if not {"ISIN", "Poids"}.issubset(df_csv.columns):
    raise ValueError("portefeuille_equilibre.csv doit contenir au moins 'ISIN' et 'Poids'.")
df_csv["ISIN"] = df_csv["ISIN"].astype(str).str.strip()
df_csv["Poids"] = pd.to_numeric(df_csv["Poids"], errors="coerce")
df_csv = df_csv.dropna(subset=["ISIN", "Poids"]).copy()
df_csv = df_csv[df_csv["ISIN"].ne("") & (df_csv["Poids"] > 0)].copy()
df_csv["Poids"] = df_csv["Poids"].astype(float) / (float(df_csv["Poids"].sum()) + 1e-12)

df_csv["Nom"] = df_csv["ISIN"].map(name_map_auto).fillna(df_csv.get("Nom", df_csv["ISIN"].astype(str)))
df_csv["Bucket"] = df_csv["ISIN"].map(bucket_map_auto).fillna("")

holdings_eq = list(zip(df_csv["ISIN"].astype(str), df_csv["Poids"].astype(float)))
name_map = dict(zip(df_csv["ISIN"].astype(str), df_csv["Nom"].astype(str)))
bucket_map = dict(zip(df_csv["ISIN"].astype(str), df_csv["Bucket"].astype(str)))

apply_momentum_core = True
rebal_freq = "Y"
mom_tilt_strength = 0.30
max_line_weight = 0.10
def_tilt_strength = 0.60

param_key = (
    f"MOM{int(100*mom_tilt_strength)}_{rebal_freq}_MAX{int(100*max_line_weight)}_"
    f"DEFtilt{int(100*def_tilt_strength)}_{int(apply_momentum_core)}"
)
key_prefix = f"eq_{param_key}_"

keys = {
    "pf_v": f"{key_prefix}pf_v",
    "bm_v": f"{key_prefix}bm_v",
    "ann_p": f"{key_prefix}ann_p",
    "ann_b": f"{key_prefix}ann_b",
    "r_p": f"{key_prefix}r_p",
    "r_b": f"{key_prefix}r_b",
    "w_data": f"{key_prefix}w_data",
    "geo_df": f"{key_prefix}geo_df",
    "sec_df": f"{key_prefix}sec_df",
    "mu_p": f"{key_prefix}mu_p",
    "vol_p": f"{key_prefix}vol_p",
    "missing": f"{key_prefix}missing",
    "asset_ret": f"{key_prefix}asset_ret",
    "weights_hist_df": f"{key_prefix}weights_hist_df",
}

if keys["pf_v"] not in st.session_state:
    (
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
        missing,
        asset_ret,
        weights_hist_df,
    ) = get_clean_data_equilibre_fixed(
        INVESTMENT=1_000_000.0,
        start_target="2022-01-02",
        end_target="2026-01-30",
        benchmark_weights=(0.50, 0.50),
        HOLDINGS_EQ=holdings_eq,
        NAME_MAP=name_map,
        BUCKET_MAP=bucket_map,
        APPLY_MOMENTUM_CORE=apply_momentum_core,
        REBAL_FREQ=rebal_freq,
        MOM_TILT_STRENGTH=mom_tilt_strength,
        DEF_TILT_STRENGTH=def_tilt_strength,
        MAX_LINE_WEIGHT=max_line_weight,
        BONDS_W_FIXED=0.50,
        ACTIONS_W_FIXED=0.50,
        DEF_W_MIN=0.15,
        CORE_MIN_W=0.02,
        DEF_W_MAX=0.20,
        DEF_BASE_W=0.10,
        DEF_K_SIGNAL=0.06,
    )

    st.session_state[keys["pf_v"]] = pf_v
    st.session_state[keys["bm_v"]] = bm_v
    st.session_state[keys["ann_p"]] = ann_p
    st.session_state[keys["ann_b"]] = ann_b
    st.session_state[keys["r_p"]] = r_p
    st.session_state[keys["r_b"]] = r_b
    st.session_state[keys["w_data"]] = w_data
    st.session_state[keys["geo_df"]] = geo_df
    st.session_state[keys["sec_df"]] = sec_df
    st.session_state[keys["mu_p"]] = mu_p
    st.session_state[keys["vol_p"]] = vol_p
    st.session_state[keys["missing"]] = missing
    st.session_state[keys["asset_ret"]] = asset_ret
    st.session_state[keys["weights_hist_df"]] = weights_hist_df
else:
    pf_v = st.session_state[keys["pf_v"]]
    bm_v = st.session_state[keys["bm_v"]]
    ann_p = st.session_state[keys["ann_p"]]
    ann_b = st.session_state[keys["ann_b"]]
    r_p = st.session_state[keys["r_p"]]
    r_b = st.session_state[keys["r_b"]]
    w_data = st.session_state[keys["w_data"]]
    geo_df = st.session_state[keys["geo_df"]]
    sec_df = st.session_state[keys["sec_df"]]
    mu_p = st.session_state.get(keys["mu_p"])
    vol_p = st.session_state.get(keys["vol_p"])
    missing = st.session_state.get(keys["missing"], [])
    asset_ret = st.session_state.get(keys["asset_ret"])
    weights_hist_df = st.session_state.get(keys["weights_hist_df"])

w_init = df_csv.set_index("ISIN")["Poids"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
w_init = w_init[w_init > 0]
w_init = w_init / (float(w_init.sum()) + 1e-12)

st.session_state["eq_last_prefix"] = key_prefix
st.session_state["eq_pf_v"] = pf_v
st.session_state["eq_bm_v"] = bm_v
st.session_state["eq_asset_ret"] = asset_ret
st.session_state["eq_name_map"] = name_map
st.session_state["eq_bucket_map"] = bucket_map
st.session_state["eq_w_init"] = w_init

c1, c2 = st.columns([1, 2.5], gap="large")

with c1:
    st.markdown(
        "<p style='color:#007BFF; font-size:16px; margin-bottom:0;'>"
        "VALEUR LIQUIDATIVE <span style='float:right; color:gray; font-size:12px;'>FIN PÉRIODE</span></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:#007BFF; font-size:48px; font-weight:bold; margin-top:0;'>{pf_v.iloc[-1]:,.2f} €</p>",
        unsafe_allow_html=True,
    )

    fig_vl = go.Figure(
        go.Scatter(
            x=pf_v.index,
            y=pf_v.values,
            fill="tozeroy",
            fillcolor="rgba(0, 123, 255, 0.1)",
            line=dict(color="#007BFF", width=2),
        )
    )
    fig_vl.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_vl, use_container_width=True)

with c2:
    df_align = pd.concat([pf_v.rename("PF"), bm_v.rename("BM")], axis=1).dropna()
    pf_idx = df_align["PF"] / df_align["PF"].iloc[0] * 100
    bm_idx = df_align["BM"] / df_align["BM"].iloc[0] * 100
    surperf = (pf_idx / bm_idx - 1.0) * 100

    fig_c = make_subplots(specs=[[{"secondary_y": True}]])
    fig_c.add_trace(
        go.Scatter(x=df_align.index, y=pf_idx, name="Portefeuille (base 100)", line=dict(color="#007BFF", width=3)),
        secondary_y=False,
    )
    fig_c.add_trace(
        go.Scatter(x=df_align.index, y=bm_idx, name="Benchmark (base 100)", line=dict(color="#FFA500", width=2.5)),
        secondary_y=False,
    )
    fig_c.add_trace(
        go.Scatter(
            x=df_align.index,
            y=surperf,
            name="Surperformance cumulée (%)",
            line=dict(color="rgba(255,255,255,0.85)", width=2),
        ),
        secondary_y=True,
    )

    fig_c.update_layout(
        title=dict(text="Performance cumulée vs Benchmark", x=0.02),
        height=280,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig_c.update_yaxes(title_text="Indice (base 100)", secondary_y=False)
    fig_c.update_yaxes(title_text="Surperformance (%)", secondary_y=True, ticksuffix="%", showgrid=False, zeroline=False)
    st.plotly_chart(fig_c, use_container_width=True)

st.divider()

c3, c4, c5 = st.columns([1.6, 1.6, 1.2], gap="large")

with c3:
    years = ann_p.index.year.astype(int).tolist()

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=years,
            y=ann_p.values,
            name="Portefeuille",
            marker_color="#007BFF",
            text=[f"{v:.1f}%" for v in ann_p.values],
            textposition="auto",
        )
    )
    fig_bar.add_trace(
        go.Bar(
            x=years,
            y=ann_b.values,
            name="Benchmark",
            marker_color="#FFA500",
            text=[f"{v:.1f}%" for v in ann_b.values],
            textposition="auto",
        )
    )

    fig_bar.update_layout(
        height=260,
        template="plotly_dark",
        barmode="group",
        margin=dict(l=0, r=0, t=50, b=0),
        title=dict(text="Rendement calendaire", x=0.02),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_bar.update_xaxes(tickmode="array", tickvals=years, ticktext=[str(y) for y in years])
    fig_bar.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_bar, use_container_width=True)

with c4:
    start_vol = pd.Timestamp("2023-01-01")
    r_p_23 = r_p.loc[r_p.index >= start_vol]
    r_b_23 = r_b.loc[r_b.index >= start_vol]

    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=r_p_23.index, y=r_p_23.values, name="Portefeuille", line=dict(color="#007BFF", width=2)))
    fig_v.add_trace(go.Scatter(x=r_b_23.index, y=r_b_23.values, name="Benchmark", line=dict(color="#FFA500", width=2, dash="dash")))
    fig_v.update_layout(
        height=260,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        title=dict(text="Volatilité glissante (252 jours)", x=0.02),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_v.update_xaxes(nticks=6)
    fig_v.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_v, use_container_width=True)

with c5:
    dfm = compute_multi_horizon_metrics(pf_v, bm_v)
    disp = dfm.copy()

    pct_rows = {"Rendement annualisé", "Volatilité annualisée", "Tracking Error"}
    for r in pct_rows:
        mask = disp["Indicateur"].eq(r)
        for coln in ["1 an", "3 ans", "5 ans"]:
            disp.loc[mask, coln] = disp.loc[mask, coln].apply(lambda x: "" if pd.isna(x) else f"{x*100:.2f}%")

    dec_rows = {"Beta", "Corrélation", "Information Ratio"}
    for r in dec_rows:
        mask = disp["Indicateur"].eq(r)
        for coln in ["1 an", "3 ans", "5 ans"]:
            disp.loc[mask, coln] = disp.loc[mask, coln].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    html = disp.to_html(index=False, escape=False, classes="metrics-table")
    st.markdown(f"<div class='metrics-wrap'>{html}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='metrics-muted'>Fenêtres glissantes finissant au {pf_v.index[-1].strftime('%d/%m/%Y')} (1Y≈252j, 3Y≈756j, 5Y≈1260j).</div>",
        unsafe_allow_html=True,
    )

st.divider()

c6, c7 = st.columns([2.2, 1], gap="large")

with c6:
    st.markdown("**Allocation : Poids, Performance et Risque**")

    years = list(weights_hist_df.index) if isinstance(weights_hist_df, pd.DataFrame) and not weights_hist_df.empty else []
    last_y = years[-1] if years else None
    prev_y = years[-2] if len(years) >= 2 else None
    prev2_y = years[-3] if len(years) >= 3 else None

    alloc_df = w_data.copy()
    alloc_df = alloc_df.rename(columns={"Poids (%)": f"Poids {last_y} (%)" if last_y is not None else "Poids (dernier) (%)"})

    if prev_y is not None:
        alloc_df[f"Poids {prev_y} (%)"] = alloc_df["ISIN"].map(weights_hist_df.loc[prev_y] * 100.0)
    else:
        alloc_df["Poids (N-1) (%)"] = np.nan

    if prev2_y is not None:
        alloc_df[f"Poids {prev2_y} (%)"] = alloc_df["ISIN"].map(weights_hist_df.loc[prev2_y] * 100.0)
    else:
        alloc_df["Poids (N-2) (%)"] = np.nan

    alloc_df = alloc_df.drop(columns=["Contr. Perf (%)", "Contr. Risque (%)"], errors="ignore")
    alloc_df["Exposition (€)"] = alloc_df["Exposition (€)"].apply(lambda x: f"{x:,.0f} €".replace(",", " "))

    for col in alloc_df.columns:
        if "Poids" in col:
            alloc_df[col] = alloc_df[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):.2f}%")

    html_alloc = alloc_df.to_html(index=False, escape=False, classes="alloc-table")
    st.markdown(f"<div class='alloc-wrap'>{html_alloc}</div>", unsafe_allow_html=True)

with c7:
    fig_g = px.pie(geo_df, values="P", names="Géo", hole=0.55)
    fig_g.update_traces(hole=0.55, textinfo="percent", textfont_size=14)
    fig_g.update_layout(
        title=dict(text="Buckets / Poches", x=0.02),
        height=320,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
    )
    st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
    st.markdown("#### Corrélations historiques (fonds)")

    if isinstance(asset_ret, pd.DataFrame) and not asset_ret.empty:
        cols = list(asset_ret.columns.astype(str))
        names = [name_map.get(c, c) for c in cols]
        corr_hist = asset_ret.corr().values.astype(float)

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_hist,
                x=names,
                y=names,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                colorbar=dict(title="Corr"),
                hovertemplate="Corr(%{y}, %{x}) = %{z:.2f}<extra></extra>",
            )
        )
        fig_corr.update_layout(height=520, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0))
        fig_corr.update_xaxes(tickangle=45, tickfont=dict(size=10))
        fig_corr.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("asset_ret indisponible : impossible d'afficher la matrice de corrélation.")

st.divider()

st.markdown("### SCÉNARIOS DE PERFORMANCES (nets de frais)")

inv = 10_000
fee_annual = 0.0135
labels = ["Scénario de tension", "Scénario défavorable", "Scénario intermédiaire", "Scénario favorable"]
shocks = [-3, -1.5, 0, 1.5]

if mu_p is None or vol_p is None or not np.isfinite(mu_p) or not np.isfinite(vol_p) or vol_p <= 0:
    st.warning("mu_p / vol_p indisponibles (ou vol<=0) : scénarios non calculables.")
else:
    mu_net = float(mu_p) - float(fee_annual)

    rows = []
    for label, z in zip(labels, shocks):
        v3 = float(inv * np.exp((mu_net - 0.5 * vol_p**2) * 3 + z * vol_p * np.sqrt(3)))
        v5 = float(inv * np.exp((mu_net - 0.5 * vol_p**2) * 5 + z * vol_p * np.sqrt(5)))
        r3 = (v3 / inv) ** (1 / 3) - 1
        r5 = (v5 / inv) ** (1 / 5) - 1

        rows.append(
            {
                "Scénarios": label,
                "": f"Projection nette de frais ({fee_annual*100:.2f}%/an)<br>Valeur estimée • Rendement annuel moyen",
                "3 ans": f"{v3:,.2f} €<br>{r3*100:.2f}%",
                "5 ans": f"{v5:,.2f} €<br>{r5*100:.2f}%",
            }
        )

    df_perf = pd.DataFrame(rows)
    st.markdown(f"<div class='perf-muted'>Investissement de {inv:,.2f}€</div>", unsafe_allow_html=True)
    st.markdown("<div class='perf-note'>Période de détention recommandée</div>", unsafe_allow_html=True)
    html = df_perf.to_html(index=False, escape=False, classes="perf-table")
    st.markdown(f"<div class='perf-table-wrap'>{html}</div>", unsafe_allow_html=True)

    st.caption(
        f"Hypothèse : rendement net = μ − frais, avec frais estimés à {fee_annual*100:.2f}%/an. "
        "Ces scénarios sont indicatifs (modèle GBM)."
    )