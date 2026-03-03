import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Portefeuille Équilibré (Rétrospectif)", layout="wide")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Calculs.ptf_equilibre_engine import build_equilibre_portfolio, get_clean_data_equilibre_fixed  # noqa: E402
from Calculs.ptf_client_engine import compute_multi_horizon_metrics  # noqa: E402

nav_left, _ = st.columns([3, 1.8])
with nav_left:
    if st.button("←", help="Page précédente"):
        st.switch_page("pages/DetailFond.py")

st.title("Portefeuille Équilibré Proposé (Rétrospectif 2022 - 2026)")

st.markdown(
    """
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
.alloc-muted{ color:#9aa0a6; font-size:12px; margin-top:6px; }

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
""",
    unsafe_allow_html=True,
)


def style_donut(fig, title: str):
    fig.update_traces(hole=0.55, textinfo="percent", textfont_size=14)
    fig.update_layout(
        title=dict(text=title, x=0.02),
        height=320,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=13),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
    )
    return fig


UNIVERSE_PATH = os.path.join(PROJECT_ROOT, "Projet", "UniversInvestissement.xlsb")


@st.cache_data
def load_universe_xlsb(path: str) -> pd.DataFrame:
    uni = pd.read_excel(path, sheet_name=0, engine="pyxlsb").copy()

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
    return uni


uni = load_universe_xlsb(UNIVERSE_PATH)
df_eq_auto = build_equilibre_portfolio(uni)

df_eq_valid = df_eq_auto.copy()
df_eq_valid["ISIN"] = df_eq_valid["ISIN"].astype(str).str.strip()
df_eq_valid["Poids"] = pd.to_numeric(df_eq_valid["Poids"], errors="coerce")
df_eq_valid = df_eq_valid.dropna(subset=["ISIN", "Poids"])
df_eq_valid = df_eq_valid[df_eq_valid["ISIN"].ne("") & (df_eq_valid["Poids"] > 0)].copy()

if df_eq_valid.empty:
    introuvables = df_eq_auto[df_eq_auto["ISIN"].astype(str).str.strip().eq("")]
    st.error("Aucun fonds n’a été résolu en ISIN depuis EQUILIBRE_MANUAL_NAMES (portefeuille vide).")
    if not introuvables.empty:
        st.write("Exemples introuvables :")
        st.write(introuvables[["Bucket", "Nom"]].head(20))
    st.stop()

HOLDINGS_EQ = list(zip(df_eq_valid["ISIN"].astype(str), df_eq_valid["Poids"].astype(float)))
NAME_MAP = dict(zip(df_eq_valid["ISIN"].astype(str), df_eq_valid["Nom"].astype(str)))
BUCKET_MAP = dict(zip(df_eq_valid["ISIN"].astype(str), df_eq_valid["Bucket"].astype(str)))

KEY_PREFIX = "eq_"

if f"{KEY_PREFIX}pf_v" not in st.session_state:
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
    ) = get_clean_data_equilibre_fixed(
        INVESTMENT=1_000_000.0,
        start_target="2022-01-02",
        end_target="2026-01-30",
        benchmark_weights=(0.50, 0.50),
        HOLDINGS_EQ=HOLDINGS_EQ,
        NAME_MAP=NAME_MAP,
        BUCKET_MAP=BUCKET_MAP,
    )

    st.session_state[f"{KEY_PREFIX}pf_v"] = pf_v
    st.session_state[f"{KEY_PREFIX}bm_v"] = bm_v
    st.session_state[f"{KEY_PREFIX}ann_p"] = ann_p
    st.session_state[f"{KEY_PREFIX}ann_b"] = ann_b
    st.session_state[f"{KEY_PREFIX}r_p"] = r_p
    st.session_state[f"{KEY_PREFIX}r_b"] = r_b
    st.session_state[f"{KEY_PREFIX}w_data"] = w_data
    st.session_state[f"{KEY_PREFIX}geo_df"] = geo_df
    st.session_state[f"{KEY_PREFIX}sec_df"] = sec_df
    st.session_state[f"{KEY_PREFIX}mu_p"] = mu_p
    st.session_state[f"{KEY_PREFIX}vol_p"] = vol_p
    st.session_state[f"{KEY_PREFIX}missing"] = missing
    st.session_state[f"{KEY_PREFIX}asset_ret"] = asset_ret
else:
    pf_v = st.session_state[f"{KEY_PREFIX}pf_v"]
    bm_v = st.session_state[f"{KEY_PREFIX}bm_v"]
    ann_p = st.session_state[f"{KEY_PREFIX}ann_p"]
    ann_b = st.session_state[f"{KEY_PREFIX}ann_b"]
    r_p = st.session_state[f"{KEY_PREFIX}r_p"]
    r_b = st.session_state[f"{KEY_PREFIX}r_b"]
    w_data = st.session_state[f"{KEY_PREFIX}w_data"]
    geo_df = st.session_state[f"{KEY_PREFIX}geo_df"]
    sec_df = st.session_state[f"{KEY_PREFIX}sec_df"]
    mu_p = st.session_state.get(f"{KEY_PREFIX}mu_p")
    vol_p = st.session_state.get(f"{KEY_PREFIX}vol_p")
    missing = st.session_state.get(f"{KEY_PREFIX}missing", [])
    asset_ret = st.session_state.get(f"{KEY_PREFIX}asset_ret")

c1, c2 = st.columns([1, 2.5])

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
    )
    st.plotly_chart(fig_vl, use_container_width=True)

with c2:
    fig_c = go.Figure()
    fig_c.add_trace(
        go.Scatter(
            x=pf_v.index,
            y=(pf_v / pf_v.iloc[0] * 100),
            name="Portefeuille",
            line=dict(color="#007BFF", width=3),
        )
    )
    fig_c.add_trace(
        go.Scatter(
            x=bm_v.index,
            y=(bm_v / bm_v.iloc[0] * 100),
            name="Benchmark",
            line=dict(color="#FFA500", width=2.5),
        )
    )
    fig_c.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_c, use_container_width=True)

st.divider()

c3, c4, c5 = st.columns([1.5, 1.5, 1.2])

with c3:
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=ann_p.index.year,
            y=ann_p.values,
            name="PTF",
            marker_color="#007BFF",
            text=ann_p.values.round(1).astype(str) + "%",
            textposition="auto",
        )
    )
    fig_bar.add_trace(
        go.Bar(
            x=ann_b.index.year,
            y=ann_b.values,
            name="Bench",
            marker_color="#FFA500",
            text=ann_b.values.round(1).astype(str) + "%",
            textposition="auto",
        )
    )
    fig_bar.update_layout(height=300, barmode="group", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

with c4:
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=r_p.index, y=r_p.values, name="PTF Vol", line=dict(color="#007BFF", width=2)))
    fig_v.add_trace(
        go.Scatter(x=r_b.index, y=r_b.values, name="Bench Vol", line=dict(color="#FFA500", width=2, dash="dash"))
    )
    fig_v.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), title="Volatilité Glissante (252j)")
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

c6, c7 = st.columns([2.2, 1])

with c6:
    st.markdown("**Allocation : Poids, Performance et Risque**")
    alloc_df = w_data.copy()

    alloc_df["Exposition (€)"] = alloc_df["Exposition (€)"].apply(lambda x: f"{x:,.0f} €".replace(",", " "))
    for coln in ["Poids (%)", "Contr. Perf (%)", "Contr. Risque (%)"]:
        alloc_df[coln] = alloc_df[coln].apply(lambda x: f"{x:.2f}%")

    html_alloc = alloc_df.to_html(index=False, escape=False, classes="alloc-table")
    st.markdown(f"<div class='alloc-wrap'>{html_alloc}</div>", unsafe_allow_html=True)

with c7:
    fig_g = px.pie(geo_df, values="P", names="Géo", hole=0.55)
    fig_g = style_donut(fig_g, "Buckets / Poches")
    st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)

    fig_s = px.pie(sec_df, values="P", names="Sec", hole=0.55)
    fig_s = style_donut(fig_s, "Secteur / Type")
    st.plotly_chart(fig_s, use_container_width=True)

st.divider()

st.markdown("### SCÉNARIOS DE PERFORMANCES")

inv = 10000
labels = ["Scénario de tension", "Scénario défavorable", "Scénario intermédiaire", "Scénario favorable"]
shocks = [-3, -1.5, 0, 1.5]

if mu_p is None or vol_p is None or not np.isfinite(mu_p) or not np.isfinite(vol_p) or vol_p <= 0:
    st.warning("mu_p / vol_p indisponibles (ou vol<=0) : scénarios non calculables.")
else:
    def scenario_row(h, sig):
        val_f = inv * np.exp((mu_p - 0.5 * vol_p**2) * h + sig * vol_p * np.sqrt(h))
        rend_a = (val_f / inv) ** (1 / h) - 1
        return val_f, rend_a

    rows = []
    for label, sig in zip(labels, shocks):
        v3, r3 = scenario_row(3, sig)
        v5, r5 = scenario_row(5, sig)
        rows.append(
            {
                "Scénarios": label,
                "": "Ce que vous pourriez récupérer après déduction des coûts<br>Rendement annuel moyen",
                "3 ans": f"{v3:,.2f} €<br>{r3*100:.2f}%",
                "5 ans": f"{v5:,.2f} €<br>{r5*100:.2f}%",
            }
        )

    df_perf = pd.DataFrame(rows)

    st.markdown(f"<div class='perf-muted'>Investissement de {inv:,.2f}€</div>", unsafe_allow_html=True)
    st.markdown("<div class='perf-note'>Période de détention recommandée</div>", unsafe_allow_html=True)

    html = df_perf.to_html(index=False, escape=False, classes="perf-table")
    st.markdown(f"<div class='perf-table-wrap'>{html}</div>", unsafe_allow_html=True)

    st.caption("Hypothèse d'investissement de 10 000,00€. Horizon recommandé : 5 ans.")