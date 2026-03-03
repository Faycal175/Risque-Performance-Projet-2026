import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Calculs.ptf_client_engine import compute_multi_horizon_metrics, get_clean_data  # noqa: E402

st.set_page_config(page_title="Audit Portefeuille Master 272", layout="wide")

nav_left, _ = st.columns([3, 1.8])
with nav_left:
    if st.button("←", help="Page précédente"):
        st.switch_page("app.py")

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
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_all_data():
    return get_clean_data()


pf_v, bm_v, ann_p, ann_b, r_p, r_b, w_data, geo_df, sec_df, mu_p, vol_p = load_all_data()

df_align = pd.concat([pf_v.rename("PF"), bm_v.rename("BM")], axis=1).dropna()
ret_pf = df_align["PF"].pct_change(fill_method=None).dropna()
ret_bm = df_align["BM"].pct_change(fill_method=None).dropna()

te_roll = ((ret_pf - ret_bm).rolling(252).std() * np.sqrt(252) * 100).dropna()

pf_idx = (1 + ret_pf).cumprod()
bm_idx = (1 + ret_bm).cumprod()
surperf = ((pf_idx / bm_idx) - 1) * 100
surperf = surperf.dropna()

st.title("Audit Stratégique de Portefeuille (2020 - 2026)")

c1, c2 = st.columns([1, 2.5])

with c1:
    st.markdown(
        "<p style='color:#007BFF; font-size:16px; margin-bottom:0;'>"
        "VALEUR LIQUIDATIVE <span style='float:right; color:gray; font-size:12px;'>AU 30/01/2026</span></p>",
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
    fig_c = make_subplots(specs=[[{"secondary_y": True}]])
    fig_c.add_trace(
        go.Scatter(
            x=df_align.index,
            y=(df_align["PF"] / df_align["PF"].iloc[0] * 100),
            name="Portefeuille",
            line=dict(color="#007BFF", width=3),
        ),
        secondary_y=False,
    )
    fig_c.add_trace(
        go.Scatter(
            x=df_align.index,
            y=(df_align["BM"] / df_align["BM"].iloc[0] * 100),
            name="Benchmark",
            line=dict(color="#FFA500", width=2.5),
        ),
        secondary_y=False,
    )
    fig_c.add_trace(
        go.Scatter(
            x=surperf.index,
            y=surperf.values,
            name="Surperformance cumulée",
            line=dict(width=2, dash="dot"),
        ),
        secondary_y=True,
    )

    fig_c.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_c.update_yaxes(title_text="Indice (base 100)", secondary_y=False)
    fig_c.update_yaxes(title_text="Surperformance (%)", secondary_y=True)
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
    fig_v.add_trace(go.Scatter(x=te_roll.index, y=te_roll.values, name="Tracking Error (252j)", line=dict(width=2, dash="dot")))
    fig_v.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), title="Volatilité & Tracking Error glissantes (252j)")
    fig_v.update_yaxes(title_text="% annualisé")
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


def style_donut(fig, title: str):
    fig.update_traces(hole=0.45, textinfo="percent", textfont_size=14)
    fig.update_layout(
        title=dict(text=title, x=0.02),
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=13),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
    )
    return fig


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
    fig_g = style_donut(fig_g, "Géo")
    st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    fig_s = px.pie(sec_df, values="P", names="Sec", hole=0.55)
    fig_s = style_donut(fig_s, "Secteur")
    st.plotly_chart(fig_s, use_container_width=True)

st.divider()