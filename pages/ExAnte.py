import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Risk Management Ex-Ante", layout="wide")

nav_left, _ = st.columns([3, 1.8])
with nav_left:
    if st.button("←", help="Page précédente"):
        st.switch_page("pages/DetailFond.py")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Calculs.eq_engine import compute_ex_ante_risk, _norm_key  # noqa: E402

CSV_PTF_PATH = os.path.join(PROJECT_ROOT, "Projet", "portefeuille_equilibre.csv")


def get_clean_session_data():
    a_ret = st.session_state.get("eq_asset_ret")
    p_v = st.session_state.get("eq_pf_v")
    b_v = st.session_state.get("eq_bm_v")

    if isinstance(a_ret, tuple):
        a_ret = a_ret[12]
    if isinstance(p_v, tuple):
        p_v = p_v[0]
    if isinstance(b_v, tuple):
        b_v = b_v[1]
    return a_ret, p_v, b_v


def get_weights_and_names():
    df_sel = st.session_state.get("selected_portfolio_df")
    if isinstance(df_sel, pd.DataFrame) and not df_sel.empty and {"ISIN", "Poids"}.issubset(df_sel.columns):
        w_ser = df_sel.set_index("ISIN")["Poids"].astype(float)
        name_map = dict(zip(df_sel["ISIN"].apply(_norm_key), df_sel.get("Nom", df_sel["ISIN"]).astype(str)))
        return w_ser, name_map

    w_df_csv = pd.read_csv(CSV_PTF_PATH)
    w_ser = w_df_csv.set_index("ISIN")["Poids"].astype(float)
    name_map = dict(zip(w_df_csv["ISIN"].apply(_norm_key), w_df_csv["Nom"].astype(str)))
    return w_ser, name_map


asset_ret, pf_v, bm_v = get_clean_session_data()
if asset_ret is None or pf_v is None or bm_v is None:
    st.error("Données manquantes. Veuillez charger la page Rétrospectif.")
    st.stop()

w_ser, name_map = get_weights_and_names()

res = compute_ex_ante_risk(asset_ret, w_ser)
vol_ex_ante = float(res["vol_ex_ante"])
mu_ex_ante = float((res["returns_final"].mean() * 252).values @ res["weights_final"].values)

ret_pf_hist = pf_v.pct_change(fill_method=None).dropna()
lambda_ewma = 0.94
v_t = float(ret_pf_hist.var())

ewma_vars = []
for r in ret_pf_hist.values:
    v_t = lambda_ewma * v_t + (1 - lambda_ewma) * float(r**2)
    ewma_vars.append(v_t)

sigma_next_day = float(np.sqrt(ewma_vars[-1]))
confidence_level = 0.95
z_score = float(norm.ppf(1 - confidence_level))

var_95_1d = abs(z_score * sigma_next_day)
cvar_95_1d = abs(sigma_next_day * (norm.pdf(z_score) / (1 - confidence_level)))

corr_mat = res["corr"]
avg_corr = float(corr_mat.values[np.triu_indices(corr_mat.shape[0], k=1)].mean())

ret_bm_inst = bm_v.pct_change(fill_method=None).reindex(ret_pf_hist.index).dropna()
common_idx = ret_pf_hist.index.intersection(ret_bm_inst.index)
beta_val = float(
    np.cov(ret_pf_hist.loc[common_idx], ret_bm_inst.loc[common_idx])[0, 1]
    / np.var(ret_bm_inst.loc[common_idx])
)

st.markdown("<h1 style='color: #FF4B4B;'>Risk Management Ex-Ante</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
<div style='background:rgba(138,43,226,0.1); border-left:5px solid #8a2be2; padding:15px; border-radius:5px;'>
  <b>Volatilité Ex-Ante (Annuelle)</b><br>
  <span style='font-size:24px; font-weight:bold; color:#8a2be2;'>{vol_ex_ante*100:.2f}%</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"\sigma_p = \sqrt{W^T \Omega W}")

with col2:
    st.markdown(
        f"""
<div style='background:rgba(255,75,75,0.1); border-left:5px solid #FF4B4B; padding:15px; border-radius:5px;'>
  <b>Espérance de Rendement (1 an)</b><br>
  <span style='font-size:24px; font-weight:bold; color:#FF4B4B;'>{mu_ex_ante*100:.2f}%</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"E[R_p] = \sum w_i \mu_i")

with col3:
    st.markdown(
        f"""
<div style='background:rgba(0,123,255,0.1); border-left:5px solid #007bff; padding:15px; border-radius:5px;'>
  <b>Corrélation Moyenne Ex-Ante</b><br>
  <span style='font-size:24px; font-weight:bold; color:#007bff;'>{avg_corr:.2f}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"\bar{\rho} = \frac{1}{N(N-1)} \sum \rho_{i,j}")

st.divider()

st.subheader("Projection de la VL à 5 ans (Mouvement Brownien Géométrique)")

horizon_days = 252 * 5
last_price = float(pf_v.iloc[-1])
dt = 1 / 252

Z_sim = np.random.standard_normal((horizon_days, 1000))
paths = (
    np.exp((mu_ex_ante - 0.5 * vol_ex_ante**2) * dt + vol_ex_ante * np.sqrt(dt) * Z_sim)
    .cumprod(axis=0)
    * last_price
)
perc = np.percentile(paths, [5, 50, 95], axis=1)
future_dates = pd.date_range(pf_v.index[-1], periods=horizon_days, freq="B")

col_stat, col_proj = st.columns([1, 2.5], gap="large")

with col_stat:
    st.markdown("#### Scénarios (Simulation)")

    df_scen = pd.DataFrame(
        {
            "Scénario": ["Optimiste (95%)", "Moyenne E(S)", "Médiane (50%)", "Stress (5%)"],
            "Valeur (€)": [
                f"{perc[2][-1]:,.0f} €",
                f"{paths[-1].mean():,.0f} €",
                f"{perc[1][-1]:,.0f} €",
                f"{perc[0][-1]:,.0f} €",
            ],
        }
    )

    st.markdown(
        """
<style>
.custom-table-wrap{
  border: 1px solid rgba(255,75,75,0.25);
  border-radius: 12px;
  overflow: hidden;
  background: rgba(255,255,255,0.02);
  margin-top: 6px;
}
table.custom-table{
  width:100%;
  border-collapse: collapse;
  font-size: 14px;
}
table.custom-table thead th{
  background:#FF4B4B;
  color:white;
  padding: 10px 12px;
  text-align:left;
  font-weight: 700;
}
table.custom-table tbody td{
  padding: 9px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
table.custom-table tbody tr:nth-child(even){
  background: rgba(255,255,255,0.03);
}
table.custom-table tbody tr:hover{
  background: rgba(255,75,75,0.10);
}
.td-right{ text-align:right; font-variant-numeric: tabular-nums; white-space:nowrap; }
</style>
""",
        unsafe_allow_html=True,
    )

    html = """
<div class="custom-table-wrap">
<table class="custom-table">
<thead>
<tr>
<th>Scénario</th>
<th style="text-align:right;">Valeur (€)</th>
</tr>
</thead>
<tbody>
"""
    for _, r in df_scen.iterrows():
        html += f"""
<tr>
<td>{r['Scénario']}</td>
<td class="td-right">{r['Valeur (€)']}</td>
</tr>
"""
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

    st.latex(r"S_t = S_0 e^{(\mu - \frac{1}{2}\sigma^2)t + \sigma W_t}")

with col_proj:
    fig_cone = go.Figure()
    fig_cone.add_trace(go.Scatter(x=future_dates, y=perc[2], line=dict(color="rgba(255, 75, 75, 0.4)", width=1), name="Haut (95%)"))
    fig_cone.add_trace(go.Scatter(x=future_dates, y=perc[0], fill="tonexty", fillcolor="rgba(255, 75, 75, 0.15)", line=dict(color="rgba(255, 75, 75, 0.4)", width=1), name="Bas (5%)"))
    fig_cone.add_trace(go.Scatter(x=future_dates, y=perc[1], name="Médiane", line=dict(color="#FF4B4B", dash="dash", width=3)))
    fig_cone.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
    st.plotly_chart(fig_cone, use_container_width=True)

st.divider()

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(
        f"""
<div style='background:rgba(255,165,0,0.1); border-left:5px solid #ffa500; padding:15px; border-radius:5px;'>
  <b>VaR Ex-Ante 95% (1 jour)</b><br>
  <span style='font-size:24px; font-weight:bold; color:#ffa500;'>{var_95_1d*100:.2f}%</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"VaR_{\alpha} = \sigma_{t+1} \cdot \Phi^{-1}(1-\alpha)")

with col5:
    st.markdown(
        f"""
<div style='background:rgba(255,0,0,0.1); border-left:5px solid #ff0000; padding:15px; border-radius:5px;'>
  <b>CVaR Ex-Ante 95% (1 jour)</b><br>
  <span style='font-size:24px; font-weight:bold; color:#ff0000;'>{cvar_95_1d*100:.2f}%</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"CVaR_{\alpha} = \sigma_{t+1} \frac{\phi(z_{\alpha})}{1-\alpha}")

with col6:
    st.markdown(
        f"""
<div style='background:rgba(40,167,69,0.1); border-left:5px solid #28a745; padding:15px; border-radius:5px;'>
  <b>Bêta Ex-Ante (vs Bench)</b><br>
  <span style='font-size:24px; font-weight:bold; color:#28a745;'>{beta_val:.2f}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"\beta_p = \frac{\sigma_{p,b}}{\sigma_b^2}")

st.divider()

st.subheader("Décomposition du Budget de Risque (CTR)")

df_risk = (
    pd.DataFrame(
        {
            "Nom": [name_map.get(i, i) for i in res["weights_final"].index],
            "Poids (%)": res["weights_final"].values * 100,
            "Budget Risque (%)": res["risk_budget_pct"].values,
        }
    )
    .sort_values("Budget Risque (%)", ascending=False)
)

c_bar, c_ana = st.columns([1.5, 1])

with c_bar:
    fig_ctr = go.Figure()
    fig_ctr.add_trace(go.Bar(y=df_risk["Nom"], x=df_risk["Budget Risque (%)"], orientation="h", marker_color="#FF4B4B", name="CTR"))
    fig_ctr.update_layout(height=550, template="plotly_dark", yaxis={"autorange": "reversed"}, xaxis_title="Contribution à la Volatilité (%)")
    st.plotly_chart(fig_ctr, use_container_width=True)

with c_ana:
    st.markdown("#### Analyse de la Contribution")
    top_f = df_risk.iloc[0]
    concentration = float(df_risk.iloc[:3]["Budget Risque (%)"].sum())

    st.info(f"Le principal risque est porté par **{top_f['Nom']}** ({top_f['Budget Risque (%)']:.1f}%).")
    st.write(f"Les 3 actifs les plus risqués concentrent **{concentration:.1f}%** de la volatilité totale.")
    st.write("---")
    st.latex(r"CTR_i = w_i \frac{\partial \sigma_p}{\partial w_i}")
    st.markdown("Une contribution élevée indique que l'actif domine le profil de risque, même avec un poids limité.")