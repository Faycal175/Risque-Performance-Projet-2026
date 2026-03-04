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

st.markdown(
    """
<style>
.pretty-table-wrap{
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  overflow: hidden;
  background: rgba(255,255,255,0.02);
  margin-top: 8px;
}
table.pretty-table{
  width:100%;
  border-collapse: collapse;
  font-size: 13.5px;
}
table.pretty-table thead th{
  padding: 10px 12px;
  text-align: left;
  font-weight: 800;
  letter-spacing: .2px;
  color: white;
  background: linear-gradient(90deg, rgba(255,75,75,0.95), rgba(255,75,75,0.65));
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
table.pretty-table tbody td{
  padding: 10px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.07);
  vertical-align: middle;
  color: rgba(255,255,255,0.92);
}
table.pretty-table tbody tr:nth-child(even){ background: rgba(255,255,255,0.02); }
table.pretty-table tbody tr:hover{ background: rgba(255,75,75,0.10); }
td.td-right, th.td-right{
  text-align:right !important;
  font-variant-numeric: tabular-nums;
  white-space:nowrap;
}
.badge{
  display:inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.10);
}
.badge-purple{ color:#b58bff; border-color: rgba(181,139,255,0.35); background: rgba(181,139,255,0.10); }
.badge-blue{ color:#2a76ff; border-color: rgba(42,118,255,0.35); background: rgba(42,118,255,0.08); }
.badge-green{ color:#28a745; border-color: rgba(40,167,69,0.35); background: rgba(40,167,69,0.10); }
.badge-red{ color:#FF4B4B; border-color: rgba(255,75,75,0.35); background: rgba(255,75,75,0.10); }
</style>
""",
    unsafe_allow_html=True,
)

asset_ret = st.session_state.get("eq_asset_ret")
pf_v = st.session_state.get("eq_pf_v")
bm_v = st.session_state.get("eq_bm_v")

if not isinstance(asset_ret, pd.DataFrame) or asset_ret.empty or not isinstance(pf_v, pd.Series) or pf_v.empty or not isinstance(bm_v, pd.Series) or bm_v.empty:
    st.error("Données manquantes. Veuillez d'abord charger la page 'Rétrospectif' (Dash_equi).")
    st.stop()

w_ser = st.session_state.get("eq_w_init")
name_map0 = st.session_state.get("eq_name_map")

if isinstance(w_ser, pd.Series) and not w_ser.empty:
    w_ser = w_ser.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    w_ser = w_ser[w_ser > 0]
    w_ser = w_ser / (float(w_ser.sum()) + 1e-12)
    if isinstance(name_map0, dict) and len(name_map0) > 0:
        name_map = {_norm_key(k): str(v) for k, v in name_map0.items()}
    else:
        name_map = {_norm_key(k): str(k) for k in w_ser.index}
else:
    df_sel = st.session_state.get("selected_portfolio_df")
    if isinstance(df_sel, pd.DataFrame) and not df_sel.empty and {"ISIN", "Poids"}.issubset(df_sel.columns):
        w_ser = df_sel.set_index("ISIN")["Poids"].astype(float)
        w_ser = w_ser.replace([np.inf, -np.inf], np.nan).dropna()
        w_ser = w_ser[w_ser > 0]
        w_ser = w_ser / (float(w_ser.sum()) + 1e-12)

        nm_col = "Nom" if "Nom" in df_sel.columns else ("Nom du Fonds" if "Nom du Fonds" in df_sel.columns else None)
        if nm_col is None:
            name_map = dict(zip(df_sel["ISIN"].apply(_norm_key), df_sel["ISIN"].astype(str)))
        else:
            name_map = dict(zip(df_sel["ISIN"].apply(_norm_key), df_sel[nm_col].astype(str)))
    else:
        w_df_csv = pd.read_csv(CSV_PTF_PATH)
        if not {"ISIN", "Poids"}.issubset(w_df_csv.columns):
            raise ValueError("portefeuille_equilibre.csv doit contenir au moins 'ISIN' et 'Poids'.")
        w_ser = w_df_csv.set_index("ISIN")["Poids"].astype(float)
        w_ser = w_ser.replace([np.inf, -np.inf], np.nan).dropna()
        w_ser = w_ser[w_ser > 0]
        w_ser = w_ser / (float(w_ser.sum()) + 1e-12)

        nm_col = "Nom" if "Nom" in w_df_csv.columns else ("Nom du Fonds" if "Nom du Fonds" in w_df_csv.columns else None)
        if nm_col is None:
            name_map = dict(zip(w_df_csv["ISIN"].apply(_norm_key), w_df_csv["ISIN"].astype(str)))
        else:
            name_map = dict(zip(w_df_csv["ISIN"].apply(_norm_key), w_df_csv[nm_col].astype(str)))

w_ser = w_ser.reindex(asset_ret.columns).fillna(0.0)
if float(w_ser.sum()) <= 0:
    st.error("Poids invalides (somme = 0).")
    st.stop()
w_ser = w_ser / (float(w_ser.sum()) + 1e-12)

res = compute_ex_ante_risk(asset_ret, w_ser)

vol_ex_ante = float(res["vol_ex_ante"])
corr_mat = res["corr"]
avg_corr = float(corr_mat.values[np.triu_indices(corr_mat.shape[0], k=1)].mean()) if isinstance(corr_mat, pd.DataFrame) and corr_mat.shape[0] >= 2 else np.nan

mu_vec = (res["returns_final"].mean(axis=0) * 252.0)
w_vec = res["weights_final"]
mu_ex_ante = float(mu_vec.reindex(w_vec.index).fillna(0.0).dot(w_vec))

ret_pf_hist = pf_v.pct_change(fill_method=None).dropna().astype(float)
ret_bm_hist = bm_v.pct_change(fill_method=None).dropna().astype(float)

lambda_ewma = 0.94
if ret_pf_hist.empty:
    sigma_next_day = np.nan
else:
    v_t = float(ret_pf_hist.var())
    for r in ret_pf_hist.values:
        v_t = lambda_ewma * v_t + (1.0 - lambda_ewma) * float(r * r)
    sigma_next_day = float(np.sqrt(max(v_t, 0.0)))

confidence_level = 0.95
z_score = float(norm.ppf(1 - confidence_level))
var_95_1d = float(abs(z_score) * sigma_next_day) if np.isfinite(sigma_next_day) else np.nan
cvar_95_1d = float(abs(sigma_next_day * (norm.pdf(z_score) / (1 - confidence_level)))) if np.isfinite(sigma_next_day) else np.nan

common = ret_pf_hist.index.intersection(ret_bm_hist.index)
if len(common) < 10:
    beta_val = np.nan
else:
    x = ret_bm_hist.loc[common].values.astype(float)
    y = ret_pf_hist.loc[common].values.astype(float)
    var_x = float(np.var(x))
    beta_val = float(np.cov(y, x, ddof=0)[0, 1] / var_x) if np.isfinite(var_x) and var_x > 0 else np.nan

st.markdown("<h1 style='color: #FF4B4B;'>Risk Management Ex-Ante</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
<div style='background:rgba(138,43,226,0.1); border-left:5px solid #8a2be2; padding:15px; border-radius:8px;'>
  <b>Volatilité Ex-Ante (Annuelle)</b><br>
  <span style='font-size:24px; font-weight:bold; color:#b58bff;'>{vol_ex_ante*100:.2f}%</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"\sigma_p = \sqrt{W^T \Omega W}")

with col2:
    st.markdown(
        f"""
<div style='background:rgba(255,75,75,0.1); border-left:5px solid #FF4B4B; padding:15px; border-radius:8px;'>
  <b>Espérance de Rendement (1 an)</b><br>
  <span style='font-size:24px; font-weight:bold; color:#FF4B4B;'>{mu_ex_ante*100:.2f}%</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"E[R_p] = \sum_i w_i \mu_i")

with col3:
    st.markdown(
        f"""
<div style='background:rgba(0,123,255,0.1); border-left:5px solid #2a76ff; padding:15px; border-radius:8px;'>
  <b>Corrélation Moyenne Ex-Ante</b><br>
  <span style='font-size:24px; font-weight:bold; color:#2a76ff;'>{avg_corr:.2f}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"\bar{\rho} = \frac{2}{N(N-1)} \sum_{i<j} \rho_{ij}")

st.divider()

st.subheader("Projection de la VL à 5 ans (Mouvement Brownien Géométrique)")

horizon_days = 252 * 5
dt = 1 / 252
S0 = 1_000_000.0

if not np.isfinite(mu_ex_ante) or not np.isfinite(vol_ex_ante) or vol_ex_ante <= 0:
    st.warning("mu_ex_ante / vol_ex_ante indisponibles : projection GBM non calculable.")
else:
    n_sims = 1000
    Z_sim = np.random.standard_normal((horizon_days, n_sims))
    growth = np.exp((mu_ex_ante - 0.5 * vol_ex_ante**2) * dt + vol_ex_ante * np.sqrt(dt) * Z_sim)

    paths = np.empty((horizon_days + 1, n_sims), dtype=float)
    paths[0, :] = S0
    paths[1:, :] = S0 * np.cumprod(growth, axis=0)

    perc = np.percentile(paths, [5, 50, 95], axis=1)
    future_dates = pd.bdate_range(pf_v.index[-1], periods=horizon_days + 1)

    col_stat, col_proj = st.columns([1, 2.5], gap="large")

    with col_stat:
        st.markdown("#### Scénarios (Simulation)")

        df_scen = pd.DataFrame(
            {
                "Scénario": [
                    "<span class='badge badge-purple'>Favorable</span> (95%)",
                    "<span class='badge badge-blue'>Moyenne</span> E(S)",
                    "<span class='badge badge-green'>Médiane</span> (50%)",
                    "<span class='badge badge-red'>Stress</span> (5%)",
                ],
                "Valeur (€)": [
                    f"{perc[2, -1]:,.0f} €".replace(",", " "),
                    f"{paths[-1, :].mean():,.0f} €".replace(",", " "),
                    f"{perc[1, -1]:,.0f} €".replace(",", " "),
                    f"{perc[0, -1]:,.0f} €".replace(",", " "),
                ],
            }
        )

        html = df_scen.to_html(index=False, escape=False, classes="pretty-table")
        html = html.replace("<th>Valeur (€)</th>", "<th class='td-right'>Valeur (€)</th>")
        html = html.replace("<td>Valeur (€)</td>", "<td class='td-right'>Valeur (€)</td>")
        st.markdown(f"<div class='pretty-table-wrap'>{html}</div>", unsafe_allow_html=True)

        st.latex(r"S_t = S_0 e^{(\mu - \frac{1}{2}\sigma^2)t + \sigma W_t}")

    with col_proj:
        fig_cone = go.Figure()
        fig_cone.add_trace(go.Scatter(x=future_dates, y=perc[2], name="Haut (95%)"))
        fig_cone.add_trace(go.Scatter(x=future_dates, y=perc[0], fill="tonexty", name="Bas (5%)"))
        fig_cone.add_trace(go.Scatter(x=future_dates, y=perc[1], name="Médiane", line=dict(dash="dash", width=3)))
        fig_cone.update_layout(
            height=400,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cone, use_container_width=True)

st.divider()

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(
        f"""
<div style='background:rgba(255,165,0,0.1); border-left:5px solid #ffa500; padding:15px; border-radius:8px;'>
  <b>VaR Ex-Ante 95% (1 jour)</b><br>
  <span style='font-size:24px; font-weight:bold; color:#ffa500;'>{(var_95_1d*100 if np.isfinite(var_95_1d) else np.nan):.2f}%</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"VaR_{\alpha} = \sigma_{t+1} \cdot \Phi^{-1}(1-\alpha)")

with col5:
    st.markdown(
        f"""
<div style='background:rgba(255,0,0,0.1); border-left:5px solid #ff0000; padding:15px; border-radius:8px;'>
  <b>CVaR Ex-Ante 95% (1 jour)</b><br>
  <span style='font-size:24px; font-weight:bold; color:#ff0000;'>{(cvar_95_1d*100 if np.isfinite(cvar_95_1d) else np.nan):.2f}%</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"CVaR_{\alpha} = \sigma_{t+1}\frac{\phi(z_{\alpha})}{1-\alpha}")

with col6:
    st.markdown(
        f"""
<div style='background:rgba(40,167,69,0.1); border-left:5px solid #28a745; padding:15px; border-radius:8px;'>
  <b>Bêta (Ex-post vs Bench)</b><br>
  <span style='font-size:24px; font-weight:bold; color:#28a745;'>{(beta_val if np.isfinite(beta_val) else np.nan):.2f}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.latex(r"\beta_p = \frac{\text{Cov}(R_p,R_b)}{\sigma_b^2}")

st.divider()

df_risk = (
    pd.DataFrame(
        {
            "Nom": [name_map.get(_norm_key(i), str(i)) for i in res["weights_final"].index],
            "Poids": (res["weights_final"].values * 100.0),
            "CTR": (res["risk_budget_pct"].values),
        }
    )
    .sort_values("CTR", ascending=False)
    .reset_index(drop=True)
)

c_bar, c_corr = st.columns([1.45, 1.05], gap="large")

with c_bar:
    st.markdown("#### Budget de risque (CTR)")
    fig_ctr = go.Figure()
    fig_ctr.add_trace(go.Bar(y=df_risk["Nom"], x=df_risk["CTR"], orientation="h", name="CTR"))
    fig_ctr.update_layout(
        height=520,
        template="plotly_dark",
        yaxis={"autorange": "reversed"},
        xaxis_title="Contribution à la volatilité (%)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_ctr, use_container_width=True)

with c_corr:
    st.markdown("#### Analyse de la Contribution au Risque")

    if not df_risk.empty:
        top = df_risk.iloc[0]
        top3 = float(df_risk.iloc[:3]["CTR"].sum()) if len(df_risk) >= 3 else float(df_risk["CTR"].sum())
        st.info(f"Le principal risque est porté par **{top['Nom']}** (**{top['CTR']:.1f}%** du risque total).")
        st.write(f"Les **3 actifs les plus risqués** concentrent **{top3:.1f}%** de la volatilité du portefeuille.")
    else:
        st.warning("Impossible d'analyser la contribution au risque.")

    st.write("---")
    st.latex(r"CTR_i = w_i \frac{\partial \sigma_p}{\partial w_i}")
    st.markdown(
        "Une contribution élevée indique que l'actif domine le profil de risque "
        "du portefeuille, même avec un poids relativement limité."
    )