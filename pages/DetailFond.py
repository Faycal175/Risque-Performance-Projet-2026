import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Stratégie Équilibrée", layout="wide")

nav_left, _ = st.columns([3, 1.8])
with nav_left:
    if st.button("←", help="Page précédente"):
        st.switch_page("pages/choix.py")

pname = st.session_state.get("selected_profile", "Équilibré")
df = st.session_state.get("selected_portfolio_df", pd.DataFrame())

st.markdown(
    """
<style>
.main-title { font-size: 38px; font-weight: 800; color: #2a76ff; margin-bottom: 0px; }
.st-emotion-cache-1y4p8pa { padding-top: 2rem; }

.ucits-box{
  background: rgba(40, 167, 69, 0.1);
  border-left: 5px solid #28a745;
  padding: 15px;
  border-radius: 5px;
  margin: 15px 0;
}

.fund-name { color: #2a76ff; font-weight: 700; font-size: 15px; }

.risk-badge { background: #007bff; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;}
.esg-badge { background: #28a745; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;}
.hybrid-badge { background: #fd7e14; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;}
.social-badge { background: #e83e8c; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;}
.conv-badge { background: #6f42c1; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;}
.style-badge { background: #6c757d; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;}
.quant-badge { background: #17a2b8; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;}
.flex-badge { background: #ffc107; color: black; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;}
.def-badge { background: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;}
.gold-badge { background: #ffc107; color: black; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;}
.em-badge { background: #e83e8c; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-right: 5px;}

.custom-table-wrap{
  border: 1px solid rgba(42,118,255,0.25);
  border-radius: 12px;
  overflow: hidden;
  background: rgba(255,255,255,0.02);
}

table.custom-table{
  width:100%;
  border-collapse: collapse;
  font-size: 14px;
}

table.custom-table thead th{
  background:#2a76ff;
  color:white;
  padding: 10px 12px;
  text-align:left;
  font-weight: 700;
  letter-spacing: .2px;
}

table.custom-table tbody td{
  padding: 9px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  vertical-align: top;
}

table.custom-table tbody tr:nth-child(even){
  background: rgba(255,255,255,0.03);
}

table.custom-table tbody tr:hover{
  background: rgba(42,118,255,0.10);
}

.td-nom{
  font-weight: 650;
  color:#2a76ff;
}

.td-poids{
  text-align: right;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.td-isin{
  white-space: nowrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 13px;
  opacity: .95;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(f"<div class='main-title'>Profil {pname} (19 Fonds)</div>", unsafe_allow_html=True)
st.markdown("---")

col_desc, col_pie = st.columns([1.5, 1])

with col_desc:
    st.subheader("Ingénierie du Portefeuille (50% / 50%)")
    st.markdown(
        """
Ce portefeuille est structuré autour d'une allocation paritaire, optimisée pour le couple rendement/risque.
Il repose sur une architecture Core-Satellite multi-moteurs testée contre des chocs macro-économiques.
"""
    )
    st.markdown(
        """
<div class='ucits-box'>
<b>Maîtrise de la règle UCITS 5/10/40 :</b><br>
7 fonds Actions Core pondérés à <b>5,71%</b> (total <b>39,97%</b>).
Les autres lignes sont à 5% ou 3,33% pour limiter la concentration.
</div>
""",
        unsafe_allow_html=True,
    )

with col_pie:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Obligataire (50%)", "Actions (50%)"],
                values=[50, 50],
                hole=0.4,
                marker_colors=["#2a76ff", "#1a1a1a"],
            )
        ]
    )
    fig.update_layout(height=280, margin=dict(l=0, r=0, b=0, t=0), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Analyse micro-économique des 3 poches")

tab_compo, tab_obli, tab_core, tab_def = st.tabs(
    ["Composition du Portefeuille", "Poche Obligataire (50%)", "Poche Actions Core (40%)", "Poche Actions Défense/Refuge (10%)"]
)

with tab_compo:
    st.markdown("### Répartition Globale des Actifs")

    if df.empty:
        st.warning("Composition introuvable (session_state vide). Veuillez sélectionner un profil depuis la page Choix.")
    else:
        df_show = df.copy()

        if "Poids" in df_show.columns:
            df_show["Poids"] = (pd.to_numeric(df_show["Poids"], errors="coerce") * 100).map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")

        cols = ["Bucket", "Nom", "ISIN", "Poids"]
        df_table = df_show.reindex(columns=cols).fillna("").copy()

        html = """
<div class="custom-table-wrap">
<table class="custom-table">
<thead>
<tr>
<th>Classe d'Actif</th>
<th>Nom du Fonds</th>
<th>Code ISIN</th>
<th style="text-align:right;">Pondération</th>
</tr>
</thead>
<tbody>
"""
        for _, r in df_table.iterrows():
            html += f"""
<tr>
<td>{str(r["Bucket"])}</td>
<td class="td-nom">{str(r["Nom"])}</td>
<td class="td-isin">{str(r["ISIN"])}</td>
<td class="td-poids">{str(r["Poids"])}</td>
</tr>
"""
        html += "</tbody></table></div>"
        st.markdown(html, unsafe_allow_html=True)

with tab_obli:
    st.markdown('### Ingénierie de la Poche Obligataire : approche "All-Weather"')
    st.markdown(
        """
<div style='background: rgba(42, 118, 255, 0.1); border-left: 4px solid #2a76ff; padding: 15px; margin-bottom: 20px;'>
<b>Synergie :</b><br>
Construction type "Barbell" : socle souverain liquide + stratégies flexibles / inflation.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
* <span class='risk-badge'>SRI 2</span> <span class='fund-name'>Socle de liquidité : BNPP Euro Bond & Amundi Global Agg</span><br>
* <span class='risk-badge'>SRI 3</span> <span class='fund-name'>Amortisseur de taux : Carmignac Pf Global Bond</span><br>
* <span class='risk-badge'>SRI 2</span> <span class='style-badge'>Couverture macro</span> <span class='fund-name'>BNPP Gb Infla-Linked</span><br>
* <span class='risk-badge'>SRI 2</span> <span class='esg-badge'>Article 9 SFDR</span> <span class='fund-name'>Prime ESG : Mirova Green Bond & Amundi Social</span>
""",
        unsafe_allow_html=True,
    )

with tab_core:
    st.markdown("### Moteur Actions Core : diversification géographique et styles")
    st.markdown(
        """
<div style='background: rgba(255, 165, 0, 0.1); border-left: 4px solid #FFA500; padding: 15px; margin-bottom: 20px;'>
<b>Synergie :</b><br>
Coeur indiciel à bas coût + satellites actifs / quant pour diversifier styles et zones.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
* <span class='style-badge'>Coûts</span> <span class='fund-name'>ETF : Amundi S&P 500 Screened & MSCI Europe</span><br>
* <span class='style-badge'>Stock-picking</span> <span class='fund-name'>Echiquier Major SRI Growth Europe</span><br>
* <span class='quant-badge'>Quant</span> <span class='fund-name'>Robeco QI Global 3D</span><br>
* <span class='em-badge'>Emergents</span> <span class='fund-name'>JPMF Emerging Markets Equity</span><br>
* <span class='flex-badge'>Allocation</span> <span class='fund-name'>BGF Global Allocation</span>
""",
        unsafe_allow_html=True,
    )

with tab_def:
    st.markdown("### Satellite Défense & Valeur Refuge : couverture des risques extrêmes")
    st.markdown(
        """
<div style='background: rgba(138, 43, 226, 0.1); border-left: 4px solid #8a2be2; padding: 15px; margin-bottom: 20px;'>
<b>Synergie :</b><br>
Sécurité civile / numérique + exposition à l'or comme assurance macro.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
* <span class='def-badge'>Sécurité</span> <span class='fund-name'>Pictet Security & NIF Lux Safety</span><br>
* <span class='gold-badge'>Or</span> <span class='fund-name'>EdR Goldsphere</span>
""",
        unsafe_allow_html=True,
    )

st.divider()

btn_left, btn_right = st.columns([1, 1], gap="large")

with btn_left:
    if st.button("Analyse rétrospective", type="primary", use_container_width=True):
        st.switch_page("pages/Dash_equi.py")

with btn_right:
    if st.button("Analyse prospective", type="primary", use_container_width=True):
        st.switch_page("pages/ExAnte.py")