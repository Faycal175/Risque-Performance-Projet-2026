import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Profil de risque", layout="wide")

nav_left, _ = st.columns([3, 1.8])
with nav_left:
    if st.button("←", help="Page précédente"):
        st.switch_page("app.py")

st.title("Gamme de produits — Portefeuilles modèles")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Calculs.ptf_equilibre_engine import EQUILIBRE_MACRO, build_equilibre_portfolio, ucits_51040_check  # noqa: E402

st.markdown(
    """
<style>
.block-container { max-width: 100% !important; padding-left: 1.2rem; padding-right: 1.2rem; }

.card{
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  padding: 18px 18px 12px 18px;
  background: rgba(255,255,255,0.04);
  width: 100%;
  margin-top: 15px;
  position: relative;
}

.card-gold{
  border: 2px solid #FFD700 !important;
  background: linear-gradient(180deg, rgba(255, 215, 0, 0.08) 0%, rgba(255,255,255,0.02) 100%) !important;
  box-shadow: 0 4px 20px rgba(255, 215, 0, 0.15);
}

.badge-gold {
  background-color: #FFD700;
  color: #111;
  font-size: 11px;
  font-weight: 900;
  text-transform: uppercase;
  padding: 4px 12px;
  border-radius: 20px;
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%);
  letter-spacing: 0.5px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.5);
}

.card-title-center{
  width:100%;
  text-align:center;
  font-size: 22px;
  font-weight: 800;
  margin: 10px 0 10px 0;
}

.small{
  color: rgba(255,255,255,0.70);
  font-size: 12.5px;
  margin: 2px 0 10px 0;
}

.info-box{
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  padding: 18px;
  background: rgba(255,255,255,0.04);
}
.info-box h3{ margin:0 0 8px 0; font-size:18px; font-weight:800; }
.info-box p{ margin:0 0 10px 0; color: rgba(255,255,255,0.85); line-height: 1.55; font-size: 14px; }
.info-box .muted{ color:#9aa0a6; font-size:12px; }

.stButton { margin-top: -10px; }
</style>
""",
    unsafe_allow_html=True,
)

UNIVERSE_PATH = os.path.join(PROJECT_ROOT, "Projet", "UniversInvestissement.xlsb")
SAVE_PATH = os.path.join(PROJECT_ROOT, "Projet", "portefeuille_equilibre.csv")


@st.cache_data
def load_universe(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, engine="pyxlsb")

    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "nom":
            col_map[c] = "Nom"
        elif "cat" in cl:
            col_map[c] = "Categorie"
        elif "isin" in cl:
            col_map[c] = "ISIN"
        elif "soci" in cl and "gest" in cl:
            col_map[c] = "Societe"

    df = df.rename(columns=col_map).copy()

    if "Societe" not in df.columns:
        df["Societe"] = ""

    for c in ["Nom", "ISIN", "Societe"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df = df[df["Nom"].ne("") & df["ISIN"].ne("")].drop_duplicates(subset=["ISIN"])
    return df


uni = load_universe(UNIVERSE_PATH)
df_eq = build_equilibre_portfolio(uni)

if df_eq.empty:
    ucits_status = "Sélection vide"
else:
    ok, max_w, over5 = ucits_51040_check(df_eq.loc[df_eq["Poids"] > 0, "Poids"])
    ucits_status = "UCITS 5/10/40 OK" if ok else f"UCITS (max={max_w*100:.1f}%, >5%={over5*100:.1f}%)"

if not df_eq.empty:
    df_eq.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")


top_left, top_right = st.columns([1.25, 1], gap="large")

with top_left:
    PROFILES_MACRO = {
        "Sécurité": {"Monétaire": 0.60, "Obligataire": 0.20, "Actions": 0.20},
        "Défensif": {"Monétaire": 0.20, "Obligataire": 0.40, "Actions": 0.40},
        "Équilibré": {"Monétaire": 0.00, "Obligataire": 0.50, "Actions": 0.50},
        "Agressif": {"Monétaire": 0.00, "Obligataire": 0.20, "Actions": 0.80},
    }

    profiles_order = ["Sécurité", "Défensif", "Équilibré", "Agressif"]
    df_alloc = pd.DataFrame(
        {
            "Profil": profiles_order,
            "Monétaire": [PROFILES_MACRO[p]["Monétaire"] * 100 for p in profiles_order],
            "Obligataire": [PROFILES_MACRO[p]["Obligataire"] * 100 for p in profiles_order],
            "Actions": [PROFILES_MACRO[p]["Actions"] * 100 for p in profiles_order],
        }
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Monétaire", x=df_alloc["Profil"], y=df_alloc["Monétaire"]))
    fig.add_trace(go.Bar(name="Obligataire", x=df_alloc["Profil"], y=df_alloc["Obligataire"]))
    fig.add_trace(go.Bar(name="Actions", x=df_alloc["Profil"], y=df_alloc["Actions"]))

    fig.update_layout(
        barmode="stack",
        height=320,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Répartition par classe d’actifs selon le profil",
        yaxis=dict(title="%", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

with top_right:
    st.markdown(
        f"""
<div class="info-box">
<h3>Logique de pondération</h3>
<p>
Gamme indicative avec <b>Défense intégrée</b> dans la poche Actions.
Le portefeuille recommandé est <b>Équilibré</b> (allocation macro 50/50).
</p>
<div class="muted">
{ucits_status} • Export CSV auto : portefeuille_equilibre.csv
</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.divider()

cols = st.columns(4, gap="large")
profiles_display = ["Sécurité", "Défensif", "Équilibré", "Agressif"]

for col, pname in zip(cols, profiles_display):
    with col:
        is_equilibre = pname == "Équilibré"

        if is_equilibre:
            card_class = "card card-gold"
            badge_html = '<div class="badge-gold">Recommandé</div>'
            status = ucits_status
        else:
            card_class = "card"
            badge_html = ""
            status = "Profil indicatif"

        st.markdown(
            f"""
<div style="position:relative;">
<div class="{card_class}">
{badge_html}
<div class="card-title-center">{pname}</div>
<div class="small" style="text-align:center;">{status} • Défense intégrée</div>
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        if is_equilibre:
            if st.button("Ouvrir la présentation", key="open_equilibre", type="primary", use_container_width=True):
                st.session_state["selected_profile"] = "Équilibré"
                st.session_state["selected_portfolio_df"] = df_eq.copy()
                st.switch_page("pages/DetailFond.py")
        else:
            st.markdown("<div style='height: 42px;'></div>", unsafe_allow_html=True)

st.caption(
    "Note: cette page présente uniquement la gamme (esthétique). "
    "La composition détaillée du portefeuille est visible uniquement dans la page de présentation."
)