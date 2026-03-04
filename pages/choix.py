import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gamme — Portefeuilles modèles", layout="wide")


nav_left, nav_right = st.columns([1, 5])
with nav_left:
    if st.button("⬅ Retour", help="Page précédente"):
        st.switch_page("app.py")


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

CSV_PATH = os.path.join(PROJECT_ROOT, "Projet", "portefeuille_equilibre.csv")


st.markdown(
    """
<style>
.block-container { max-width: 100% !important; padding-left: 1.2rem; padding-right: 1.2rem; }

.hero{
  border: 1px solid rgba(255,255,255,0.12);
  background: radial-gradient(1200px 380px at 20% 10%, rgba(42,118,255,0.35), rgba(0,0,0,0) 60%),
              radial-gradient(900px 320px at 80% 20%, rgba(255,215,0,0.20), rgba(0,0,0,0) 55%),
              linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  border-radius: 22px;
  padding: 18px 18px 14px 18px;
  margin-top: 8px;
  position: relative;
  overflow: hidden;
}

.hero h1{
  margin: 0;
  font-size: 34px;
  font-weight: 950;
  letter-spacing: .2px;
}
.hero p{
  margin: 6px 0 0 0;
  opacity: .85;
  font-size: 14.5px;
  line-height: 1.6;
}

.section-title{
  margin-top: 8px;
  font-size: 20px;
  font-weight: 950;
}

.panel{
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.04);
}

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

.card{
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
  background: rgba(255,255,255,0.04);
  width: 100%;
  margin-top: 6px;
  position: relative;
  min-height: 120px;
}

.card-gold{
  border: 2px solid #FFD700 !important;
  background: linear-gradient(180deg, rgba(255, 215, 0, 0.10) 0%, rgba(255,255,255,0.02) 100%) !important;
  box-shadow: 0 6px 22px rgba(255, 215, 0, 0.16);
}

.badge-gold {
  background-color: #FFD700;
  color: #111;
  font-size: 12px;
  font-weight: 950;
  text-transform: uppercase;
  padding: 4px 12px;
  border-radius: 20px;
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%);
  letter-spacing: 0.6px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.55);
}

.card-title-center{
  width:100%;
  text-align:center;
  font-size: 24px;
  font-weight: 950;
  margin: 8px 0 8px 0;
}
.small{
  color: rgba(255,255,255,0.78);
  font-size: 13.5px;
  margin: 0;
  text-align:center;
}

.pill{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 950;
  margin-right: 10px;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(42,118,255,0.14);
  color: rgba(255,255,255,0.92);
}

.def-pill{
  background: rgba(220,53,69,0.16);
  border-color: rgba(220,53,69,0.35);
  color: rgba(255,255,255,0.94);
}

.note{
  opacity:.78;
  font-size: 13px;
  line-height: 1.55;
}

.hr{
  height: 1px;
  background: rgba(255,255,255,0.10);
  margin: 12px 0;
}

.kpi-wrap{
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  overflow: hidden;
  background: rgba(255,255,255,0.04);
}
.kpi-head{
  padding: 12px 14px;
  border-bottom: 1px solid rgba(255,255,255,0.10);
  font-size: 16px;
  font-weight: 950;
}
table.kpi-table{
  width: 100%;
  border-collapse: collapse;
  font-size: 14.5px;
}
table.kpi-table td{
  padding: 10px 14px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  vertical-align: top;
  opacity: .92;
}
table.kpi-table td:first-child{
  width: 58%;
  opacity: .78;
  font-weight: 850;
}
table.kpi-table tr:last-child td{ border-bottom: none; }
.kpi-strong{ font-weight: 950; opacity: 1; }

.stButton { margin-top: -8px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_portfolio_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig").copy()
    except Exception:
        df = pd.read_csv(csv_path).copy()

    # normalise noms de colonnes
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if "isin" in cl:
            col_map[c] = "ISIN"
        elif cl == "poids" or "poids" in cl or "weight" in cl:
            col_map[c] = "Poids"
        elif cl == "nom" or "fonds" in cl:
            col_map[c] = "Nom"
        elif cl == "bucket" or "poche" in cl or "classe" in cl:
            col_map[c] = "Bucket"
    df = df.rename(columns=col_map).copy()

    if not {"ISIN", "Poids"}.issubset(df.columns):
        raise ValueError("Le CSV doit contenir au minimum les colonnes 'ISIN' et 'Poids'.")

    df["ISIN"] = df["ISIN"].astype(str).str.strip()
    df["Poids"] = pd.to_numeric(df["Poids"], errors="coerce")
    df = df.dropna(subset=["ISIN", "Poids"]).copy()
    df = df[df["ISIN"].ne("") & (df["Poids"] > 0)].copy()

    # si poids en % (ex 10, 12.5...), convertit en fraction
    if len(df) > 0 and float(df["Poids"].max()) > 1.0 + 1e-12:
        df["Poids"] = df["Poids"] / 100.0

    # normalisation somme=1
    s = float(df["Poids"].sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Somme des poids invalide dans le CSV.")
    df["Poids"] = df["Poids"] / (s + 1e-12)

    # colonnes optionnelles
    if "Nom" not in df.columns:
        df["Nom"] = df["ISIN"]
    if "Bucket" not in df.columns:
        df["Bucket"] = ""

    # dédoublonne (au cas où)
    df = df.groupby(["ISIN"], as_index=False).agg({"Poids": "sum", "Nom": "first", "Bucket": "first"})
    df["Poids"] = df["Poids"] / (float(df["Poids"].sum()) + 1e-12)

    return df


try:
    df_eq = load_portfolio_csv(CSV_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

st.markdown(
    """
<div class="hero">
  <h1>Gamme de produits — Portefeuilles modèles</h1>
  <p>
    Quatre profils de risque, une allocation stratégique lisible, et une signature “Défense intégrée”
    pour renforcer la résilience en période de stress.
  </p>
</div>
""",
    unsafe_allow_html=True,
)


# INPUTS (MACRO DEMO)

PROFILES_MACRO = {
    "Sécurité": {"Monétaire": 0.60, "Obligataire": 0.20, "Actions": 0.20},
    "Défensif": {"Monétaire": 0.20, "Obligataire": 0.40, "Actions": 0.40},
    "Équilibré": {"Monétaire": 0.00, "Obligataire": 0.50, "Actions": 0.50},
    "Agressif": {"Monétaire": 0.00, "Obligataire": 0.20, "Actions": 0.80},
}
profiles_order = ["Sécurité", "Défensif", "Équilibré", "Agressif"]

# ✅ uniquement depuis le CSV
n_lines = int((df_eq["Poids"] > 0).sum())

fee_portfolio_mgmt = 0.35
fee_platform = 0.20
fee_underlying_est = 0.80
fee_all_in = fee_portfolio_mgmt + fee_platform + fee_underlying_est


# TOP: STACKED + KPI / STORY

top_left, top_right = st.columns([1.25, 1], gap="large")

with top_left:
    st.markdown("<div class='section-title'>Répartition stratégique par profil</div>", unsafe_allow_html=True)

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
        height=340,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="%", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
<div class="panel note" style="margin-top:10px;">
  <b>Lecture rapide :</b> le niveau d’actions augmente avec le profil (20% → 80%).
  Le profil <b>Équilibré</b> combine un potentiel de performance et un amortisseur obligataire,
  et c’est celui que l’on privilégie comme allocation “cœur”.
</div>
""",
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown("<div class='section-title'>Pilotage & cadre</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="kpi-wrap">
  <div class="kpi-head">Synthèse</div>
  <table class="kpi-table">
    <tr><td>Profil recommandé</td><td class="kpi-strong">Équilibré (50% / 50%)</td></tr>
    <tr><td>Nombre de lignes (CSV)</td><td class="kpi-strong">{n_lines}</td></tr>
    <tr><td>Frais de pilotage (indicatif)</td><td class="kpi-strong">{fee_portfolio_mgmt:.2f}% / an</td></tr>
    <tr><td>Frais d’enveloppe (indicatif)</td><td class="kpi-strong">{fee_platform:.2f}% / an</td></tr>
    <tr><td>Frais des supports (indicatif)</td><td class="kpi-strong">{fee_underlying_est:.2f}% / an</td></tr>
    <tr><td>Total (ordre de grandeur)</td><td class="kpi-strong">{fee_all_in:.2f}% / an</td></tr>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="panel">
  <h4 style="margin:0 0 6px 0; font-size:16px; font-weight:950;">
    <span class="pill def-pill">défense</span>Signature “Défense intégrée”
  </h4>
  <div class="note">
    Une poche thématique dédiée à la résilience (sécurité, cyber, infrastructures critiques)
    complétée par une exposition “refuge” (or). L’objectif est de mieux tenir les phases de stress,
    sans dénaturer l’allocation stratégique du profil.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.divider()





st.markdown("<div class='section-title'>Choisir un profil</div>", unsafe_allow_html=True)

cols = st.columns(4, gap="large")
profiles_display = ["Sécurité", "Défensif", "Équilibré", "Agressif"]

def subtitle_for_profile(p: str) -> str:
    if p == "Sécurité":
        return "Priorité stabilité • faible exposition actions"
    if p == "Défensif":
        return "Risque modéré • amortisseur renforcé"
    if p == "Équilibré":
        return "Cœur de gamme • Défense intégrée"
    return "Recherche performance • exposition actions élevée"

for col, pname in zip(cols, profiles_display):
    with col:
        is_equilibre = pname == "Équilibré"
        

        if is_equilibre:
            card_class = "card card-gold"
            badge_html = '<div class="badge-gold">Recommandé</div>'
        else:
            card_class = "card"
            badge_html = ""

        st.markdown(
            f"""
<div style="position:relative;">
<div class="{card_class}">
{badge_html}
<div class="card-title-center">{pname}</div>
<div class="small">{subtitle_for_profile(pname)}</div>

</div>
</div>
""",
            unsafe_allow_html=True,
        )

        if is_equilibre:
            if st.button("Ouvrir la présentation", key="open_equilibre", type="primary", use_container_width=True):
                # ✅ on passe EXACTEMENT la compo du CSV (avec vrais poids)
                st.session_state["selected_profile"] = "Équilibré"
                st.session_state["selected_portfolio_df"] = df_eq.copy()
                st.switch_page("pages/DetailFond.py")
        else:
            st.markdown("<div style='height: 42px;'></div>", unsafe_allow_html=True)

st.caption(
    "Cette page présente la gamme et le cadre de pilotage. "
    "La composition détaillée et les analyses sont accessibles via la page de présentation du portefeuille."
)