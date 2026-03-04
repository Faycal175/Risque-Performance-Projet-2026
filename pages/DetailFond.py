# pages/DetailFond.py 


from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Stratégie Équilibrée", layout="wide")


# =========================================================
# NAV
# =========================================================
nav_left, nav_right = st.columns([3, 1.8])

with nav_left:
    st.markdown("<div style='margin-top:50px'></div>", unsafe_allow_html=True)

    if st.button("←", help="Page précédente"):
        st.switch_page("pages/choix.py")

# =========================================================
# PATHS
# =========================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CSV_PATH = PROJECT_ROOT / "Projet" / "portefeuille_equilibre.csv"


# =========================================================
# HELPERS
# =========================================================
def load_portfolio_df_from_state_or_csv() -> tuple[str, pd.DataFrame]:
    pname = st.session_state.get("selected_profile", "Équilibré")
    df0 = st.session_state.get("selected_portfolio_df", None)

    if isinstance(df0, pd.DataFrame) and not df0.empty:
        return pname, df0.copy()

    if CSV_PATH.exists():
        try:
            df_csv = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        except Exception:
            df_csv = pd.read_csv(CSV_PATH)
        return pname, df_csv.copy()

    return pname, pd.DataFrame()


def normalize_portfolio_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Bucket", "Nom", "ISIN", "Poids"])

    out = df.copy()
    col_ren = {}
    for c in out.columns:
        cl = str(c).strip().lower()
        if cl == "bucket":
            col_ren[c] = "Bucket"
        elif cl == "nom":
            col_ren[c] = "Nom"
        elif "isin" in cl:
            col_ren[c] = "ISIN"
        elif cl == "poids" or "pond" in cl or "weight" in cl:
            col_ren[c] = "Poids"

    out = out.rename(columns=col_ren).copy()

    for c in ["Bucket", "Nom", "ISIN", "Poids"]:
        if c not in out.columns:
            out[c] = ""

    out["ISIN"] = out["ISIN"].astype(str).str.strip()
    out["Nom"] = out["Nom"].astype(str).str.strip()
    out["Bucket"] = out["Bucket"].astype(str).str.strip()
    out = out[out["ISIN"].ne("")].copy()
    return out[["Bucket", "Nom", "ISIN", "Poids"]].copy()


def format_weights_percent(series: pd.Series) -> pd.Series:
    """
    - Si max <= 1 => fractions -> *100
    - Sinon déjà en %
    ✅ évite le bug "100% partout"
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([""] * len(series), index=series.index)

    mx = float(s.max())
    if mx <= 1.0 + 1e-12:
        s = s * 100.0

    return s.map(lambda x: "" if pd.isna(x) else f"{float(x):.2f}%")


def bucket_totals_pct(df_norm: pd.DataFrame) -> dict[str, float]:
    """
    retourne {bucket: pct}
    """
    if df_norm.empty:
        return {}

    w = pd.to_numeric(df_norm["Poids"], errors="coerce").fillna(0.0)
    mx = float(w.max()) if len(w) else 0.0
    w_frac = w if mx <= 1.0 + 1e-12 else (w / 100.0)

    tmp = df_norm.copy()
    tmp["_w"] = w_frac.clip(lower=0.0)

    g = tmp.groupby("Bucket")["_w"].sum()
    return {str(k): float(v * 100.0) for k, v in g.items()}


# =========================================================
# LOAD DATA
# =========================================================
pname, df_raw = load_portfolio_df_from_state_or_csv()
df = normalize_portfolio_columns(df_raw)
n_funds = int(df["ISIN"].nunique()) if not df.empty else 0

bkt = bucket_totals_pct(df)
b_obl = bkt.get("Obligataire", 50.0)
b_act = 100.0 - b_obl if (0 <= b_obl <= 100) else 50.0
b_def = bkt.get("Actions - Défense", 10.0)  # indicatif
b_core = bkt.get("Actions - Core", max(0.0, b_act - b_def))  # indicatif


# =========================================================
# PARAMS (affichés) — alignés avec ton engine
# =========================================================
# (tu peux aussi les importer depuis Calculs.ptf_equilibre_engine si tu préfères)
MAX_LINE_WEIGHT = 0.10          # 10% max par ligne
CORE_MIN_PER_FUND = 0.02        # 2% min par fond core (quand possible)
DEF_MIN_W = 0.15                # 15% min de défense dans le portefeuille (dans ton engine actuel)
DEF_MAX_W = 0.20                # 20% max de défense
REBAL_FREQ = "Annuel (Y)"
SIGNAL = "Momentum 12M–1M"
MOM_TILT_STRENGTH = 0.30
DEF_TILT_STRENGTH = 0.60


# =========================================================
# CSS (style + orga)
# =========================================================
st.markdown(
    """
<style>
.block-container{ max-width: 1200px !important; padding-top: 1.1rem; }

.hero{
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 22px;
  padding: 18px 18px 14px 18px;
  background: radial-gradient(1200px 300px at 10% 0%, rgba(42,118,255,0.28), rgba(0,0,0,0) 60%),
              rgba(255,255,255,0.03);
  position: relative;
  overflow: hidden;
}

.hero-title{
  font-size: 40px;
  font-weight: 950;
  letter-spacing: .2px;
  margin: 0;
  color: #2a76ff;
}

.hero-sub{
  margin-top: 6px;
  color: rgba(255,255,255,0.80);
  font-size: 15.5px;
  line-height: 1.55;
}

.pills{ margin-top: 10px; display:flex; flex-wrap: wrap; gap: 8px; }
.pill{
  display:inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 900;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  color: rgba(255,255,255,0.90);
}

.grid-2{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }

.card{
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  padding: 14px 14px 12px 14px;
  background: rgba(255,255,255,0.04);
}

.card h4{
  margin: 0 0 8px 0;
  font-size: 16px;
  font-weight: 950;
}

.card p{
  margin: 0;
  color: rgba(255,255,255,0.86);
  font-size: 14.8px;
  line-height: 1.60;
}

.kpi-grid{
  display:grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
  margin-top: 12px;
}

.kpi{
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 12px 12px 10px 12px;
  background: rgba(255,255,255,0.03);
}

.kpi .t{
  color: rgba(255,255,255,0.70);
  font-size: 12px;
  font-weight: 900;
  margin-bottom: 4px;
}

.kpi .v{
  font-size: 18px;
  font-weight: 950;
  color: rgba(255,255,255,0.95);
}

.kpi .s{
  margin-top: 3px;
  color: rgba(255,255,255,0.70);
  font-size: 12.5px;
  line-height: 1.35;
}

.callout{
  border-left: 6px solid #FF4B4B;
  background: rgba(255,75,75,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  margin-top: 12px;
  color: rgba(255,255,255,0.90);
  font-size: 14.8px;
  line-height: 1.55;
}

.custom-table-wrap{
  border: 1px solid rgba(42,118,255,0.25);
  border-radius: 14px;
  overflow: hidden;
  background: rgba(255,255,255,0.02);
}

table.custom-table{
  width:100%;
  border-collapse: collapse;
  font-size: 14.2px;
}

table.custom-table thead th{
  background:#2a76ff;
  color:white;
  padding: 10px 12px;
  text-align:left;
  font-weight: 950;
  letter-spacing: .2px;
}

table.custom-table tbody td{
  padding: 10px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  vertical-align: top;
}

table.custom-table tbody tr:nth-child(even){
  background: rgba(255,255,255,0.03);
}

table.custom-table tbody tr:hover{
  background: rgba(42,118,255,0.10);
}

.td-nom{ font-weight: 850; color:#2a76ff; }
.td-poids{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
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


# =========================================================
# HERO
# =========================================================
st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">Profil {pname} • {n_funds} fonds</div>
  <div class="hero-sub">
    Allocation macro stable <b>50% Obligations / 50% Actions</b> + pilotage <b>Momentum</b> au sein des Actions,
    avec une poche <b>Core</b> et une poche <b>Défense/Refuge</b> (cyber/sécurité + or).
  </div>
  

</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")


# =========================================================
# TOP: DONUT + STRAT SUMMARY (bulles mieux organisées)
# =========================================================
st.subheader("Cadre Technique")

colA, colB = st.columns([1.25, 1], gap="large")

with colA:
    st.markdown(
        """
<div class="grid-2">
  <div class="card">
    <h4>1) Signal Momentum (12M–1M)</h4>
    <p>
      <b>On calcule pour chaque fonds une mesure de tendance :</b>
      <b>performance 12 mois</b> moins <b>performance 1 mois</b>.
      <br><br>
      Objectif : capter une dynamique persistante et éviter la partie “bruit” très court terme.
    </p>
  </div>

  <div class="card">
    <h4>2) Décision annuelle</h4>
    <p>
      Une fois par an, le signal est observé et sert à :
      <br>
      • <b>tilter</b> les poids <u>dans</u> Core et Défense<br>
      • <b>ajuster</b> le poids total de la poche Défense (borné).
    </p>
  </div>
</div>

<div class="callout">
  <b>Indicateur clé utilisé :</b> Momentum 12M–1M calculé sur rendements journaliers
</div>
""",
    unsafe_allow_html=True,
)

# ✅ formule en vrai LaTeX, en dessous
    st.latex(r"s_i(t)=\mathrm{Mom}_{12m}(t)-\mathrm{Mom}_{1m}(t)")
with colB:
    # donut macro (indicatif)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Obligataire", "Actions"],
                values=[50, 50],
                hole=0.46,
                marker_colors=["#2a76ff", "#1a1a1a"],
            )
        ]
    )

    fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, b=0, t=10),
        legend=dict(orientation="h", y=-0.2)
    )

    st.plotly_chart(fig, use_container_width=True)

    # KPIs compréhensibles investisseur
    st.markdown(
        """
<div class="kpi-grid">

<div class="kpi">
<div class="t">Diversification maximale</div>
<div class="v">10%</div>
<div class="s">aucune position ne dépasse 10% du portefeuille</div>
</div>

<div class="kpi">
<div class="t">Diversification actions</div>
<div class="v">≥ 2%</div>
<div class="s">chaque fonds actions core conserve un poids significatif</div>
</div>

<div class="kpi">
<div class="t">Poche Défense</div>
<div class="v">15% – 20%</div>
<div class="s">exposition dynamique aux actifs Defense/Sécurité </div>
</div>

</div>
""",
    unsafe_allow_html=True,
)

st.divider()


# =========================================================
# EXPLICATION STRATÉGIQUE
# =========================================================
st.subheader("Ce que cela signifie concrètement pour vous")

st.markdown(
"""
<div class="grid-2">

<div class="card">
<h4>Pilotage du portefeuille</h4>
<p>
Le portefeuille repose sur une structure simple :
<br><br>

• <b>50% d'obligations</b> pour stabiliser le portefeuille<br>
• <b>50% d'actions</b> pour capter la croissance des marchés<br><br>

Au sein des actions, deux moteurs sont utilisés :
<br>
• une poche <b>Core</b> investie dans les grandes tendances économiques<br>
• une poche <b>Défense / refuge</b> (sécurité, or) 
</p>
</div>

<div class="card">
<h4>Contrôle du risque</h4>
<p>
Plusieurs règles de gestion permettent d’éviter les concentrations excessives :
<br><br>

• <b>aucune ligne ne dépasse 10%</b> du portefeuille<br>
• les investissements sont répartis sur plusieurs fonds<br>
• l’exposition aux actifs défensifs reste <b>encadrée</b>
afin d’éviter un biais excessif.
<br><br>

Ces contraintes permettent de maintenir un portefeuille
<b>diversifié et robuste</b> dans différents environnements de marché.
</p>
</div>

</div>
""",
    unsafe_allow_html=True,
)

st.divider()


# =========================================================
# TABS
# =========================================================
st.subheader("Composition et poches")

tab_compo, tab_obli, tab_core, tab_def = st.tabs(
    [
        "Composition du portefeuille",
        "Poche obligataire (50%)",
        "Poche actions core",
        "Poche actions défense/refuge",
    ]
)


# =========================================================
# TAB COMPOSITION (table)
# =========================================================
with tab_compo:
    if df.empty:
        st.warning(
            "Composition introuvable. Sélectionne un profil depuis la page Choix "
            "(ou vérifie Projet/portefeuille_equilibre.csv)."
        )
    else:
        df_show = df.copy()
        df_show["Poids"] = format_weights_percent(df_show["Poids"])

        # ordre agréable : Bucket puis poids desc si possible
        # (on trie sur poids numérique brut)
        w_num = pd.to_numeric(df["Poids"], errors="coerce").fillna(0.0)
        mx = float(w_num.max()) if len(w_num) else 0.0
        w_frac = w_num if mx <= 1.0 + 1e-12 else (w_num / 100.0)
        df_show["_w"] = w_frac
        df_show = df_show.sort_values(["Bucket", "_w"], ascending=[True, False]).drop(columns=["_w"]).reset_index(drop=True)

        html = """
<div class="custom-table-wrap">
<table class="custom-table">
<thead>
<tr>
<th>Classe d'actif</th>
<th>Nom du fonds</th>
<th>Code ISIN</th>
<th style="text-align:right;">Pondération</th>
</tr>
</thead>
<tbody>
"""
        for _, r in df_show.iterrows():
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
    st.markdown(
        f"""
<div class="card">
  <h4>Obligataire (socle)</h4>
  <p>
    Objectif : stabiliser le portefeuille, porter du rendement via crédit/IG/HY, et amortir les phases de stress.
    <br><br>
    <b>Garde-fou :</b> chaque ligne reste plafonnée à {int(MAX_LINE_WEIGHT*100)}%.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

with tab_core:
    st.markdown(
        f"""
<div class="card">
  <h4>Actions Core</h4>
  <p>
    Objectif : capter la prime actions avec diversification géographique/style.
    <br><br>
    <b>Tilt Momentum</b> appliqué entre les fonds Core (intensité {MOM_TILT_STRENGTH}).
    <br>
    <b>Min par fonds</b> : {int(CORE_MIN_PER_FUND*100)}% (si compatible avec le budget Core).
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

with tab_def:
    st.markdown(
        f"""
<div class="card">
  <h4>Actions Défense / Refuge</h4>
  <p>
    Objectif : renforcer la résilience en stress (sécurité/cyber + or).
    <br><br>
    <b>Pilotage</b> : le poids total Défense varie entre {int(DEF_MIN_W*100)}% et {int(DEF_MAX_W*100)}%
    selon la dynamique (momentum moyen), puis les poids sont tilté intra-poche (intensité {DEF_TILT_STRENGTH}).
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# NAV BUTTONS
# =========================================================
st.divider()
btn_left, btn_right = st.columns([1, 1], gap="large")
with btn_left:
    if st.button("Analyse rétrospective", type="primary", use_container_width=True):
        st.switch_page("pages/Dash_equi.py")
with btn_right:
    if st.button("Analyse prospective", type="primary", use_container_width=True):
        st.switch_page("pages/ExAnte.py")