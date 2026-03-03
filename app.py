import streamlit as st
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="VSOF Gestion Privée — Espace Conseiller",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# CSS INSTITUTIONNEL SOBRE
# =========================================================
st.markdown("""
<style>
:root{
--bg:#0b1220;
--panel:rgba(255,255,255,0.04);
--stroke:rgba(255,255,255,0.10);
--text:rgba(255,255,255,0.92);
--muted:rgba(255,255,255,0.60);
--brand1:#2a76ff;
--brand2:#00d2ff;
}

.stApp{
background:
radial-gradient(1200px 600px at 15% 5%, rgba(42,118,255,0.20), transparent 60%),
linear-gradient(180deg, var(--bg), #060a14 70%);
color:var(--text);
}

/* Centre verticalement */
.main-wrapper{
min-height: 20vh;
display:flex;
flex-direction:column;
justify-content:center;
}

.hero{
text-align:center;
margin-bottom:60px;
}

.hero-title{
font-size:48px;
font-weight:900;
background:-webkit-linear-gradient(45deg, var(--brand1), var(--brand2));
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:10px;
}

.hero-sub{
color:var(--muted);
font-size:16px;
}

.card{
border:1px solid var(--stroke);
background:var(--panel);
border-radius:20px;
padding:28px;
height:220px;
display:flex;
flex-direction:column;
justify-content:space-between;
}

/* On enlève totalement le hover */
.card:hover{
transform:none;
border:1px solid var(--stroke);
background:var(--panel);
}

.card-title{
font-size:20px;
font-weight:800;
margin-bottom:10px;
}

.card-text{
font-size:14px;
color:var(--muted);
line-height:1.5;
}

.stButton > button{
border-radius:10px;
font-weight:700;
height:46px;
border:1px solid rgba(255,255,255,0.15);
}

.footer{
text-align:center;
color: rgba(255,255,255,0.40);
font-size: 12px;
margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONTENU CENTRÉ
# =========================================================
today = datetime.now().strftime("%d/%m/%Y")

st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# HERO
st.markdown(f"""
<div class="hero">
<div class="hero-title">VSOF Gestion Privée</div>
<div class="hero-sub">
Espace Conseiller — Audit patrimonial et allocation d’actifs<br>
Version démonstration • {today}
</div>
</div>
""", unsafe_allow_html=True)

# BLOCS CENTRÉS
_, col1, col2, _ = st.columns([1.5, 4, 4, 1.5], gap="large")

with col1:
    st.markdown("""
<div class="card">
<div>
<div class="card-title">Votre Portefeuille</div>
<div class="card-text">
Analyse ex-post des performances, allocation, volatilité et comparaison au benchmark.
</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:-55px;'>", unsafe_allow_html=True)
    if st.button("Accéder à l’audit", use_container_width=True):
        st.switch_page("pages/dashboard.py")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="card">
<div>
<div class="card-title">Profils d’investissement</div>
<div class="card-text">
Construction d’allocations recommandées selon le profil de risque du client.
</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:-55px;'>", unsafe_allow_html=True)
    if st.button("Voir les profils", use_container_width=True, type="primary"):
        st.switch_page("pages/choix.py")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.markdown("<div class='footer'>© VSOF Gestion Privée — Plateforme CGP</div>", unsafe_allow_html=True)