import streamlit as st
from datetime import datetime


st.set_page_config(
    page_title="VSOF Gestion Privée — Espace Conseiller",
    page_icon="🏦",
    layout="wide",
)

today = datetime.now().strftime("%d/%m/%Y")


st.markdown("""
<style>

:root{
--bg:#070b14;
--text:rgba(255,255,255,0.92);
--muted:rgba(255,255,255,0.65);
--brand1:#2a76ff;
--brand2:#00d2ff;
}

.stApp{
background:
radial-gradient(1200px 620px at 10% 8%, rgba(42,118,255,0.22), transparent 60%),
radial-gradient(1000px 520px at 90% 12%, rgba(0,210,255,0.14), transparent 55%),
linear-gradient(180deg, var(--bg), #050711 70%, #03040b 100%);
color:var(--text);
}

.brand-title{
font-size:58px;
font-weight:950;
background: linear-gradient(45deg,var(--brand1),var(--brand2));
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:10px;
text-align:center;
}

.hero-title{
font-size:34px;
font-weight:900;
text-align:center;
margin-bottom:10px;
}

.hero-sub{
color:var(--muted);
font-size:15px;
line-height:1.6;
text-align:center;
}

.hero-chips{
margin-top:180px;
display:flex;
justify-content:center;
gap:10px;
}

.chip{
display:flex;
align-items:center;
gap:8px;
padding:6px 12px;
border-radius:999px;
border:1px solid rgba(255,255,255,0.14);
background:rgba(255,255,255,0.04);
font-size:12px;
font-weight:900;
}

.dot{
width:8px;
height:8px;
border-radius:50%;
background:#2a76ff;
}

.buttons{
margin-top:80px;
}

.stButton > button{
height:50px;
border-radius:14px;
font-weight:900;
border:1px solid rgba(255,255,255,0.14);
background:rgba(255,255,255,0.04);
}

.footer{
margin-top:40px;
font-size:12px;
text-align:center;
color:rgba(255,255,255,0.4);
}

</style>
""", unsafe_allow_html=True)


left, center, right = st.columns([2,3,2])

with center:


    st.markdown('<div class="brand-title">VSOF Gestion Privée</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Plateforme Conseiller</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hero-sub">
    Audit patrimonial, allocation d’actifs et portefeuilles modèles.<br>
    Version démonstration • {today}
    </div>
    """, unsafe_allow_html=True)

  

    st.markdown('<div class="buttons">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Résumé de votre Portefeuille", use_container_width=True):
            st.switch_page("pages/dashboard.py")

    with c2:
        if st.button("Nos Portefeuilles", use_container_width=True, type="primary"):
            st.switch_page("pages/choix.py")

    st.markdown("</div>", unsafe_allow_html=True)

 
    # FOOTER

    st.markdown(
        "<div class='footer'>© VSOF Gestion Privée — Plateforme CGP</div>",
        unsafe_allow_html=True
    )