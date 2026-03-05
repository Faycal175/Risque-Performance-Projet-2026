# Portfolio Management & Risk Dashboard  
### Master 272 — Risque & Performance (Université Paris-Dauphine)

Ce projet a été réalisé dans le cadre du cours **Risque & Performance du Master 272**.  

L'objectif est de développer un **outil d'analyse et de gestion de portefeuille d'investisseur**, permettant :

- la construction et le backtest d'un portefeuille équilibré
- l'analyse de performance vs benchmark
- l'étude de l'allocation et des corrélations
- la mesure du risque ex-ante
- la simulation de scénarios futurs

L'application est développée sous forme de **dashboard interactif avec Streamlit**.

## Application en ligne

Une version de l'application est accessible en ligne à l'adresse suivante :

https://risque-performance-projet-2026.onrender.com

Cependant, certaines fonctionnalités peuvent ne pas être disponibles. En effet, l'application est hébergée sur la version gratuite de Render, qui impose des limitations de mémoire et de ressources.

Ces contraintes peuvent empêcher l'exécution de certains calculs plus lourds (simulations, analyses complètes, etc.).

Pour une expérience complète et pour accéder à toutes les fonctionnalités de l'application, nous vous recommandons de lancer l'application localement en suivant les instructions décrites dans la section située en bas de ce document.

---

# Objectifs du projet

L'outil permet de reproduire plusieurs analyses utilisées en **gestion d'actifs** et **gestion de portefeuille**, notamment :

- analyse de performance
- décomposition du risque
- corrélations entre actifs
- contribution au risque
- Value-at-Risk
- projection Monte-Carlo

L'application est conçue comme un **outil d'aide à la décision pour un investisseur**.

---

# Architecture de l'application

L'application est organisée en plusieurs modules :

### 1️ Dashboard rétrospectif
Analyse historique du portefeuille :

- performance vs benchmark
- rendement annuel
- volatilité glissante
- allocation du portefeuille
- matrice de corrélation

### 2️ Analyse Ex-Ante du risque
Mesure du risque futur du portefeuille :

- volatilité ex-ante (Markowitz)
- espérance de rendement
- corrélation moyenne
- contribution au risque
- VaR / CVaR
- bêta vs benchmark


---

# Stratégie d'investissement


Structure :

- **50% Obligations**
- **50% Actions**

La poche actions est divisée entre :

- **Core**
- **Actifs Défense**

---

# Stratégie Momentum

Une **stratégie momentum** est utilisée pour ajuster les pondérations.

Le signal momentum est calculé comme :

Momentum Score = Momentum(12 mois) − Momentum(1 mois)

avec :

- Momentum 12 mois ≈ rendement sur 252 jours
- Momentum 1 mois ≈ rendement sur 21 jours

Le signal est utilisé pour :

- tilting des poids du **core actions**
- ajustement de la poche **défensive**

Le portefeuille est **rebalancé annuellement**.

---

## Arborescence du projet

```text
Projet_Perff/
├─ app.py
├─ Calculs/
│  ├─ __init__.py
│  ├─ eq_engine.py
│  ├─ ptf_client_engine.py
│  ├─ ptf_equilibre_engine.py
│  
├─ pages/
│  ├─ choix.py
│  ├─ Dash_equi.py
│  ├─ dashboard.py
│  ├─ DetailFond.py
│  └─ ExAnte.py
└─ Projet/
   ├─ Indices/
   ├─ Data.xlsb
   ├─ UniversInvestissement.xlsb
   ├─ Univers-Prix.parquet
   ├─ forex.parquet
   ├─ portefeuille_equilibre.csv
   ├─ Données historiques EAGG(1).csv
   └─ Sujet_v1.pdf
```

---

# Installation

Cloner le repository :

```bash
git clone https://github.com/Faycal175/Risque-Performance-Projet-2026.git
cd Risque-Performance-Projet-2026

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

# Lancement
```bash
streamlit run app.py
```

