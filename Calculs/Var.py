import pandas as pd
import numpy as np

from scipy.stats import norm, skew, kurtosis

# =========================================================
# 1) CONFIGURATION DU PORTEFEUILLE (TES POIDS RÉELS)
# =========================================================
# 10 fonds Oblig + 3 fonds Défense à 5% | 8 fonds Core à 4.375%
weights = {
    # Obligataire & Défense (13 fonds)
    "LU1161527038": 0.05, "FR0011288513": 0.05, "LU1897613763": 0.05, 
    "LU1191877379": 0.05, "LU0075938133": 0.05, "LU0616241476": 0.05, 
    "LU0249332619": 0.05, "LU0336083497": 0.05, "LU1472740767": 0.05, 
    "FR0013531266": 0.05, "LU0270904781": 0.05, "LU1951225553": 0.05, "FR0010664086": 0.05,
    # Actions Core (8 fonds)
    "LU0072462186": 0.04375, "LU0171296865": 0.04375, "LU1883854199": 0.04375, 
    "LU1244893696": 0.04375, "LU0217576759": 0.04375, "FR0000284689": 0.04375, 
    "LU1123620707": 0.04375, "LU0996179007": 0.04375
}

# =========================================================
# 2) GÉNÉRATION DES DONNÉES DE MARCHÉ (SIMULATION)
# =========================================================
np.random.seed(42)
n_jours = 1260 # ~5 ans de données
isins = list(weights.keys())

# Simulation de rendements corrélés pour les fonds
# (En situation réelle, tu ferais : returns_df = pd.read_csv('tes_prix.csv').pct_change())
data = np.random.normal(0.0002, 0.01, (n_jours, len(isins))) 
returns_df = pd.DataFrame(data, columns=isins)

# Calcul du rendement quotidien du portefeuille (Rp = Somme des w_i * r_i)
port_returns = returns_df.dot(pd.Series(weights))

# =========================================================
# 3) FONCTION DE CALCUL DES VaR (95%)
# =========================================================
def backtest_var(returns, alpha=0.95, window=252):
    df = pd.DataFrame(returns, columns=['Returns'])
    lmd = 0.94 # Lambda pour EWMA
    
    # --- 1. VaR EWMA (Réactive) ---
    v_t = returns.var()
    ewma_vols = []
    for r in returns:
        v_t = lmd * v_t + (1 - lmd) * (r**2)
        ewma_vols.append(np.sqrt(v_t))
    df['VaR_EWMA'] = abs(norm.ppf(1-alpha) * pd.Series(ewma_vols).shift(1))

    # --- 2. Fenêtre Glissante (Param, Hist, Cornish-Fisher) ---
    for i in range(window, len(returns)):
        sub = returns.iloc[i-window:i]
        mu, sigma = sub.mean(), sub.std()
        
        # Paramétrique (Normale)
        df.loc[df.index[i], 'VaR_Param'] = abs(mu + norm.ppf(1-alpha) * sigma)
        
        # Historique (Quantile)
        df.loc[df.index[i], 'VaR_Hist'] = abs(sub.quantile(1-alpha))
        
        # Cornish-Fisher (Ajustée Skewness/Kurtosis)
        s, k = skew(sub), kurtosis(sub)
        z = norm.ppf(1-alpha)
        z_cf = z + (z**2-1)*s/6 + (z**3-3*z)*k/24 - (2*z**3-5*z)*(s**2)/36
        df.loc[df.index[i], 'VaR_CF'] = abs(mu + z_cf * sigma)
        
    return df.dropna()

# =========================================================
# 4) ANALYSE ET VERDICT
# =========================================================
df_final = backtest_var(port_returns)

print(f"{'Méthode':<20} | {'Exceptions':<10} | {'% Exceptions':<10}")
print("-" * 50)

summary = {}
for col in ['VaR_EWMA', 'VaR_Param', 'VaR_Hist', 'VaR_CF']:
    # Une exception : Perte réelle > VaR prédite
    # (On compare Returns < -VaR car nos VaR sont en valeurs absolues positives)
    exceptions = df_final[df_final['Returns'] < -df_final[col]]
    nb = len(exceptions)
    pct = (nb / len(df_final)) * 100
    summary[col] = pct
    print(f"{col:<20} | {nb:<10} | {pct:.2f}%")

